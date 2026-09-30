#!/usr/bin/env python3
"""Fail-closed preflight/artifact checks for the GR00T N1.7 UNITREE_G1_SONIC pack.

This script is the read-only half of scripts/groot-unitree-dex3-sonic.sh: the
shell wrapper decides *what* to launch, this decides whether the inputs that
launch depends on are actually the pinned ones.  It only ever reads, and it is
stdlib-only so that it runs both under /opt/venvs/groot-n17/bin/python and
under a bare python3 in the shell tests.

The contract it enforces is the stock UNITREE_G1_SONIC modality config from the
pinned source (commit 1a1837f20538b7d7e21f977a11a5aee14f99803c):

  gr00t/configs/data/embodiment_configs.py  -> "unitree_g1_sonic"
  gr00t/data/embodiment_tags.py             -> EmbodimentTag.UNITREE_G1_SONIC
  gr00t/data/dataset/lerobot_episode_loader.py -> meta/*.json|jsonl requirements
  gr00t/data/stats.py                       -> meta/stats.json + relative_stats.json

so this file hard-codes the key list/dims instead of importing gr00t: importing
the package would pull torch/torchcodec into a check that must be able to run
before the GPU stack is up, and a drifted import would silently weaken the
check it is supposed to be.

Exit codes: 0 = all requested checks passed, 2 = at least one check failed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# --- stock UNITREE_G1_SONIC contract (pinned source) -------------------------
EMBODIMENT_TAG = "UNITREE_G1_SONIC"

# Official SONIC exporter storage layout, which the pack preserves: the measured
# body lives in `observation.state` with the hands sitting next to their arms
# (left_arm, left_hand, right_arm, right_hand), gravity has its own column and
# the SONIC action fields stay separate.  The model instead consumes the
# *registered* key order below, so every modality group declares the column and
# slice it reads and the model-facing 46D/78D is the concatenation in registered
# order.
#
# Storage order is not registered order: left_hand and right_arm are swapped
# between the two.  A pack that stored the body in registered order while
# declaring these slices would silently feed hand channels to an arm, which is
# exactly what this table refuses.
STATE_COLUMN = "observation.state"
STATE_STORAGE_DIM = 43
GRAVITY_COLUMN = "observation.projected_gravity"
# Keyed and iterated in the *registered* order the model consumes
# (left_arm, right_arm, left_hand, right_hand), while each value is the storage
# slice in `observation.state` (left_arm, left_hand, right_arm, right_hand).
# The two orders differ on purpose: the hands sit next to their arms in the
# exporter's storage and the model re-concatenates in registered order.
STATE_EXPECTATIONS: dict[str, tuple[str, tuple[int, int]]] = {
    "left_leg": (STATE_COLUMN, (0, 6)),
    "right_leg": (STATE_COLUMN, (6, 12)),
    "waist": (STATE_COLUMN, (12, 15)),
    "left_arm": (STATE_COLUMN, (15, 22)),
    "right_arm": (STATE_COLUMN, (29, 36)),
    "left_hand": (STATE_COLUMN, (22, 29)),
    "right_hand": (STATE_COLUMN, (36, 43)),
    "projected_gravity": (GRAVITY_COLUMN, (0, 3)),
}
ACTION_EXPECTATIONS: dict[str, tuple[str, tuple[int, int]]] = {
    "motion_token": ("action.motion_token", (0, 64)),
    "left_hand_joints": ("teleop.left_hand_joints", (0, 7)),
    "right_hand_joints": ("teleop.right_hand_joints", (0, 7)),
}
# The loader defaults an omitted original_key to these columns.
DEFAULT_COLUMNS = {"state": STATE_COLUMN, "action": "action"}

# Registered key iteration order the model consumes; the dims are the
# concatenation of that order's storage slices (46D state / 78D action).
STOCK_STATE_KEYS = list(STATE_EXPECTATIONS)
STOCK_ACTION_KEYS = list(ACTION_EXPECTATIONS)
STATE_DIM = sum(end - start for _column, (start, end) in STATE_EXPECTATIONS.values())
ACTION_DIM = sum(end - start for _column, (start, end) in ACTION_EXPECTATIONS.values())

STOCK_VIDEO_KEYS = ["ego_view"]
STOCK_LANGUAGE_KEY = "human.task_description"
ACTION_HORIZON = 40
DATASET_FPS = 50
LEROBOT_CODEBASE_VERSION = "v2.1"

# training.optim as OmegaConf writes it under the top-level training: mapping.
TRAIN_OPTIM_RE = re.compile(r"^  optim:\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)

STAT_FIELDS = ("mean", "std", "min", "max", "q01", "q99")
STATS_FINGERPRINTS_KEY = "__fingerprints__"


def required_stats_columns() -> dict[str, int]:
    """Columns the loader reads (via each group's original_key) and their widths."""
    required: dict[str, int] = {}
    for column, (_start, end) in [*STATE_EXPECTATIONS.values(), *ACTION_EXPECTATIONS.values()]:
        required[column] = max(required.get(column, 0), end)
    return required


REQUIRED_STATS_COLUMNS = required_stats_columns()

# Files the official stats writer may leave behind when it is killed mid-flush.
PARTIAL_MARKER_PATTERNS = (
    "*.tmp",
    "*.partial",
    "*.writing",
    ".incomplete",
    ".partial",
    "INCOMPLETE",
    "PARTIAL",
)
META_JSON_FILES = (
    "info.json",
    "modality.json",
    "stats.json",
    "relative_stats.json",
    "episodes.jsonl",
    "tasks.jsonl",
)
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class Report:
    """Accumulates check results so one run reports every problem at once."""

    def __init__(self, quiet: bool = False) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.records: dict[str, object] = {}
        self.quiet = quiet

    def ok(self, message: str) -> None:
        if not self.quiet:
            print(f"PASS: {message}")

    def fail(self, message: str) -> None:
        self.errors.append(message)
        if not self.quiet:
            print(f"FAIL: {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        if not self.quiet:
            print(f"WARN: {message}")

    def record(self, key: str, value: object) -> None:
        self.records[key] = value

    def check(self, condition: bool, message: str) -> bool:
        if condition:
            self.ok(message)
        else:
            self.fail(message)
        return condition


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, report: Report, label: str) -> object | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        report.fail(f"{label} is missing: {path}")
    except (json.JSONDecodeError, OSError) as exc:
        report.fail(f"{label} is unreadable: {path}: {exc}")
    return None


def load_jsonl(path: Path, report: Report, label: str) -> list[dict] | None:
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    report.fail(f"{label} line {lineno} is not JSON: {path}: {exc}")
                    return None
    except FileNotFoundError:
        report.fail(f"{label} is missing: {path}")
        return None
    except OSError as exc:
        report.fail(f"{label} is unreadable: {path}: {exc}")
        return None
    return rows


def check_partial_markers(root: Path, report: Report, label: str) -> None:
    """A leftover tmp/partial file means a previous writer died mid-write.

    The official stats writer flushes through ``tempfile.NamedTemporaryFile``
    with a dot-prefixed name (``.stats.json.<random>.tmp``), and pathlib globs
    skip dotfiles, so the scan walks the tree and matches names instead.
    """
    markers: list[str] = []
    for directory, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if any(fnmatch.fnmatch(filename, pattern) for pattern in PARTIAL_MARKER_PATTERNS):
                markers.append(str(Path(directory) / filename))
    # Zero-byte metadata is the other shape of a torn write.
    meta = root / "meta"
    for name in META_JSON_FILES:
        path = meta / name
        if path.is_file() and path.stat().st_size == 0:
            markers.append(f"{path} (0 bytes)")
    if markers:
        report.fail(f"{label} has stale partial markers: {', '.join(sorted(markers))}")
    else:
        report.ok(f"{label} has no stale partial markers")


def check_split_root(
    split_root: Path,
    report: Report,
    label: str,
    *,
    ego_key: str,
    expected_fps: int,
    want_contract: bool = True,
    want_stats: bool = True,
) -> None:
    if not split_root.is_dir():
        report.fail(f"{label} does not exist: {split_root}")
        return
    report.ok(f"{label} exists: {split_root}")
    check_partial_markers(split_root, report, label)

    meta = split_root / "meta"
    info = load_json(meta / "info.json", report, f"{label} meta/info.json")
    modality = load_json(meta / "modality.json", report, f"{label} meta/modality.json")
    info_dict = info if isinstance(info, dict) else {}
    modality_dict = modality if isinstance(modality, dict) else {}

    if want_contract:
        episodes = load_jsonl(meta / "episodes.jsonl", report, f"{label} meta/episodes.jsonl")
        tasks = load_jsonl(meta / "tasks.jsonl", report, f"{label} meta/tasks.jsonl")
        if isinstance(info, dict):
            check_info(info, report, label, expected_fps=expected_fps)
        if isinstance(modality, dict):
            check_modality(modality, info_dict, report, label, ego_key=ego_key)
            check_group_layout(
                modality, report, label, "state", STATE_EXPECTATIONS, STATE_DIM, info=info_dict
            )
            check_group_layout(
                modality, report, label, "action", ACTION_EXPECTATIONS, ACTION_DIM, info=info_dict
            )
        check_episodes(episodes, info_dict, report, label)
        check_tasks(tasks, report, label)
        check_payload_files(split_root, info_dict, report, label)

    if want_stats:
        stats = load_json(meta / "stats.json", report, f"{label} meta/stats.json")
        relative_stats = load_json(
            meta / "relative_stats.json", report, f"{label} meta/relative_stats.json"
        )
        if isinstance(stats, dict):
            check_stats(stats, report, label, required_columns=REQUIRED_STATS_COLUMNS)
        check_relative_stats(
            relative_stats if isinstance(relative_stats, dict) else None,
            stats if isinstance(stats, dict) else None,
            report,
            label,
        )


def check_info(info: dict, report: Report, label: str, *, expected_fps: int) -> None:
    version = info.get("codebase_version")
    report.check(
        version == LEROBOT_CODEBASE_VERSION,
        f"{label} info.json codebase_version is {version!r} (expected {LEROBOT_CODEBASE_VERSION!r})",
    )
    fps = info.get("fps")
    report.check(fps == expected_fps, f"{label} info.json fps is {fps!r} (expected {expected_fps})")

    total_episodes = info.get("total_episodes")
    report.check(
        isinstance(total_episodes, int) and total_episodes > 0,
        f"{label} info.json total_episodes is {total_episodes!r} (must be > 0)",
    )
    total_frames = info.get("total_frames")
    report.check(
        isinstance(total_frames, int) and total_frames > 0,
        f"{label} info.json total_frames is {total_frames!r} (must be > 0)",
    )
    chunks_size = info.get("chunks_size")
    report.check(
        isinstance(chunks_size, int) and chunks_size > 0,
        f"{label} info.json chunks_size is {chunks_size!r} (must be > 0)",
    )
    for key in ("data_path", "video_path"):
        pattern = info.get(key)
        report.check(
            isinstance(pattern, str) and "{episode_index" in pattern,
            f"{label} info.json {key} is a usable pattern ({pattern!r})",
        )

    features = info.get("features")
    if not isinstance(features, dict):
        report.fail(f"{label} info.json has no features mapping")
        return
    report.record(
        f"{label}_total_episodes", total_episodes if isinstance(total_episodes, int) else None
    )
    report.record(f"{label}_total_frames", total_frames if isinstance(total_frames, int) else None)


def check_group_layout(
    modality: dict,
    report: Report,
    label: str,
    group: str,
    expectations: dict[str, tuple[str, tuple[int, int]]],
    expected_dim: int,
    *,
    info: dict,
) -> None:
    """Require the official SONIC storage layout for one modality group.

    Each registered key must declare the exact column and slice it reads.  The
    slices are validated individually -- not for contiguity in key iteration
    order -- because storage order (hands next to their arms) differs from the
    registered order the model consumes.
    """
    section = modality.get(group)
    if not isinstance(section, dict):
        report.fail(f"{label} modality.json has no {group} section")
        return
    keys = list(section)
    expected_keys = list(expectations)
    report.check(
        keys == expected_keys,
        f"{label} {group} keys are {keys} (expected the registered order {expected_keys})",
    )

    features = info.get("features", {}) if isinstance(info, dict) else {}
    default_column = DEFAULT_COLUMNS[group]
    declared_dims: list[int] = []
    slices_ok = keys == expected_keys
    for key, (column, (start, end)) in expectations.items():
        entry = section.get(key)
        if not isinstance(entry, dict):
            report.fail(f"{label} {group}.{key} is missing")
            slices_ok = False
            continue
        declared = (entry.get("start"), entry.get("end"))
        if not isinstance(declared[0], int) or not isinstance(declared[1], int):
            report.fail(f"{label} {group}.{key} has no integer slice: {declared!r}")
            slices_ok = False
            continue
        declared_dims.append(declared[1] - declared[0])
        if declared != (start, end):
            slices_ok = False
        report.check(
            declared == (start, end),
            f"{label} {group}.{key} slices {declared[0]}:{declared[1]} "
            f"(expected {start}:{end} of {column!r}, the official storage slice)",
        )
        original_key = entry.get("original_key", default_column)
        report.check(
            original_key == column,
            f"{label} {group}.{key} reads {original_key!r} (expected {column!r})",
        )
        feature = features.get(column)
        if not isinstance(feature, dict):
            report.fail(f"{label} info.json does not declare the column {column!r}")
            slices_ok = False
            continue
        shape = feature.get("shape")
        width = shape[0] if isinstance(shape, list) and shape else None
        report.check(
            isinstance(width, int) and width >= end,
            f"{label} {column} is {width!r} wide (group {group}.{key} reads {start}:{end})",
        )
        dtype = feature.get("dtype")
        report.check(
            isinstance(dtype, str) and dtype.startswith("float"),
            f"{label} {column} dtype is {dtype!r} (expected a float dtype)",
        )

    report.check(
        sum(declared_dims) == expected_dim,
        f"{label} {group} declared slices sum to {sum(declared_dims)} dims "
        f"(the model-facing concatenation must be {expected_dim}D)",
    )
    if slices_ok:
        report.ok(
            f"{label} {group} matches the official SONIC storage layout; "
            f"registered-order concatenation is {expected_dim}D"
        )


def check_modality(
    modality: dict, info: dict, report: Report, label: str, *, ego_key: str
) -> None:
    video = modality.get("video")
    if not isinstance(video, dict):
        report.fail(f"{label} modality.json has no video section")
    else:
        report.check(
            sorted(video) == sorted(STOCK_VIDEO_KEYS),
            f"{label} video keys are {sorted(video)} (expected exactly {STOCK_VIDEO_KEYS})",
        )
        features = info.get("features", {}) if isinstance(info, dict) else {}
        for key, entry in video.items():
            original_key = entry.get("original_key", f"observation.images.{key}")
            report.check(
                original_key in features,
                f"{label} video key {key!r} maps to a declared feature ({original_key})",
            )

    annotation = modality.get("annotation")
    if not isinstance(annotation, dict) or STOCK_LANGUAGE_KEY not in annotation:
        report.fail(
            f"{label} modality.json is missing annotation.{STOCK_LANGUAGE_KEY} "
            "(the stock SONIC language key)"
        )
    else:
        report.ok(f"{label} language key annotation.{STOCK_LANGUAGE_KEY} is present")


def check_stats(
    stats: dict,
    report: Report,
    label: str,
    required_columns: dict[str, int] | None = None,
) -> None:
    """Validate the statistics the loader will read.

    ``get_dataset_statistics`` looks up every modality group by its resolved
    ``original_key`` and slices ``[start:end]`` out of that column's stat
    vectors, so each of the five storage columns needs its six vectors, wide
    enough for the slice the groups read from it.
    """
    if not required_columns:
        required_columns = REQUIRED_STATS_COLUMNS
    for column, required_width in sorted(required_columns.items()):
        entry = stats.get(column)
        if not isinstance(entry, dict):
            report.fail(
                f"{label} meta/stats.json has no {column} entry "
                "(regenerate with gr00t/data/stats.py)"
            )
            continue
        missing = [field for field in STAT_FIELDS if field not in entry]
        if missing:
            report.fail(f"{label} stats.{column} is missing fields {missing}")
            continue
        bad = [
            field
            for field in STAT_FIELDS
            if not isinstance(entry[field], list) or len(entry[field]) < required_width
        ]
        report.check(
            not bad,
            f"{label} stats.{column} has {required_width}+-wide {'/'.join(STAT_FIELDS)} vectors"
            + (f" (bad: {bad})" if bad else ""),
        )

    fingerprints = stats.get(STATS_FINGERPRINTS_KEY)
    if not isinstance(fingerprints, dict):
        report.fail(
            f"{label} meta/stats.json has no {STATS_FINGERPRINTS_KEY} sidecar; "
            "it was not produced by the pinned gr00t/data/stats.py"
        )
        return
    missing_fp = [column for column in required_columns if column not in fingerprints]
    report.check(
        not missing_fp,
        f"{label} stats fingerprints cover {sorted(required_columns)}"
        + (f" (missing: {missing_fp})" if missing_fp else ""),
    )
    report.record(f"{label}_stats_fingerprints", fingerprints)


def check_relative_stats(
    relative_stats: dict | None, stats: dict | None, report: Report, label: str
) -> None:
    if relative_stats is None:
        return
    report.ok(f"{label} meta/relative_stats.json is present and parses")
    if STATS_FINGERPRINTS_KEY not in relative_stats:
        report.warn(
            f"{label} meta/relative_stats.json has no {STATS_FINGERPRINTS_KEY} sidecar; "
            "confirm it came from the pinned gr00t/data/stats.py"
        )
    # The stock SONIC action configs are all ABSOLUTE, so stats.py legitimately
    # writes only the fingerprint sidecar here.  Anything else is a surprise.
    extra = [key for key in relative_stats if key != STATS_FINGERPRINTS_KEY]
    if extra and not stats:
        report.warn(f"{label} relative_stats has entries {extra} while stats.json is empty")


def check_episodes(
    episodes: list[dict] | None, info: dict, report: Report, label: str
) -> None:
    if episodes is None:
        return
    report.check(len(episodes) > 0, f"{label} meta/episodes.jsonl lists {len(episodes)} episodes")
    declared = info.get("total_episodes") if isinstance(info, dict) else None
    if isinstance(declared, int):
        report.check(
            len(episodes) == declared,
            f"{label} episodes.jsonl count {len(episodes)} matches info.json total_episodes {declared}",
        )
    lengths: list[int] = []
    malformed = 0
    for row in episodes:
        length = row.get("length")
        if not isinstance(length, int) or length <= 0:
            malformed += 1
        else:
            lengths.append(length)
    report.check(malformed == 0, f"{label} every episode has a positive length ({malformed} bad)")
    short = [n for n, length in enumerate(lengths) if length < ACTION_HORIZON]
    if short:
        report.fail(
            f"{label} has {len(short)} episode(s) shorter than the {ACTION_HORIZON}-step action "
            f"horizon (first: {short[:5]}); those episodes cannot produce a training window"
        )
    elif lengths:
        report.ok(
            f"{label} episode lengths are >= {ACTION_HORIZON} "
            f"(min {min(lengths)}, max {max(lengths)}, total {sum(lengths)} frames)"
        )
        report.record(f"{label}_episode_count", len(lengths))
        report.record(f"{label}_frame_count", sum(lengths))


def check_tasks(tasks: list[dict] | None, report: Report, label: str) -> None:
    if tasks is None:
        return
    report.check(len(tasks) > 0, f"{label} meta/tasks.jsonl lists {len(tasks)} tasks")


def check_payload_files(split_root: Path, info: dict, report: Report, label: str) -> None:
    parquet_files = sorted((split_root / "data").rglob("*.parquet")) if (split_root / "data").is_dir() else []
    report.check(len(parquet_files) > 0, f"{label} ships {len(parquet_files)} parquet episode file(s)")
    video_files = sorted((split_root / "videos").rglob("*.mp4")) if (split_root / "videos").is_dir() else []
    report.check(len(video_files) > 0, f"{label} ships {len(video_files)} mp4 video file(s)")

    declared = info.get("total_episodes") if isinstance(info, dict) else None
    if isinstance(declared, int) and parquet_files:
        report.check(
            len(parquet_files) == declared,
            f"{label} parquet file count {len(parquet_files)} matches total_episodes {declared}",
        )
    pointers = []
    for path in parquet_files[:8]:
        try:
            with open(path, "rb") as handle:
                if handle.read(len(GIT_LFS_POINTER_PREFIX)) == GIT_LFS_POINTER_PREFIX:
                    pointers.append(str(path))
        except OSError as exc:
            report.fail(f"{label} cannot read {path}: {exc}")
    if pointers:
        report.fail(
            f"{label} has unresolved Git LFS pointer(s) instead of episode data: {pointers}"
        )
    report.record(f"{label}_parquet_files", len(parquet_files))
    report.record(f"{label}_video_files", len(video_files))


def check_dataset_splits(
    dataset_root: Path,
    report: Report,
    splits: list[str],
    train_id: str,
    val_id: str,
    *,
    ego_key: str,
    expected_fps: int,
    want_contract: bool,
    want_stats: bool,
) -> None:
    ids = {"train": train_id, "val": val_id}
    for split in splits:
        check_split_root(
            dataset_root / ids[split],
            report,
            f"{split} split",
            ego_key=ego_key,
            expected_fps=expected_fps,
            want_contract=want_contract,
            want_stats=want_stats,
        )


def check_model(
    model_dir: Path,
    report: Report,
    *,
    expected_repo: str,
    expected_revision: str,
) -> None:
    if not model_dir.is_dir():
        report.fail(f"base model directory is missing: {model_dir}")
        return
    report.ok(f"base model directory exists: {model_dir}")

    config = load_json(model_dir / "config.json", report, "base model config.json")
    if isinstance(config, dict):
        model_type = config.get("model_type")
        report.check(
            model_type == "Gr00tN1d7",
            f"base model model_type is {model_type!r} (expected 'Gr00tN1d7')",
        )
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        path = model_dir / name
        report.check(path.is_file(), f"base model ships {name}")
    weights = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
    report.check(len(weights) > 0, f"base model ships weight shard(s) ({len(weights)} found)")

    provenance = load_json(model_dir / "MODEL_PROVENANCE.json", report, "base model provenance")
    if isinstance(provenance, dict):
        repo = provenance.get("repo")
        revision = provenance.get("revision")
        report.check(
            repo == expected_repo,
            f"base model provenance repo is {repo!r} (expected {expected_repo!r})",
        )
        report.check(
            revision == expected_revision,
            f"base model provenance revision is {revision!r} (expected {expected_revision!r})",
        )
        report.record(
            "model_provenance",
            {key: provenance.get(key) for key in ("repo", "revision", "variant", "fetched_at_utc")},
        )


def check_source(source_dir: Path, report: Report, *, expected_commit: str) -> None:
    if not source_dir.is_dir():
        report.fail(f"pinned GR00T source is missing: {source_dir}")
        return
    report.ok(f"pinned GR00T source exists: {source_dir}")
    for relative in ("gr00t/experiment/launch_finetune.py", "gr00t/data/stats.py"):
        report.check((source_dir / relative).is_file(), f"pinned source ships {relative}")

    try:
        actual = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        report.fail(f"cannot read the GR00T source revision at {source_dir}: {exc}")
        return
    report.check(
        actual == expected_commit,
        f"GR00T source revision is {actual} (expected the pinned {expected_commit})",
    )
    try:
        status = subprocess.run(
            ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        report.fail(f"cannot read the GR00T source status at {source_dir}: {exc}")
        return
    report.check(not status, "GR00T source has no tracked modifications")
    report.record("source_commit", actual)


def read_effective_optimizer(run_dir: Path) -> tuple[str | None, Path | None]:
    """Read ``training.optim`` from the launcher's own config dump.

    ``experiment_cfg/conf.yaml`` is written by the pinned ``experiment.run()``
    from the live Config object right before training, so it records the
    optimizer the Trainer actually built -- not what a wrapper intended to set.
    This module is stdlib-only, so the scalar is pulled out with a line match
    instead of a YAML parser: OmegaConf dumps ``optim`` as an unquoted scalar at
    two-space indent inside the top-level ``training:`` mapping.
    """
    for name in ("conf.yaml", "config.yaml"):
        path = run_dir / "experiment_cfg" / name
        if not path.is_file():
            continue
        match = TRAIN_OPTIM_RE.search(path.read_text(encoding="utf-8", errors="replace"))
        if match:
            return match.group(1), path
    return None, None


def check_artifacts(
    run_dir: Path, report: Report, *, expect_steps: int | None, expect_optim: str | None = None
) -> None:
    if not run_dir.is_dir():
        report.fail(f"run directory is missing: {run_dir}")
        return
    report.ok(f"run directory exists: {run_dir}")

    experiment_cfg = run_dir / "experiment_cfg"
    if not experiment_cfg.is_dir():
        report.fail(f"run directory has no experiment_cfg (launcher never got to setup): {run_dir}")
    else:
        report.check(
            any((experiment_cfg / name).is_file() for name in ("config.yaml", "conf.yaml")),
            "run experiment_cfg holds config.yaml/conf.yaml",
        )
        if expect_optim is not None:
            effective, source = read_effective_optimizer(run_dir)
            where = source.name if source is not None else "experiment_cfg/{config,conf}.yaml"
            report.check(
                effective is not None,
                f"{where} records training.optim (expected {expect_optim!r})",
            )
            if effective is not None:
                report.check(
                    effective == expect_optim,
                    f"effective optimizer is {effective!r} (expected {expect_optim!r}); "
                    "a mismatch means the launcher shim did not apply OPTIM",
                )
                report.record("optimizer", effective)
    processor = run_dir / "processor"
    for name in ("processor_config.json", "statistics.json"):
        report.check(
            (processor / name).is_file() or (run_dir / name).is_file(),
            f"run ships {name}",
        )

    checkpoints: list[tuple[int, Path]] = []
    for path in sorted(run_dir.glob("checkpoint-*")):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir():
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        report.fail(f"run directory has no checkpoint-<step> directory: {run_dir}")
        return
    checkpoints.sort()
    step, latest = checkpoints[-1]
    report.ok(f"latest checkpoint is {latest.name}")
    report.record("checkpoint_steps", [n for n, _ in checkpoints])

    if expect_steps is not None:
        report.check(
            step == expect_steps,
            f"latest checkpoint is step {step} (expected exactly {expect_steps} optimizer steps)",
        )

    state = load_json(latest / "trainer_state.json", report, f"{latest.name}/trainer_state.json")
    if isinstance(state, dict):
        global_step = state.get("global_step")
        report.check(
            global_step == step,
            f"{latest.name} trainer_state.json global_step is {global_step!r} (expected {step})",
        )
        report.record("global_step", global_step)
    report.check((latest / "experiment_cfg").is_dir(), f"{latest.name} ships experiment_cfg/")
    for name in ("processor_config.json", "statistics.json"):
        report.check(
            (latest / name).is_file() or (latest / "processor" / name).is_file(),
            f"{latest.name} ships {name}",
        )
    weights = (
        sorted(latest.glob("*.safetensors"))
        + sorted(latest.glob("*.bin"))
        + sorted(latest.glob("pytorch_model*.pt"))
    )
    empty = [str(path) for path in weights if path.stat().st_size == 0]
    report.check(bool(weights) and not empty, f"{latest.name} ships non-empty model weights")
    if empty:
        report.fail(f"{latest.name} has zero-byte weight files: {empty}")


def compute_hashes(dataset_root: Path, report: Report, splits: list[str], train_id: str, val_id: str) -> None:
    ids = {"train": train_id, "val": val_id}
    manifest: dict[str, object] = {}
    for split in splits:
        split_root = dataset_root / ids[split]
        entry: dict[str, object] = {}
        for name in ("info.json", "modality.json", "stats.json", "relative_stats.json", "episodes.jsonl", "tasks.jsonl"):
            path = split_root / "meta" / name
            if path.is_file():
                entry[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        manifest[split] = entry
    report.record("dataset_meta_sha256", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="append",
        choices=["source", "model", "dataset", "stats", "artifacts", "hashes"],
        default=[],
        help="check to run (repeatable; default: source, model, dataset, stats)",
    )
    parser.add_argument("--dataset-root", type=Path, help="pack root holding the split directories")
    parser.add_argument("--split-root", type=Path, help="validate a single LeRobot split root")
    parser.add_argument("--train-id", default="train")
    parser.add_argument("--val-id", default="val")
    parser.add_argument(
        "--splits",
        default="train,val",
        help="comma-separated splits for dataset/stats/hash checks (default: train,val)",
    )
    parser.add_argument("--ego-key", default=STOCK_VIDEO_KEYS[0], help="expected ego camera key")
    parser.add_argument("--expected-fps", type=int, default=DATASET_FPS)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--expected-model-repo", default="")
    parser.add_argument("--expected-model-revision", default="")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--expected-source-commit", default="")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--expect-steps", type=int, default=None)
    parser.add_argument(
        "--expect-optim",
        default=None,
        help="artifact check: training.optim recorded in the run's experiment_cfg "
        "(e.g. adafactor); omit to skip the effective-optimizer check",
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable summary")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    checks = args.check or ["source", "model", "dataset", "stats"]
    splits = [split for split in args.splits.split(",") if split]
    unknown = [split for split in splits if split not in ("train", "val")]
    if unknown:
        print(f"FAIL: unknown split(s): {unknown}", file=sys.stderr)
        return 2

    report = Report(quiet=args.json)
    if "source" in checks:
        if args.source_dir is None:
            report.fail("--source-dir is required for the source check")
        else:
            check_source(args.source_dir, report, expected_commit=args.expected_source_commit)
    if "model" in checks:
        if args.model_dir is None:
            report.fail("--model-dir is required for the model check")
        else:
            check_model(
                args.model_dir,
                report,
                expected_repo=args.expected_model_repo,
                expected_revision=args.expected_model_revision,
            )
    if "dataset" in checks or "stats" in checks:
        want_contract = "dataset" in checks
        want_stats = "stats" in checks
        if args.dataset_root is None and args.split_root is None:
            report.fail("--dataset-root or --split-root is required for the dataset/stats checks")
        elif args.split_root is not None:
            check_split_root(
                args.split_root,
                report,
                "dataset split",
                ego_key=args.ego_key,
                expected_fps=args.expected_fps,
                want_contract=want_contract,
                want_stats=want_stats,
            )
        else:
            check_dataset_splits(
                args.dataset_root,
                report,
                splits,
                args.train_id,
                args.val_id,
                ego_key=args.ego_key,
                expected_fps=args.expected_fps,
                want_contract=want_contract,
                want_stats=want_stats,
            )
    if "artifacts" in checks:
        if args.run_dir is None:
            report.fail("--run-dir is required for the artifact check")
        else:
            check_artifacts(
                args.run_dir, report, expect_steps=args.expect_steps, expect_optim=args.expect_optim
            )
    if "hashes" in checks and args.dataset_root is not None:
        compute_hashes(args.dataset_root, report, splits, args.train_id, args.val_id)

    if args.json:
        print(
            json.dumps(
                {"ok": not report.errors, "errors": report.errors, "warnings": report.warnings, "records": report.records},
                indent=2,
                sort_keys=True,
            )
        )
    if report.errors:
        if not args.json:
            print(f"FAIL: {len(report.errors)} check(s) failed", file=sys.stderr)
        return 2
    if not args.json:
        print("PASS: all requested preflight checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
