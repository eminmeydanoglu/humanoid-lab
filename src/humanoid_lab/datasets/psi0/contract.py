"""Psi0 fine-tuning data contract for the Unitree Dex3 SONIC v1 pack.

The pack under ``data/datasets/psi0-unitree-dex3-sonic-v1`` is produced by the
dataset owner.  This module is the only place where the Psi0 loader's
expectations for that pack are written down, so a field/shape drift fails in the
preflight instead of half way through a forward pass.  The conversion pipeline
(``humanoid_lab.datasets.psi0_dex3``) imports the field names and widths from
here rather than declaring a second copy.

Field names follow the training plan (``psi_egitime_hazirlik.md`` §7 and §9)
with one addition the loader forces: the per-frame action validity must be
carried **wide**.  Psi0's ``SonicRepackTransform`` multiplies the mask field into
an ``(action_chunk_size, action_dim)`` loss mask and reshapes it to that shape
(``transform_psi0_sonic.py``), so a scalar ``(T,)`` mask raises
``cannot reshape array of size T into shape (T, 80)``.  The wide layout is also
what upstream's own SONIC pack ships as ``action.mask``
(``scripts/data/merge_sonic_v1.py``).

Normalisation widths are the *raw* widths (64 / 14 / 43): the loader pads the
statistics itself with ``pad_to_len``, and a stats file that is already padded
would concatenate to 45 or 94 channels and stop matching the action tensor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: Data-root relative location of the pack (``data/datasets/...``).
DATASET_DIR = "psi0-unitree-dex3-sonic-v1"
#: The same layout built from one or two episodes per task with
#: ``scripts/convert-psi0-sonic-dataset.py --output-root ...-mini``.  Every gate
#: run uses this pack so it never needs the full corpus; it is otherwise a
#: sibling of the production pack.
GATE_DATASET_DIR = f"{DATASET_DIR}-mini"
#: Directory names ``data.root_dir`` is allowed to carry.
DATASET_DIR_ALIASES = (GATE_DATASET_DIR,)
#: Psi0 ``train_repo_ids`` / ``val_repo_ids``: two LeRobot packs under the root.
TRAIN_REPO_ID = "train"
VAL_REPO_ID = "val"
SPLIT_MANIFEST = "split_manifest.json"

#: Control frequency of the pack and the action horizon in frames (1 s at 30 Hz).
FPS = 30.0
ACTION_CHUNK = 30

#: 15D synthetic standing legs/waist + 14D arms + 14D Dex3 hands, padded to 45.
STATE_DIM = 43
STATE_MODEL_DIM = 45
#: 64D SONIC body token + 14D Dex3 hands, padded to 80.
BODY_TOKEN_DIM = 64
HAND_DIM = 14
ACTION_DIM = BODY_TOKEN_DIM + HAND_DIM
ACTION_MODEL_DIM = 80

IMAGE_KEY = "observation.images.egocentric"
STATE_KEY = "observation.state"
BODY_TOKEN_KEY = "action.body_token_v1_1"
HAND_KEY = "action"
#: Wide per-frame action validity, ``action_dim`` wide and in repacked order.
MASK_KEY = "action.mask"
#: Alternative names the loader accepts when the pack is explicitly configured
#: with them (``--data.transform.repack.action-mask-key`` / ``MASK_KEY=``).  The
#: plan calls the mask ``training_valid_mask``; it still has to be ``action_dim``
#: wide for the loader to read it.
MASK_KEY_ALIASES = ("training_valid_mask",)
#: Per-frame strict anchor mask produced by the converter
#: (``anchor_valid[i] = all(valid_30[i:i+30])``).  Psi0's loader has no
#: anchor-level filter hook, so this field is not fed to the loss; it is the
#: auditable source of the plan's rule and is checked against ``action.mask`` in
#: the Gate 2 loader check.
ANCHOR_MASK_KEY = "anchor_valid"

#: The canonical instruction is written as its own frame field, not read back
#: from ``task_index``: seven upstream collections share one task label and
#: GraspSquare carries CameraPackaging's, so the configuration is the only
#: source of the instruction text.
INSTRUCTION_KEY = "task_description"
#: LeRobot's own frame task string, accepted when a pack has no
#: ``task_description`` column.
INSTRUCTION_KEY_ALIASES = ("task",)

STATS_FILENAME = "stats_psi0.json"
STATS_PATH = f"meta/{STATS_FILENAME}"

#: Normalisation statistics the loader needs, as {stats key: raw width}.
STAT_ACTION_KEYS = (BODY_TOKEN_KEY, HAND_KEY)
STAT_ACTION_WIDTHS = {BODY_TOKEN_KEY: BODY_TOKEN_DIM, HAND_KEY: HAND_DIM}
STAT_STATE_WIDTHS = {STATE_KEY: STATE_DIM}

REQUIRED_DATASET_META = (
    "meta/info.json",
    "meta/modality.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    STATS_PATH,
)
#: LeRobot v3 files the same content as parquet; either layout is accepted for
#: the two task/episode tables.
ALTERNATE_DATASET_META = {
    "meta/tasks.jsonl": ("meta/tasks.parquet",),
    "meta/episodes.jsonl": ("meta/episodes",),
}

#: Frame tables every LeRobot pack carries.  ``next.done`` is a v2 field and is
#: not required.
REQUIRED_FRAME_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")

#: Widths of the numeric features, in the order they are checked.
NUMERIC_FEATURES = {
    STATE_KEY: STATE_DIM,
    BODY_TOKEN_KEY: BODY_TOKEN_DIM,
    HAND_KEY: HAND_DIM,
}


class DatasetContractError(ValueError):
    """The produced pack does not satisfy the Psi0 loader contract."""


@dataclass(frozen=True)
class DatasetReport:
    """What the pack actually declares, once it has passed validation."""

    repo_dir: Path
    fps: float
    total_episodes: int
    total_frames: int
    mask_key: str
    instruction_key: str
    stats_path: Path
    task_count: int
    features: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.repo_dir}: {self.total_episodes} episodes / {self.total_frames} frames "
            f"@ {self.fps:g} Hz, {self.task_count} tasks, mask={self.mask_key}, "
            f"instruction={self.instruction_key}, stats={self.stats_path.name}"
        )


def resolve_repo_dir(pack_root: Path | str, repo_id: str = TRAIN_REPO_ID) -> Path:
    """One split of the pack; ``pack_root`` is the value of ``--data.root_dir``."""
    return Path(pack_root) / repo_id


def pack_root_for(data_root: Path | str) -> Path:
    """The pack root under a LeRobot data root (``.../datasets``)."""
    return Path(data_root) / DATASET_DIR


def _load_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise DatasetContractError(f"{what} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetContractError(f"{what} is not valid JSON: {path}: {error}") from error


def _feature_shape(features: Mapping[str, Any], key: str) -> list[int] | None:
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        return None
    shape = feature.get("shape")
    return list(shape) if shape is not None else []


def _require_feature(features: Mapping[str, Any], key: str, width: int | None, repo_dir: Path) -> Mapping[str, Any]:
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        raise DatasetContractError(
            f"{repo_dir}/meta/info.json declares no feature {key!r}; the Psi0 loader reads it"
        )
    if width is None:
        return feature
    shape = _feature_shape(features, key)
    if shape != [width]:
        raise DatasetContractError(
            f"{repo_dir}/meta/info.json declares {key!r} with shape {shape}, expected [{width}]"
        )
    return feature


def _require_stats_block(stats: Mapping[str, Any], key: str, width: int, stats_path: Path) -> None:
    block = stats.get(key)
    if not isinstance(block, Mapping):
        raise DatasetContractError(
            f"{stats_path} has no {key!r} block; the loader normalises {key!r} from it"
        )
    for bound in ("min", "max"):
        values = block.get(bound)
        if not isinstance(values, (list, tuple)) or len(values) != width:
            got = len(values) if isinstance(values, (list, tuple)) else values
            raise DatasetContractError(
                f"{stats_path}: {key!r} {bound} holds {got} values, expected the raw width {width} "
                "(the loader pads the statistics itself, so an already padded block would stop "
                "matching the action tensor)"
            )
        for index, value in enumerate(values):
            if not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
                raise DatasetContractError(f"{stats_path}: {key!r} {bound}[{index}] is not finite: {value!r}")
    lo, hi = block["min"], block["max"]
    for index, (low, high) in enumerate(zip(lo, hi)):
        if high < low:
            raise DatasetContractError(f"{stats_path}: {key!r} dim {index} has max {high} < min {low}")


def resolve_mask_key(features: Mapping[str, Any], requested: str | None = None) -> str:
    """Return the action-mask feature, defaulting to :data:`MASK_KEY`.

    ``requested=None`` means the contract default, not "search for anything
    mask-like": the loader reads one concrete field name, so silently picking a
    different one would only move the failure into the transform.
    """
    key = requested or MASK_KEY
    if key not in features:
        expected = ", ".join(repr(alias) for alias in MASK_KEY_ALIASES) or "none"
        raise DatasetContractError(
            f"the pack declares no {key!r} action-mask feature; it needs the "
            f"{ACTION_MODEL_DIM}-wide per-frame validity field {MASK_KEY!r} "
            f"(alternatives accepted when configured explicitly: {expected})"
        )
    return str(key)


def validate_dataset(
    repo_dir: Path | str,
    *,
    mask_key: str | None = None,
    instruction_key: str = INSTRUCTION_KEY,
) -> DatasetReport:
    """Fail-closed check of one produced split against the Psi0 loader contract."""
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise DatasetContractError(f"dataset split is missing: {repo_dir}")

    for relative in REQUIRED_DATASET_META:
        if (repo_dir / relative).exists():
            continue
        alternates = ALTERNATE_DATASET_META.get(relative, ())
        if any((repo_dir / alternate).exists() for alternate in alternates):
            continue
        raise DatasetContractError(f"{repo_dir} is missing {relative}")

    info = _load_json(repo_dir / "meta/info.json", "meta/info.json")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise DatasetContractError(f"{repo_dir}/meta/info.json has no features table")

    fps = float(info.get("fps", 0.0))
    if fps != FPS:
        raise DatasetContractError(f"{repo_dir}/meta/info.json declares {fps} fps, expected {FPS}")

    for key, width in NUMERIC_FEATURES.items():
        feature = _require_feature(features, key, width, repo_dir)
        if feature.get("dtype") != "float32":
            raise DatasetContractError(
                f"{repo_dir}/meta/info.json declares {key!r} as {feature.get('dtype')!r}, expected 'float32'"
            )
        names = feature.get("names")
        if names:
            declared = names[0] if isinstance(names[0], (list, tuple)) else names
            if len(declared) != width:
                raise DatasetContractError(
                    f"{repo_dir}/meta/info.json declares {len(declared)} names for {key!r}, expected {width}"
                )

    image = _require_feature(features, IMAGE_KEY, None, repo_dir)
    if image.get("dtype") not in ("video", "image"):
        raise DatasetContractError(
            f"{repo_dir}/meta/info.json declares {IMAGE_KEY!r} as {image.get('dtype')!r}, "
            "expected a video/image feature"
        )

    resolved_mask = resolve_mask_key(features, mask_key)
    mask = _require_feature(features, resolved_mask, ACTION_MODEL_DIM, repo_dir)
    if mask.get("dtype") not in ("float32", "bool", "int64"):
        raise DatasetContractError(
            f"{repo_dir}/meta/info.json declares {resolved_mask!r} as {mask.get('dtype')!r}, "
            "expected float32 or bool"
        )

    for key in REQUIRED_FRAME_KEYS:
        if key not in features:
            raise DatasetContractError(f"{repo_dir}/meta/info.json declares no frame key {key!r}")

    anchor = _require_feature(features, ANCHOR_MASK_KEY, None, repo_dir)
    if anchor.get("shape") != [1]:
        raise DatasetContractError(
            f"{repo_dir}/meta/info.json declares {ANCHOR_MASK_KEY!r} with shape {anchor.get('shape')}, "
            "expected the per-frame scalar [1]"
        )

    if instruction_key not in features:
        accepted = ", ".join(repr(name) for name in (INSTRUCTION_KEY, *INSTRUCTION_KEY_ALIASES))
        raise DatasetContractError(
            f"{repo_dir}/meta/info.json declares no instruction feature {instruction_key!r} "
            f"(accepted: {accepted})"
        )

    stats_path = repo_dir / STATS_PATH
    stats = _load_json(stats_path, STATS_PATH)
    if not isinstance(stats, Mapping):
        raise DatasetContractError(f"{stats_path} is not a JSON object")
    for key, width in STAT_ACTION_WIDTHS.items():
        _require_stats_block(stats, key, width, stats_path)
        block = stats[key]
        if not any(high > low for low, high in zip(block["min"], block["max"])):
            raise DatasetContractError(
                f"{stats_path}: {key!r} has no varying dimension, so normalisation would be a no-op"
            )
    for key, width in STAT_STATE_WIDTHS.items():
        _require_stats_block(stats, key, width, stats_path)

    tasks = info.get("total_tasks")
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    if total_episodes < 1 or total_frames < ACTION_CHUNK:
        raise DatasetContractError(
            f"{repo_dir}/meta/info.json reports {total_episodes} episodes / {total_frames} frames; "
            f"at least one episode of {ACTION_CHUNK} frames is needed for one action chunk"
        )

    return DatasetReport(
        repo_dir=repo_dir,
        fps=fps,
        total_episodes=total_episodes,
        total_frames=total_frames,
        mask_key=resolved_mask,
        instruction_key=instruction_key,
        stats_path=stats_path,
        task_count=int(tasks) if tasks is not None else 0,
        features=dict(features),
    )


def validate_pack(
    pack_root: Path | str,
    *,
    mask_key: str | None = None,
    instruction_key: str = INSTRUCTION_KEY,
) -> tuple[DatasetReport, DatasetReport]:
    """Validate both splits of the pack; the plan requires a non-empty val split."""
    train = validate_dataset(resolve_repo_dir(pack_root, TRAIN_REPO_ID), mask_key=mask_key, instruction_key=instruction_key)
    val = validate_dataset(resolve_repo_dir(pack_root, VAL_REPO_ID), mask_key=mask_key, instruction_key=instruction_key)
    if train.mask_key != val.mask_key:
        raise DatasetContractError(f"train uses mask {train.mask_key!r} but val uses {val.mask_key!r}")
    if train.instruction_key != val.instruction_key:
        raise DatasetContractError("train and val disagree on the instruction key")
    return train, val
