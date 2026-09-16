"""One-episode pilot pipeline: raw source -> canonical reference -> QC -> manifest.

The raw collection stays read-only and untouched; every artifact lands in a fresh
timestamped directory below ``first_tur_processed``.  The pipeline never writes a
token by itself — encoding runs in the environment that pins onnxruntime — but it
does write the encoder *input* together with the QC decisions that decide whether
the episode may be used at all.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_lab.controllers.sonic import SONIC_REFERENCE_JOINT_ORDER

from .adapters import nvidia_apple_to_plate, nvidia_fruits, unitree_dex3
from .encoder_observation import load_encoder_layout
from .joints import reorder
from .production import PRODUCTION_ORIENTATION_POLICY, write_encoder_input
from .provenance import assert_processed_destination, sha256_file
from .quality import (
    DEFAULT_THRESHOLDS,
    clamp_report,
    derivative_metrics,
    evaluate_thresholds,
    finite_report,
    hand_range_decision,
    hand_range_report,
    load_joint_limits,
    overall_result,
    permutation_round_trip,
    range_report,
    resampling_report,
)
from .reference import ARM_NAMES, STANDING_COMPLETION_SCOPE, StandingPose, deployment_standing_pose
from .schema import PROCESSED_FPS, CanonicalEpisodeBuild

PILOT_SCHEMA_VERSION = 1
PILOT_KINDS = ("unitree", "fruits", "apple")
BODY_JOINT_NAMES = SONIC_REFERENCE_JOINT_ORDER

#: Source-specific hand range policy against the pinned Dex3 model.
#:
#: The Unitree Dex3 collection reaches 2.0944 rad (= 2*pi/3 = 120 deg) on the
#: index/middle distal joints and 0.192 rad past zero on the proximal ones, in
#: both ``action`` and measured ``observation.state``.  The pinned
#: ``g1_29dof_with_hand`` model (and all three shipped URDFs, plus the pinning
#: deployment's own ``MAX_LIMITS_*``/``MIN_LIMITS_*`` tables in ``dex3_hands.hpp``)
#: stop at 1.7453 rad (100 deg).  120 deg is the older Dex3 revision's documented
#: stroke, so these are real recorded values on wider-stroke hardware, not a wrong
#: permutation: an exhaustive search over all 5040 column permutations per hand
#: finds the canonical order already optimal, and the measured state violates in
#: the same channels as the commanded action.
#:
#: Measured over a 551-episode sample covering all 13 official collections, the
#: excess is strongly bimodal and never in between: either exactly 0.3491 rad
#: (= 2.0944 - 1.7453, the revision stroke difference) or ~0.0010 rad (a resting
#: pose sitting a hair past the modelled zero).  The tolerance is therefore the
#: discriminator, and the channel count is only a sanity bound: a whole-hand
#: revision plausibly moves every joint, so all 14 channels may deviate, but none
#: may deviate by more than the documented stroke difference.
#:
#: AppleToPlate's hands come from measured ``observation.state``.  Its left hand is
#: inside the modelled range throughout; the right hand rests about 0.053 rad below
#: the model's zero and barely moves, a real resting offset rather than a mapping
#: error.
HAND_RANGE_POLICY: dict[str, dict[str, float]] = {
    "unitree": {"allowed_channels": 14, "excess_tolerance_rad": 0.35},
    "apple": {"allowed_channels": 2, "excess_tolerance_rad": 0.06},
    "fruits": {"allowed_channels": 0, "excess_tolerance_rad": 0.0},
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def converter_revision() -> str:
    """Git commit of the checkout that produced a pilot, or ``unknown``."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git in some images
        return "unknown"


@dataclass(frozen=True)
class PilotSpec:
    """Where one pilot episode comes from."""

    kind: str
    dataset: Path
    episode_index: int
    pilot_name: str
    assume_unitree_mujoco_body_order: bool = False
    bulk_conversion_status: str = "eligible"
    bulk_conversion_reason: str = ""
    hands_from_state: bool = False

    @classmethod
    def from_config(
        cls,
        config: Path,
        kind: str,
        *,
        raw_root: Path,
        episode_index: int | None = None,
        assume_unitree_mujoco_body_order: bool = False,
    ) -> "PilotSpec":
        document = json.loads(Path(config).read_text(encoding="utf-8"))
        if kind not in document:
            raise ValueError(f"{config} does not define pilot {kind!r}; known: {sorted(document)}")
        entry = document[kind]
        index = int(entry["episode"] if episode_index is None else episode_index)
        return cls(
            kind=kind,
            dataset=Path(raw_root) / str(entry["dataset"]),
            episode_index=index,
            pilot_name=f"{kind}_{Path(str(entry['dataset'])).name}_ep{index:03d}",
            assume_unitree_mujoco_body_order=assume_unitree_mujoco_body_order,
            bulk_conversion_status=str(entry.get("bulk_conversion_status", "eligible")),
            bulk_conversion_reason=str(entry.get("bulk_conversion_reason", "")),
            hands_from_state=bool(entry.get("hands_from_state", False)),
        )

    @classmethod
    def bulk_from_config(
        cls,
        config: Path,
        kind: str,
        *,
        raw_root: Path,
        dataset: str | None = None,
        assume_unitree_mujoco_body_order: bool = False,
    ) -> "PilotSpec":
        """Resolve the bulk-conversion source of ``kind``.

        A kind may convert more than the single pilot dataset: ``bulk_datasets``
        lists every raw dataset the kind covers, while ``dataset`` stays the pilot
        episode's source.  ``dataset`` selects one entry explicitly; without it
        the first (or only) bulk dataset is used.  Gates and hand policy are
        inherited from the kind entry, so a kind marked ``excluded`` cannot be
        converted through any of its datasets.
        """
        document = json.loads(Path(config).read_text(encoding="utf-8"))
        if kind not in document:
            raise ValueError(f"{config} does not define pilot {kind!r}; known: {sorted(document)}")
        entry = document[kind]
        names = [str(name) for name in entry.get("bulk_datasets") or [entry["dataset"]]]
        if dataset is not None:
            if dataset not in names:
                raise ValueError(
                    f"{kind} does not declare bulk dataset {dataset!r}; declared: {names}"
                )
            chosen = dataset
        else:
            chosen = names[0]
        return cls(
            kind=kind,
            dataset=Path(raw_root) / chosen,
            episode_index=0,
            pilot_name=f"{kind}_{Path(chosen).name}_ep000",
            assume_unitree_mujoco_body_order=assume_unitree_mujoco_body_order,
            bulk_conversion_status=str(entry.get("bulk_conversion_status", "eligible")),
            bulk_conversion_reason=str(entry.get("bulk_conversion_reason", "")),
            hands_from_state=bool(entry.get("hands_from_state", False)),
        )

    @classmethod
    def bulk_datasets(cls, config: Path, kind: str) -> list[str]:
        document = json.loads(Path(config).read_text(encoding="utf-8"))
        if kind not in document:
            raise ValueError(f"{config} does not define pilot {kind!r}; known: {sorted(document)}")
        entry = document[kind]
        return [str(name) for name in entry.get("bulk_datasets") or [entry["dataset"]]]


def map_source_episode(spec: PilotSpec, standing: StandingPose) -> CanonicalEpisodeBuild:
    """Apply the dataset-specific mapping and return the shared canonical contract."""
    if spec.kind == "unitree":
        return unitree_dex3.load_episode(spec.dataset, spec.episode_index, standing)
    if spec.kind == "fruits":
        return nvidia_fruits.load_episode(spec.dataset, spec.episode_index, standing=standing)
    if spec.kind == "apple":
        return nvidia_apple_to_plate.load_body_pilot(
            spec.dataset,
            spec.episode_index,
            standing=standing,
            assume_unitree_mujoco_body_order=spec.assume_unitree_mujoco_body_order,
            hands_from_state=spec.hands_from_state,
        )
    raise ValueError(f"unknown pilot kind {spec.kind!r}; known: {PILOT_KINDS}")


def source_video(spec: PilotSpec) -> tuple[Path, float | None, float | None]:
    """Locate the head/ego camera clip of the pilot episode and its cut window."""
    if spec.kind == "unitree":
        import pyarrow.parquet as pq

        for path in sorted((spec.dataset / "meta/episodes").glob("chunk-*/*.parquet")):
            for row in pq.read_table(path).to_pylist():
                if int(row["episode_index"]) != spec.episode_index:
                    continue
                key = "videos/observation.images.cam_left_high"
                return (
                    spec.dataset
                    / f"videos/observation.images.cam_left_high/chunk-{int(row[f'{key}/chunk_index']):03d}"
                    / f"file-{int(row[f'{key}/file_index']):03d}.mp4",
                    float(row[f"{key}/from_timestamp"]),
                    float(row[f"{key}/to_timestamp"]),
                )
        raise ValueError(f"episode {spec.episode_index} has no Dex3 head-camera segment")
    return spec.dataset / f"videos/chunk-000/observation.images.ego_view/episode_{spec.episode_index:06d}.mp4", None, None


def extract_source_video(spec: PilotSpec, destination: Path) -> dict[str, Any]:
    source, start, stop = source_video(spec)
    if not source.is_file():
        raise FileNotFoundError(f"source video missing: {source}")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None and stop is not None:
        command += ["-ss", f"{start}", "-to", f"{stop}"]
    command += ["-i", str(source), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination)]
    subprocess.run(command, check=True)
    return {
        "camera": "observation.images.cam_left_high" if start is not None else "observation.images.ego_view",
        "source_path": str(source),
        "clip_start_s": start,
        "clip_stop_s": stop,
        "packed_file_sha256": None if start is not None else sha256_file(source),
        "clip_sha256": sha256_file(destination),
    }


def write_canonical_reference(output: Path, build: CanonicalEpisodeBuild) -> None:
    """Persist the adapter output in the shared canonical reference format."""
    episode = build.episode
    directory = output / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / "timestamps.csv", episode.timestamps, delimiter=",")
    np.savetxt(directory / "joint_pos.csv", episode.joint_pos, delimiter=",")
    np.savetxt(directory / "joint_vel.csv", episode.joint_vel, delimiter=",")
    np.savetxt(directory / "body_pos.csv", episode.body_pos, delimiter=",")
    np.savetxt(directory / "body_quat.csv", episode.body_quat_wxyz, delimiter=",")
    (directory / "metadata.txt").write_text(
        "SONIC v1.1 reference motion: 50 Hz, joint_pos/joint_vel in official IsaacLab order, "
        "body_quat wxyz, body part index [0] = root\n",
        encoding="utf-8",
    )
    (directory / "provenance.json").write_text(
        json.dumps(build.provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "reference.npz",
        timestamps=episode.timestamps,
        joint_pos=episode.joint_pos,
        joint_vel=episode.joint_vel,
        body_quat_wxyz=episode.body_quat_wxyz,
        body_pos=episode.body_pos,
        left_hand_joints=episode.left_hand_joints,
        right_hand_joints=episode.right_hand_joints,
        hands_applied=np.asarray([build.final_action_allowed()], dtype=np.bool_),
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {
                "frame_index": np.arange(len(episode.timestamps), dtype=np.int64),
                "timestamp": episode.timestamps,
                "joint_pos": episode.joint_pos.tolist(),
                "joint_vel": episode.joint_vel.tolist(),
                "body_quat_wxyz": episode.body_quat_wxyz.tolist(),
                "left_hand_joints": episode.left_hand_joints.tolist(),
                "right_hand_joints": episode.right_hand_joints.tolist(),
            }
        ),
        output / "reference.parquet",
    )


def _source_echo(spec: PilotSpec) -> dict[str, Any] | None:
    """Source joint channels with the canonical indices they must land on.

    The QC step interpolates these directly and compares them with the stored
    canonical channels, so a wrong permutation or a broken resampling shows up as
    a numeric mismatch instead of a plausible-looking trajectory.
    """
    if spec.kind == "unitree":
        action, timestamps = unitree_dex3.resolve_episode_rows(spec.dataset, spec.episode_index)
        arms = reorder(action, unitree_dex3.UNITREE_ACTION_NAMES, unitree_dex3.ARM_SOURCE_NAMES)
        return {
            "label": "recorded_arm_action",
            "timestamps": timestamps,
            "values": arms,
            "names": ARM_NAMES,
            "target_indices": np.array([BODY_JOINT_NAMES.index(name) for name in ARM_NAMES]),
        }
    import pyarrow.parquet as pq

    if spec.kind == "fruits":
        slices = nvidia_fruits.action_slices(spec.dataset)
        blocks = nvidia_fruits.BODY_BLOCKS
        names = tuple(
            name
            for block in blocks
            for name in json.loads((spec.dataset / "meta/info.json").read_text(encoding="utf-8"))["features"]["action"]["names"][slices[block]]
        )
        path = nvidia_fruits.episode_path(spec.dataset, spec.episode_index)
    else:
        slices = nvidia_apple_to_plate.body_block_slices(spec.dataset)
        blocks = nvidia_apple_to_plate.BODY_BLOCKS
        names = nvidia_apple_to_plate.assumed_body_names(slices)
        path = spec.dataset / f"data/chunk-000/episode_{spec.episode_index:06d}.parquet"
    table = pq.read_table(path, columns=["action", "timestamp"])
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64).reshape(-1)
    source = np.concatenate([action[:, slices[block]] for block in blocks], axis=1)
    order = [names.index(name) for name in BODY_JOINT_NAMES]
    return {
        "label": "recorded_body_action",
        "timestamps": timestamps,
        "values": source[:, order],
        "names": BODY_JOINT_NAMES,
        "target_indices": np.arange(len(BODY_JOINT_NAMES)),
    }


def _synthetic_report(build: CanonicalEpisodeBuild, spec: PilotSpec, standing: StandingPose) -> dict[str, Any] | None:
    """How far the synthetic channels sit from the frame they were declared to hold."""
    if spec.kind not in {"unitree", "fruits", "apple"}:
        return None
    episode = build.episode
    synthetic_ids = (
        [index for index, name in enumerate(BODY_JOINT_NAMES) if name not in set(ARM_NAMES)]
        if spec.kind == "unitree"
        else []
    )
    body_error = 0.0
    if synthetic_ids:
        reference = np.asarray(standing.joint_pos, dtype=np.float64)[synthetic_ids]
        body_error = float(np.abs(episode.joint_pos[:, synthetic_ids] - reference).max())
    root_position_error = float(np.abs(episode.body_pos - np.asarray(standing.body_pos, dtype=np.float64)).max())
    root_orientation_error = float(
        np.abs(episode.body_quat_wxyz - np.asarray(standing.body_quat_wxyz, dtype=np.float64)).max()
    )
    return {
        "frozen_joints": synthetic_ids,
        "max_abs_error_rad": max(body_error, root_position_error, root_orientation_error),
        "joint_error_rad": body_error,
        "root_position_error_m": root_position_error,
        "root_orientation_error": root_orientation_error,
        "source": standing.source,
        "scope": STANDING_COMPLETION_SCOPE if spec.kind == "unitree" else "synthetic_root_only",
    }


def _write_raw_hand_command(output: Path, build: CanonicalEpisodeBuild) -> dict[str, Any] | None:
    """Persist source-rate hand channels that must not enter the canonical slots."""
    raw = build.raw_hand_command
    if raw is None:
        return None
    arrays = {name: np.asarray(value) for name, value in raw.items()}
    np.savez_compressed(output / "raw_hand_command.npz", **arrays)
    return {
        "path": "raw_hand_command.npz",
        "rate": "source",
        "meaning": "normalized hand command, not radians; excluded from the canonical hand slots",
        "arrays": {name: list(np.shape(value)) for name, value in arrays.items()},
        "dtype": {name: str(value.dtype) for name, value in arrays.items()},
    }


def _write_state_reference(output: Path, reference: Any, *, stem: str) -> dict[str, Any]:
    """Write a source-rate reference (recorded state or action) plus its metadata."""
    np.savez_compressed(output / f"{stem}.npz", **reference.to_arrays())
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(
            pa.table(
                {
                    "frame_index": np.arange(len(reference.timestamps), dtype=np.int64),
                    "timestamp": np.asarray(reference.timestamps, dtype=np.float64),
                    "joint_pos": np.asarray(reference.joint_pos, dtype=np.float32).tolist(),
                    "joint_vel": np.asarray(reference.joint_vel, dtype=np.float32).tolist(),
                    "left_hand_joints": np.asarray(reference.left_hand_joints, dtype=np.float32).tolist(),
                    "right_hand_joints": np.asarray(reference.right_hand_joints, dtype=np.float32).tolist(),
                }
            ),
            output / f"{stem}.parquet",
        )
    except ImportError:  # pragma: no cover - parquet is present in the conversion env
        pass
    return {
        "path": f"{stem}.npz",
        "frames": int(len(reference.timestamps)),
        "source_fps": float(reference.provenance.get("source_fps", 0.0)),
        "duration_s": float(reference.timestamps[-1] - reference.timestamps[0]),
        "hands_applied": bool(reference.hands_applied),
        "provenance": reference.provenance,
    }


def _source_rate_references(spec: PilotSpec, limits_path: Path) -> dict[str, Any]:
    """Recorded-state and source-rate action references, where the source has them."""
    if spec.kind == "fruits":
        limits = load_joint_limits(limits_path)
        return {
            "state": nvidia_fruits.load_state_reference(spec.dataset, spec.episode_index, limits=limits),
            "action": nvidia_fruits.load_action_source_reference(spec.dataset, spec.episode_index),
        }
    if spec.kind == "apple":
        limits = load_joint_limits(limits_path)
        return {
            "state": nvidia_apple_to_plate.load_state_reference(spec.dataset, spec.episode_index, limits=limits),
            "action": nvidia_apple_to_plate.load_action_source_reference(spec.dataset, spec.episode_index),
        }
    return {}


def evaluate_episode_qc(
    build: CanonicalEpisodeBuild,
    spec: PilotSpec,
    standing: StandingPose,
    *,
    observation: np.ndarray,
    clamp_fraction: np.ndarray,
    limits_path: Path,
    layout_path: Path | None,
) -> dict[str, Any]:
    episode = build.episode
    limits = load_joint_limits(limits_path)
    derivatives = derivative_metrics(episode.joint_pos, PROCESSED_FPS)
    finite = finite_report(episode.joint_pos)
    ranges = range_report(episode.joint_pos, BODY_JOINT_NAMES, limits)
    clamp = clamp_report(clamp_fraction)
    values: dict[str, float] = {
        "finite.non_finite_count": float(finite["non_finite_count"]),
        "range.violation_count": float(ranges["violation_count"]),
        "velocity.abs_p99": derivatives["velocity_abs_p99"],
        "acceleration.abs_p99": derivatives["acceleration_abs_p99"],
        "jerk.abs_p99": derivatives["jerk_abs_p99"],
        "future_clamp.mean_fraction": clamp["mean_fraction"],
    }
    echo = _source_echo(spec)
    resampling = None
    if echo is not None:
        resampling = resampling_report(
            echo["timestamps"],
            echo["values"],
            episode.timestamps,
            episode.joint_pos[:, echo["target_indices"]],
            tuple(echo["names"]),
        )
        values["source_echo.max_abs_error_rad"] = resampling["max_abs_error_rad"]
    synthetic = _synthetic_report(build, spec, standing)
    if synthetic is not None:
        values["synthetic.max_abs_error_rad"] = synthetic["max_abs_error_rad"]
    decisions = evaluate_thresholds(values, DEFAULT_THRESHOLDS)
    # The body range check above never looks at the hands, so a hand trajectory
    # could sit arbitrarily far outside the modelled Dex3 range unnoticed.
    hands = hand_range_report(episode.left_hand_joints, episode.right_hand_joints, limits)
    hand_decision = None
    if build.final_action_allowed():
        policy = HAND_RANGE_POLICY.get(spec.kind, {"allowed_channels": 0, "excess_tolerance_rad": 0.0})
        hand_decision = hand_range_decision(hands, **policy)
        decisions.append(hand_decision)
    round_trips: dict[str, Any] = {}
    for part, item in (build.provenance.get("name_check") or {}).items():
        if isinstance(item, dict) and item.get("mode") == "name_round_trip":
            round_trips[part] = permutation_round_trip(tuple(item["source"]), tuple(item["target"]))
        else:
            round_trips[part] = {"ok": True, "mode": item if isinstance(item, str) else "identity_by_construction"}
    layout = load_encoder_layout(layout_path) if layout_path is not None else None
    threshold_result = overall_result(decisions)
    return {
        "thresholds": DEFAULT_THRESHOLDS,
        "values": values,
        "decisions": decisions,
        "threshold_result": threshold_result,
        "result": threshold_result,
        "final_action_result": "PASS" if build.final_action_allowed() else "UNVERIFIED",
        "episode_result": (
            "FAIL"
            if threshold_result == "FAIL"
            else "PASS"
            if build.final_action_allowed()
            else "UNVERIFIED"
        ),
        "finite": finite,
        "joint_range": ranges,
        "hand_range": {**hands, "decision": hand_decision},
        "derivatives": derivatives,
        "future_clamp": clamp,
        "source_echo": resampling,
        "source_echo_label": None if echo is None else echo["label"],
        "synthetic": synthetic,
        "name_round_trip": round_trips,
        "encoder_layout_verified": layout is not None,
        "encoder_layout_dim": None if layout is None else layout.total_dim,
        "hand_schema_status": build.hand_schema_status,
        "observation_shape": list(observation.shape),
        "observation_finite": bool(np.isfinite(observation).all()),
    }


def prepare_pilot(
    spec: PilotSpec,
    *,
    raw_root: Path,
    processed_root: Path,
    limits_path: Path,
    standing: StandingPose | None = None,
    layout_path: Path | None = None,
    timestamp: str | None = None,
    with_video: bool = True,
) -> dict[str, Any]:
    """Materialise one pilot episode and return its manifest."""
    assert_processed_destination(raw_root, processed_root)
    standing = standing or deployment_standing_pose()
    stamp = timestamp or utc_timestamp()
    output = Path(processed_root) / "pilots" / spec.pilot_name / stamp
    if output.exists():
        raise FileExistsError(f"pilot output already exists: {output}")
    build = map_source_episode(spec, standing)
    episode = build.episode
    output.mkdir(parents=True)
    write_canonical_reference(output, build)
    encoder_input = write_encoder_input(output, episode)
    observation = encoder_input.observation
    clamp = encoder_input.future_clamp_fraction
    video = extract_source_video(spec, output / "source.mp4") if with_video else None
    raw_hand = _write_raw_hand_command(output, build)
    source_rate = _source_rate_references(spec, limits_path)
    source_rate_artifacts: dict[str, Any] = {}
    if "state" in source_rate:
        source_rate_artifacts["recorded_state"] = _write_state_reference(
            output, source_rate["state"], stem="reference_state"
        )
    if "action" in source_rate:
        source_rate_artifacts["encoder_action_source"] = _write_state_reference(
            output, source_rate["action"], stem="reference_action_source"
        )
    qc = evaluate_episode_qc(
        build,
        spec,
        standing,
        observation=observation,
        clamp_fraction=clamp,
        limits_path=limits_path,
        layout_path=layout_path,
    )
    manifest = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "pilot": spec.pilot_name,
        "kind": spec.kind,
        "dataset": str(spec.dataset),
        "dataset_name": spec.dataset.name,
        "episode_index": spec.episode_index,
        "created_utc": stamp,
        "converter_git_commit": converter_revision(),
        "raw_root": str(Path(raw_root).resolve()),
        "processed_dir": str(output.resolve()),
        "source_fps": build.provenance.get("source_fps"),
        "processed_fps": PROCESSED_FPS,
        "frames": int(len(episode.timestamps)),
        "duration_s": float(episode.timestamps[-1] - episode.timestamps[0]),
        "lookahead": {"frames": 10, "step": 5, "seconds": 0.9},
        "encoder_orientation_policy": PRODUCTION_ORIENTATION_POLICY.value,
        "encoder_orientation_policy_note": (
            "offline deterministic assumption: the measured base heading is replaced by the reference "
            "root heading at the current frame (equivalent to upstream's refheading variant with an "
            "identity apply_delta_heading); the live C++ semantic uses the recorded robot base"
        ),
        "standing_completion_policy": build.provenance.get("standing_completion_policy"),
        "standing_completion_scope": STANDING_COMPLETION_SCOPE if spec.kind == "unitree" else "not_applicable",
        "hand_schema_status": build.hand_schema_status,
        "hand_schema_reason": build.hand_schema_reason,
        "final_action_78d": {
            "status": "available" if build.final_action_allowed() else "blocked",
            "reason": "hand channels resolved" if build.final_action_allowed() else build.hand_schema_reason,
        },
        "source_video": video,
        "raw_hand_command": raw_hand,
        "encoder_source_field": "action",
        "visual_recorded_source_field": (
            "observation.state" if "state" in source_rate_artifacts else None
        ),
        "source_rate_references": source_rate_artifacts,
        "bulk_conversion": {
            "status": spec.bulk_conversion_status,
            "reason": spec.bulk_conversion_reason,
        },
        "provenance": build.provenance,
        "notes": list(build.notes),
        "qc": qc,
        "human_review": "pending_human_review",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "human_review.json").write_text(
        json.dumps({"status": "pending_human_review", "notes": ""}, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_manifest(pilot_dir: Path) -> dict[str, Any]:
    return json.loads((Path(pilot_dir) / "run_manifest.json").read_text(encoding="utf-8"))


def latest_pilot_dir(processed_root: Path, pilot_name: str) -> Path:
    """Newest run of a pilot that actually carries a review file.

    Ordering by timestamp alone is fragile: any later run that died before
    ``human_review.json`` was written (an aborted bulk episode, a crash) becomes
    "the latest" and shadows the real pilot, which turns the review gate into a
    confusing "status is 'missing'" refusal.  Only runs that produced a review
    file are candidates; if none has one, fall back to the newest run so the
    error still names a real directory.
    """
    root = Path(processed_root) / "pilots" / pilot_name
    if not root.is_dir():
        raise FileNotFoundError(f"no pilot outputs under {root}")
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no pilot outputs under {root}")
    reviewed = [path for path in candidates if (path / "human_review.json").is_file()]
    return (reviewed or candidates)[-1]


def review_status(pilot_dir: Path) -> str:
    path = Path(pilot_dir) / "human_review.json"
    if not path.is_file():
        return "missing"
    return str(json.loads(path.read_text(encoding="utf-8")).get("status", "missing"))


def require_approved_review(pilot_dir: Path, *, allow_statuses: tuple[str, ...]) -> None:
    status = review_status(pilot_dir)
    if status not in allow_statuses:
        raise SystemExit(
            f"{pilot_dir}: human review status is {status!r}; bulk conversion needs one of {allow_statuses}. "
            f"Edit {Path(pilot_dir) / 'human_review.json'} after inspecting the review page."
        )
