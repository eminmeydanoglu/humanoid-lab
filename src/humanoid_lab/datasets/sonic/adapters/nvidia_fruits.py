"""Fruits adapter: resolve body and hand blocks from modality metadata.

The collection is a 43D whole-body recording (LeRobot v2.1, 20 Hz) whose
``info.json`` names every channel, so both the body blocks and the Dex3 hands are
remapped by name into the canonical orders.  Only the root is missing: these are
static loco-manipulation demonstrations with no recorded pelvis pose, so the
root is fixed to the deployment standing assumption and that choice is written
into the provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..joints import canonical_hand, hand_order, sonic_reference_body
from ..reference import StandingPose
from ..schema import HAND_SCHEMA_VERIFIED, CanonicalEpisode, CanonicalEpisodeBuild
from ..state_reference import StateReference, build_state_reference
from ..timeline import finite_difference, linear_resample, uniform_timeline
from humanoid_lab.controllers.sonic import SONIC_REFERENCE_JOINT_ORDER

EXPECTED = ("left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand")
BODY_BLOCKS = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")
SOURCE_FPS = 20.0


def action_slices(dataset: Path) -> dict[str, slice]:
    modality = json.loads((dataset / "meta/modality.json").read_text(encoding="utf-8"))["action"]
    if tuple(modality) != EXPECTED:
        raise ValueError(f"unexpected Fruits action block order: {tuple(modality)}")
    result = {name: slice(int(spec["start"]), int(spec["end"])) for name, spec in modality.items()}
    if result["right_hand"].stop != 43:
        raise ValueError("Fruits action must be 43D")
    return result


def episode_path(dataset: Path, episode_index: int) -> Path:
    return dataset / f"data/chunk-000/episode_{episode_index:06d}.parquet"


def _read_episode(dataset: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, slice]]:
    """Return action, observation.state, timestamps and the declared slices."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("Fruits conversion requires pyarrow") from exc

    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    action_names = tuple(info["features"]["action"]["names"])
    state_names = tuple(info["features"]["observation.state"]["names"])
    if len(action_names) != 43 or len(state_names) != 43:
        raise ValueError("Fruits action and state must each declare 43 named channels")
    if float(info["fps"]) != SOURCE_FPS:
        raise ValueError(f"unexpected Fruits fps: {info['fps']}")
    slices = action_slices(dataset)
    path = episode_path(dataset, episode_index)
    if not path.is_file():
        raise FileNotFoundError(f"Fruits episode parquet missing: {path}")
    table = pq.read_table(path, columns=["action", "observation.state", "timestamp"])
    if table.num_rows < 2:
        raise ValueError(f"Fruits episode {episode_index} has {table.num_rows} rows")
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64).reshape(-1)
    if len(timestamps) != len(action) or len(action) != len(state):
        raise ValueError("Fruits timestamp/action/state lengths differ")
    return action, state, timestamps, slices


def _decode(values: np.ndarray, names: tuple[str, ...], slices: dict[str, slice]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a 43D Fruits row set into canonical body29 and Dex3 hands."""
    body_source = np.concatenate([values[:, slices[block]] for block in BODY_BLOCKS], axis=1)
    body_names = tuple(name for block in BODY_BLOCKS for name in names[slices[block]])
    body = sonic_reference_body(body_source, body_names)
    left = canonical_hand(values[:, slices["left_hand"]], tuple(names[slices["left_hand"]]), "left")
    right = canonical_hand(values[:, slices["right_hand"]], tuple(names[slices["right_hand"]]), "right")
    return body, left, right


def load_state_reference(
    dataset: Path,
    episode_index: int,
    *,
    limits: dict[str, tuple[float, float]],
) -> StateReference:
    """Measured ``observation.state`` at 20 Hz, decoded into canonical orders."""
    _, state, timestamps, slices = _read_episode(dataset, episode_index)
    names = tuple(json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))["features"]["observation.state"]["names"])
    body, left, right = _decode(state, names, slices)
    return build_state_reference(
        timestamps,
        body,
        left,
        right,
        source_field="observation.state",
        source_fps=SOURCE_FPS,
        limits=limits,
        mapping={
            "body_joint_order_source": "observation.state names, remapped by name",
            "hand_joint_order_source": "observation.state names -> canonical Dex3 motor order",
            "hand_units": "radians (measured qpos)",
            "blocks": {name: [slices[name].start, slices[name].stop] for name in EXPECTED},
        },
    )


def load_action_source_reference(dataset: Path, episode_index: int) -> StateReference:
    """The action trajectory at source rate (row i <-> frame i) for kinematic replay."""
    action, _, timestamps, slices = _read_episode(dataset, episode_index)
    names = tuple(json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))["features"]["action"]["names"])
    body_names = tuple(name for block in BODY_BLOCKS for name in names[slices[block]])
    body, left, right = _decode(action, names, slices)
    velocity = np.zeros_like(body)
    velocity[:-1] = np.diff(body, axis=0) * SOURCE_FPS
    reference = StateReference(
        timestamps=timestamps,
        joint_pos=body.astype(np.float32),
        joint_vel=velocity.astype(np.float32),
        left_hand_joints=left.astype(np.float32),
        right_hand_joints=right.astype(np.float32),
        hands_applied=True,
        provenance={
            "source_field": "action",
            "semantics": "desired joint positions (command), 20 Hz, row i <-> frame i",
            "source_fps": SOURCE_FPS,
            "velocity_source": "finite difference of the action trajectory at source rate",
            "body_joint_order": "official SONIC reference / IsaacLab",
            "hand_joint_order": "canonical Dex3 motor order",
            "hand_units": "radians",
            "note": "this is the trajectory the encoder observation is built from, not the 50 Hz reference",
        },
    )
    reference.validate()
    return reference


def load_episode(
    dataset: Path,
    episode_index: int,
    *,
    standing: StandingPose,
) -> CanonicalEpisodeBuild:
    """Load one Fruits episode as the canonical 50 Hz whole-body reference."""
    action, _, timestamps, slices = _read_episode(dataset, episode_index)
    names = tuple(json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))["features"]["action"]["names"])
    body_names = tuple(name for block in BODY_BLOCKS for name in names[slices[block]])
    body, left, right = _decode(action, names, slices)

    target = uniform_timeline(timestamps)
    body_resampled = linear_resample(timestamps, body, target)
    episode: CanonicalEpisode = CanonicalEpisode(
        timestamps=target,
        joint_pos=body_resampled.astype(np.float32),
        joint_vel=finite_difference(body_resampled).astype(np.float32),
        body_quat_wxyz=np.tile(standing.body_quat_wxyz.astype(np.float32), (len(target), 1)),
        body_pos=np.tile(standing.body_pos.astype(np.float32), (len(target), 1)),
        left_hand_joints=linear_resample(timestamps, left, target).astype(np.float32),
        right_hand_joints=linear_resample(timestamps, right, target).astype(np.float32),
    )
    episode.validate()
    provenance = {
        "source_collection": "nvidia_g1_fruits_1k",
        "source_kind": "lerobot_v2.1",
        "source_fps": SOURCE_FPS,
        "processed_fps": 50.0,
        "source_semantics": "43D action = desired whole-body joint positions",
        "source_modality_blocks": {name: [slices[name].start, slices[name].stop] for name in EXPECTED},
        "body_joint_order_source": "info.json action names, remapped by name",
        "hand_joint_order_source": "info.json action names -> canonical Dex3 motor order",
        "name_check": {
            "body": {
                "mode": "name_round_trip",
                "source": list(body_names),
                "target": list(SONIC_REFERENCE_JOINT_ORDER),
            },
            "left_hand": {
                "mode": "name_round_trip",
                "source": list(names[slices["left_hand"]]),
                "target": list(hand_order("left")),
            },
            "right_hand": {
                "mode": "name_round_trip",
                "source": list(names[slices["right_hand"]]),
                "target": list(hand_order("right")),
            },
        },
        "velocity_source": "re-derived by finite difference of the resampled 50 Hz positions",
        "resampling": "linear interpolation of the action trajectory onto a uniform 50 Hz timeline",
        "root_assumption": "synthetic_static_standing_root",
        "root_assumption_detail": {
            "body_pos": [float(value) for value in standing.body_pos],
            "body_quat_wxyz": [float(value) for value in standing.body_quat_wxyz],
            "source": standing.source,
            "reason": "the collection records no pelvis pose; stationary manipulation is assumed",
        },
        "state_sources": {
            "left_leg": "measured_desired_position",
            "right_leg": "measured_desired_position",
            "waist": "measured_desired_position",
            "left_arm": "measured_desired_position",
            "right_arm": "measured_desired_position",
            "left_hand": "measured_desired_position",
            "right_hand": "measured_desired_position",
            "root_position": "synthetic_static_standing_root",
            "root_orientation": "synthetic_upright_fixed",
            "joint_velocities": "derived_from_resampled_positions",
        },
        "episode_frames": int(action.shape[0]),
    }
    return CanonicalEpisodeBuild(
        episode=episode,
        provenance=provenance,
        hand_schema_status=HAND_SCHEMA_VERIFIED,
        hand_schema_reason="Fruits metadata names every hand channel; remap is name-based",
        notes=("root pose is synthetic; treat root-tracking metrics as not applicable",),
    )
