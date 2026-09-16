"""Unitree Dex3 v3 semantic field resolver.

The collection records only upper-body teleoperation: ``action`` holds the
desired arm and Dex3 hand positions of the *same* row (the collector writes the
IK solution it sends to the robot into the action column).  There is no leg,
waist or root recording, so the lower body is completed with one validated
standing frame and the root with an explicit upright assumption
(:mod:`humanoid_lab.datasets.sonic.reference`).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..joints import LEFT_HAND_ORDER, RIGHT_HAND_ORDER, reorder
from ..reference import STANDING_COMPLETION_SCOPE, StandingPose, compose_unitree_static_completion
from ..schema import HAND_SCHEMA_VERIFIED, CanonicalEpisode, CanonicalEpisodeBuild

UNITREE_ACTION_NAMES = (
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw", "kRightShoulderPitch",
    "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow", "kRightWristRoll",
    "kRightWristPitch", "kRightWristYaw", "kLeftHandThumb0", "kLeftHandThumb1",
    "kLeftHandThumb2", "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0",
    "kLeftHandIndex1", "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
)

ARM_SOURCE_NAMES = UNITREE_ACTION_NAMES[:14]


def _unitree_hand_field(side: str, canonical_name: str) -> str:
    """``thumb_0_joint`` -> ``kLeftHandThumb0`` for the declared side."""
    prefix = "kLeftHand" if side == "left" else "kRightHand"
    parts = canonical_name.replace("_joint", "").split("_")
    return prefix + "".join(part.title() for part in parts)


#: Explicit source-name -> canonical-name map; no string munging at load time.
HAND_SOURCE_NAMES: dict[str, tuple[str, ...]] = {
    side: tuple(
        _unitree_hand_field(side, name)
        for name in (LEFT_HAND_ORDER if side == "left" else RIGHT_HAND_ORDER)
    )
    for side in ("left", "right")
}


def validate_metadata(dataset: Path) -> dict[str, object]:
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    feature = info["features"]["action"]
    names = tuple(feature["names"][0])
    if names != UNITREE_ACTION_NAMES or feature["shape"] != [28]:
        raise ValueError("unexpected Unitree Dex3 action schema")
    if float(info["fps"]) != 30.0:
        raise ValueError("unexpected Unitree Dex3 fps")
    declared = {name for side in HAND_SOURCE_NAMES.values() for name in side}
    unknown = sorted(declared - set(UNITREE_ACTION_NAMES))
    if unknown:
        raise ValueError(f"hand remap references unknown Unitree fields: {unknown}")
    return {"fps": 30.0, "action_names": names, "total_episodes": int(info["total_episodes"])}


@lru_cache(maxsize=32)
def episode_row_index(dataset: Path) -> dict[int, dict[str, object]]:
    """Read LeRobot episode metadata once per collection, not once per episode."""
    import pyarrow.parquet as pq

    rows: dict[int, dict[str, object]] = {}
    for path in sorted((Path(dataset) / "meta/episodes").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            index = int(row["episode_index"])
            if index in rows:
                raise ValueError(f"duplicate Unitree episode metadata row {index}")
            rows[index] = row
    return rows


def resolve_episode_rows(dataset: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``action`` and ``timestamp`` of one episode as stored."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("Unitree conversion requires pyarrow") from exc

    try:
        row = episode_row_index(Path(dataset))[episode_index]
    except KeyError as error:
        raise ValueError(f"episode {episode_index} has no metadata row") from error
    data_path = dataset / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    table = pq.read_table(data_path, columns=["episode_index", "timestamp", "action"]).slice(start, stop - start)
    rows = table.to_pylist()
    if not rows or any(int(item["episode_index"]) != episode_index for item in rows):
        raise ValueError("episode metadata does not match data rows")
    action = np.asarray([item["action"] for item in rows], dtype=np.float32)
    timestamps = np.asarray([item["timestamp"] for item in rows], dtype=np.float64)
    return action, timestamps


def load_episode(dataset: Path, episode_index: int, standing: StandingPose) -> CanonicalEpisodeBuild:
    """Load same-row desired actions and return the canonical 50 Hz reference."""
    metadata = validate_metadata(dataset)
    action, timestamps = resolve_episode_rows(dataset, episode_index)
    names = tuple(metadata["action_names"])
    arms = reorder(action, names, ARM_SOURCE_NAMES)
    hands = {
        side: reorder(action, names, HAND_SOURCE_NAMES[side])
        for side in ("left", "right")
    }
    episode: CanonicalEpisode = compose_unitree_static_completion(
        timestamps, arms, hands["left"], hands["right"], standing=standing
    )
    provenance = {
        "source_collection": "unitree_dex3",
        "source_kind": "lerobot_v3",
        "source_fps": 30.0,
        "processed_fps": 50.0,
        "source_semantics": "same-row desired action (IK solution sent to the robot)",
        "source_indexing": "no shift; action and state are written in the same control tick",
        "body_joint_order": "official SONIC reference / IsaacLab (name-based remap)",
        "standing_completion_policy": "static_stable_frame",
        "standing_completion_scope": STANDING_COMPLETION_SCOPE,
        "standing_completion_source": standing.source,
        "standing_completion_provenance": standing.provenance,
        "velocity_source": "re-derived by finite difference of the composed 50 Hz positions",
        "state_sources": {
            "left_leg": "synthetic_static_standing",
            "right_leg": "synthetic_static_standing",
            "waist": "synthetic_static_standing",
            "left_arm": "measured_absolute_action",
            "right_arm": "measured_absolute_action",
            "left_hand": "measured_dex3_motor_order",
            "right_hand": "measured_dex3_motor_order",
            "root_position": "synthetic_static_standing",
            "root_orientation": "synthetic_upright_fixed",
            "joint_velocities": "derived_from_composed_positions",
        },
        "episode_frames": int(action.shape[0]),
        "name_check": {
            "body": "identity_by_construction: synthetic standing frame plus canonical arm names",
            "left_hand": "explicit_field_map: " + " -> ".join(HAND_SOURCE_NAMES["left"]),
            "right_hand": "explicit_field_map: " + " -> ".join(HAND_SOURCE_NAMES["right"]),
        },
        "limits": "arm/hand values are recorded desired positions, not a leg or waist trajectory",
    }
    notes = (
        "lower body, waist and root are synthetic for the whole episode; this pilot is valid for "
        "stationary manipulation only",
    )
    return CanonicalEpisodeBuild(
        episode=episode,
        provenance=provenance,
        hand_schema_status=HAND_SCHEMA_VERIFIED,
        hand_schema_reason="Unitree Dex3 metadata names every hand motor; canonical order is the motor order",
        notes=notes,
    )
