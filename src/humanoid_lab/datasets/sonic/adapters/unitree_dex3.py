"""Unitree Dex3 v3 semantic field resolver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..joints import LEFT_HAND_ORDER, RIGHT_HAND_ORDER, reorder
from ..reference import compose_unitree_standing_reference
from ..schema import CanonicalEpisode

UNITREE_ACTION_NAMES = (
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw", "kRightShoulderPitch",
    "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow", "kRightWristRoll",
    "kRightWristPitch", "kRightWristYaw", "kLeftHandThumb0", "kLeftHandThumb1",
    "kLeftHandThumb2", "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0",
    "kLeftHandIndex1", "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
)


def validate_metadata(dataset: Path) -> dict[str, object]:
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    feature = info["features"]["action"]
    names = tuple(feature["names"][0])
    if names != UNITREE_ACTION_NAMES or feature["shape"] != [28]:
        raise ValueError("unexpected Unitree Dex3 action schema")
    if float(info["fps"]) != 30.0:
        raise ValueError("unexpected Unitree Dex3 fps")
    return {"fps": 30.0, "action_names": names, "total_episodes": int(info["total_episodes"])}


def load_episode(dataset: Path, episode_index: int, idle_reference: Path) -> CanonicalEpisode:
    """Load same-row desired actions and return the canonical 50 Hz reference."""
    metadata = validate_metadata(dataset)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("Unitree conversion requires pyarrow") from exc

    episode_files = sorted((dataset / "meta/episodes").glob("chunk-*/*.parquet"))
    matches: list[dict[str, object]] = []
    for path in episode_files:
        table = pq.read_table(path)
        matches.extend(row for row in table.to_pylist() if int(row["episode_index"]) == episode_index)
    if len(matches) != 1:
        raise ValueError(f"episode {episode_index} resolved to {len(matches)} metadata rows")
    row = matches[0]
    data_path = dataset / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    table = pq.read_table(data_path, columns=["episode_index", "timestamp", "action"]).slice(start, stop - start)
    rows = table.to_pylist()
    if not rows or any(int(item["episode_index"]) != episode_index for item in rows):
        raise ValueError("episode metadata does not match data rows")
    action = np.asarray([item["action"] for item in rows], dtype=np.float32)
    timestamps = np.asarray([item["timestamp"] for item in rows], dtype=np.float64)
    names = tuple(metadata["action_names"])
    arm_names = UNITREE_ACTION_NAMES[:14]
    arms = reorder(action, names, arm_names)
    left_source = tuple("kLeftHand" + name.replace("_joint", "").title().replace("_", "") for name in LEFT_HAND_ORDER)
    right_source = tuple("kRightHand" + name.replace("_joint", "").title().replace("_", "") for name in RIGHT_HAND_ORDER)
    # Unitree metadata uses Thumb/Middle/Index with numeric suffixes.
    normalize = lambda value: value.replace("Thumb", "Thumb").replace("Middle", "Middle").replace("Index", "Index")
    left = reorder(action, names, tuple(normalize(name) for name in left_source))
    right = reorder(action, names, tuple(normalize(name) for name in right_source))
    idle = {
        "idle_timestamps": np.loadtxt(idle_reference / "timestamps.csv", delimiter=",", ndmin=1),
        "idle_joint_pos": np.loadtxt(idle_reference / "joint_pos.csv", delimiter=",", ndmin=2),
        "idle_joint_vel": np.loadtxt(idle_reference / "joint_vel.csv", delimiter=",", ndmin=2),
        "idle_body_pos": np.loadtxt(idle_reference / "body_pos.csv", delimiter=",", ndmin=2),
        "idle_body_quat_wxyz": np.loadtxt(idle_reference / "body_quat.csv", delimiter=",", ndmin=2),
    }
    return compose_unitree_standing_reference(timestamps, arms, left, right, **idle)
