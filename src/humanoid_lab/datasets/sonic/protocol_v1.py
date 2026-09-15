"""Official SONIC packed ZMQ Protocol v1 helpers."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator

import numpy as np

from .schema import CanonicalEpisode

HEADER_SIZE = 1280
LOOKAHEAD_FRAMES = 46


def pack_command_message(*, start: bool, stop: bool, planner: bool) -> bytes:
    descriptors = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    header = json.dumps({"v": 1, "endian": "le", "count": 1, "fields": descriptors}, separators=(",", ":")).encode()
    return b"command" + header.ljust(HEADER_SIZE, b"\0") + struct.pack("BBB", start, stop, planner)


def pack_pose_message(episode: CanonicalEpisode, frame_indices: np.ndarray) -> bytes:
    episode.validate()
    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < LOOKAHEAD_FRAMES:
        raise ValueError(f"Protocol v1 requires at least {LOOKAHEAD_FRAMES} staged frames")
    if np.any(np.diff(indices) != 1):
        raise ValueError("frame_index must be contiguous and strictly increasing")
    fields = {
        "joint_pos": np.ascontiguousarray(episode.joint_pos, dtype=np.float32),
        "joint_vel": np.ascontiguousarray(episode.joint_vel, dtype=np.float32),
        "body_quat_w": np.ascontiguousarray(episode.body_quat_wxyz, dtype=np.float32),
        "frame_index": np.ascontiguousarray(indices, dtype=np.int64),
        "left_hand_joints": np.ascontiguousarray(episode.left_hand_joints[-1], dtype=np.float32),
        "right_hand_joints": np.ascontiguousarray(episode.right_hand_joints[-1], dtype=np.float32),
    }
    if any(value.shape[0] != len(indices) for name, value in fields.items() if name in {"joint_pos", "joint_vel", "body_quat_w"}):
        raise ValueError("pose arrays and frame_index length differ")
    dtype_names = {np.dtype("float32"): "f32", np.dtype("int64"): "i64"}
    descriptors = [{"name": name, "dtype": dtype_names[value.dtype], "shape": list(value.shape)} for name, value in fields.items()]
    header = json.dumps({"v": 1, "endian": "le", "count": 1, "fields": descriptors}, separators=(",", ":")).encode()
    if len(header) > HEADER_SIZE:
        raise ValueError("Protocol v1 header exceeds fixed size")
    return b"pose" + header.ljust(HEADER_SIZE, b"\0") + b"".join(value.tobytes() for value in fields.values())


def staged_windows(episode: CanonicalEpisode, window: int = LOOKAHEAD_FRAMES) -> Iterator[tuple[np.ndarray, CanonicalEpisode]]:
    """Yield full future windows, clamping only after the source tail."""
    episode.validate()
    n = len(episode.timestamps)
    for start in range(n):
        source = np.minimum(np.arange(start, start + window), n - 1)
        yield np.arange(start, start + window, dtype=np.int64), CanonicalEpisode(
            timestamps=episode.timestamps[start] + np.arange(window, dtype=np.float64) / 50.0,
            joint_pos=episode.joint_pos[source],
            joint_vel=episode.joint_vel[source], body_quat_wxyz=episode.body_quat_wxyz[source],
            body_pos=episode.body_pos[source], left_hand_joints=episode.left_hand_joints[source],
            right_hand_joints=episode.right_hand_joints[source],
        )
