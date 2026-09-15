"""Official SONIC latent ZMQ Protocol v4 packing helpers."""

from __future__ import annotations

import json

import numpy as np

HEADER_SIZE = 1280


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: int | np.ndarray,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
) -> bytes:
    """Pack one 64D motion token and optional Dex3 targets on the ``pose`` topic."""
    token = np.asarray(motion_token, dtype=np.float32)
    if token.shape == (64,):
        token = token.reshape(1, 64)
    if token.shape != (1, 64) or not np.isfinite(token).all():
        raise ValueError(f"motion_token must be finite [64] or [1,64], got {token.shape}")

    index = np.asarray(frame_index, dtype=np.int64).reshape(-1)
    if index.shape != (1,):
        raise ValueError(f"frame_index must contain one value, got {index.shape}")

    fields: dict[str, np.ndarray] = {
        "token_state": np.ascontiguousarray(token),
        "frame_index": np.ascontiguousarray(index),
    }
    for name, value in (
        ("left_hand_joints", left_hand_joints),
        ("right_hand_joints", right_hand_joints),
    ):
        if value is None:
            continue
        hand = np.asarray(value, dtype=np.float32)
        if hand.shape == (7,):
            hand = hand.reshape(1, 7)
        if hand.shape != (1, 7) or not np.isfinite(hand).all():
            raise ValueError(f"{name} must be finite [7] or [1,7], got {hand.shape}")
        fields[name] = np.ascontiguousarray(hand)

    dtype_names = {np.dtype("float32"): "f32", np.dtype("int64"): "i64"}
    descriptors = [
        {"name": name, "dtype": dtype_names[value.dtype], "shape": list(value.shape)}
        for name, value in fields.items()
    ]
    header = json.dumps(
        {"v": 4, "endian": "le", "count": 1, "fields": descriptors},
        separators=(",", ":"),
    ).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ValueError("Protocol v4 header exceeds fixed size")
    return b"pose" + header.ljust(HEADER_SIZE, b"\0") + b"".join(
        value.tobytes() for value in fields.values()
    )
