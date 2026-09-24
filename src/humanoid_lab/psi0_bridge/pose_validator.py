"""Strict validation of the one-frame SONIC Protocol v4 policy pose packet."""

from __future__ import annotations

import json

import numpy as np

from humanoid_lab.datasets.sonic.protocol_v4 import HEADER_SIZE


class PosePacketError(ValueError):
    """A policy packet cannot safely be forwarded to SONIC."""


_FIELDS = (
    ("token_state", "f32", [1, 64], "<f4"),
    ("frame_index", "i64", [1], "<i8"),
    ("left_hand_joints", "f32", [1, 7], "<f4"),
    ("right_hand_joints", "f32", [1, 7], "<f4"),
)


def validate_pose_packet(packet: bytes) -> int:
    """Return its frame index, or reject a malformed/incomplete 64D + Dex3 pose.

    This enforces the exact four-field wire layout used by the production
    action adapters. In particular, a valid body token without *both* hands
    is insufficient for a real G1 Dex3 policy action.
    """
    if not isinstance(packet, bytes):
        raise PosePacketError("pose packet must be bytes")
    prefix_size = 4 + HEADER_SIZE
    if len(packet) < prefix_size or packet[:4] != b"pose":
        raise PosePacketError("missing pose topic or fixed Protocol v4 header")
    header_bytes = packet[4:prefix_size]
    header_json, separator, padding = header_bytes.partition(b"\0")
    if not separator or not header_json or any(padding):
        raise PosePacketError("invalid fixed header padding")
    try:
        header = json.loads(header_json.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PosePacketError("invalid Protocol v4 JSON header") from exc
    expected = {
        "v": 4, "endian": "le", "count": 1,
        "fields": [
            {"name": name, "dtype": dtype, "shape": shape}
            for name, dtype, shape, _ in _FIELDS
        ],
    }
    if (header != expected or type(header.get("v")) is not int
            or type(header.get("count")) is not int
            or any(type(size) is not int for field in header.get("fields", [])
                   for size in field.get("shape", []))):
        raise PosePacketError("unexpected Protocol v4 fields or metadata")
    expected_size = prefix_size + sum(np.dtype(wire_dtype).itemsize * int(np.prod(shape))
                                      for _, _, shape, wire_dtype in _FIELDS)
    if len(packet) != expected_size:
        raise PosePacketError(f"pose packet length {len(packet)} != {expected_size}")
    offset = prefix_size
    index = -1
    for name, _, shape, wire_dtype in _FIELDS:
        size = np.dtype(wire_dtype).itemsize * int(np.prod(shape))
        values = np.frombuffer(packet, dtype=wire_dtype, count=int(np.prod(shape)), offset=offset)
        offset += size
        if name == "frame_index":
            index = int(values[0])
            if index < 0:
                raise PosePacketError("frame_index must be nonnegative")
        elif not np.isfinite(values).all():
            raise PosePacketError(f"{name} contains nonfinite values")
    return index
