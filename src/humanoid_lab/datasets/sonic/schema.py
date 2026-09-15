"""Canonical arrays shared by conversion, encoding, and replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BODY_DIM = 29
HAND_DIM = 7
SONIC_TOKEN_DIM = 64
STATE_DIM = 46
ACTION_DIM = 78
PROCESSED_FPS = 50.0


@dataclass(frozen=True)
class CanonicalEpisode:
    """One 50 Hz episode in SONIC's external reference contract.

    ``joint_pos`` and ``joint_vel`` are always in official SONIC
    reference/IsaacLab order.  Hand vectors retain the side-specific Unitree
    Dex3 motor order declared by the source metadata.
    """
    timestamps: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_quat_wxyz: np.ndarray
    body_pos: np.ndarray
    left_hand_joints: np.ndarray
    right_hand_joints: np.ndarray

    def validate(self) -> None:
        n = len(self.timestamps)
        expected = {
            "joint_pos": (n, BODY_DIM),
            "joint_vel": (n, BODY_DIM),
            "body_quat_wxyz": (n, 4),
            "body_pos": (n, 3),
            "left_hand_joints": (n, HAND_DIM),
            "right_hand_joints": (n, HAND_DIM),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        ts = np.asarray(self.timestamps)
        if ts.shape != (n,) or n == 0 or not np.isfinite(ts).all():
            raise ValueError("timestamps must be a non-empty finite vector")
        if n > 1 and not np.all(np.diff(ts) > 0):
            raise ValueError("timestamps must be strictly increasing")


def compose_action(motion_token: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in (motion_token, left, right))
    if arrays[0].shape[-1] != SONIC_TOKEN_DIM or arrays[1].shape[-1] != HAND_DIM or arrays[2].shape[-1] != HAND_DIM:
        raise ValueError("action components must end in 64, 7, and 7 values")
    result = np.concatenate(arrays, axis=-1)
    if result.shape[-1] != ACTION_DIM or not np.isfinite(result).all():
        raise ValueError("canonical action must be finite 78D")
    return result
