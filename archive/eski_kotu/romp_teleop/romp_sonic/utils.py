"""Small shared helpers for the ROMP -> SONIC bridge.

Only numpy is used here so this module imports in both the ROMP env and the
SONIC (container) env.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

# --- SMPL-24 kinematic tree (standard SMPL joint order) ---------------------
# 0 pelvis, 1 lhip, 2 rhip, 3 spine1, 4 lknee, 5 rknee, 6 spine2,
# 7 lankle, 8 rankle, 9 spine3, 10 lfoot, 11 rfoot, 12 neck, 13 lcollar,
# 14 rcollar, 15 head, 16 lshoulder, 17 rshoulder, 18 lelbow, 19 relbow,
# 20 lwrist, 21 rwrist, 22 lhand, 23 rhand
SMPL24_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
SMPL24_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                  16, 17, 18, 19, 20, 21]
SMPL24_EDGES = [(c, p) for c, p in enumerate(SMPL24_PARENTS) if p >= 0]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Rate:
    """Simple rolling-rate meter + periodic status printer."""

    def __init__(self, every: float = 1.0):
        self.every = every
        self.t0 = time.time()
        self.n = 0
        self.last = self.t0
        self.fps = 0.0

    def tick(self) -> None:
        self.n += 1
        now = time.time()
        if now - self.last >= self.every:
            self.fps = self.n / (now - self.last)
            self.n = 0
            self.last = now


def rodrigues_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle -> 3x3 rotation matrix (numpy, no scipy dependency)."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rotvec / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


# ROMP represents the body with ~180deg about X baked into global_orient
# (its camera convention is Y-down); SONIC's SMPL convention is Y-up. Applying
# this flip to ROMP's global orientation makes the skeleton world-upright.
ROMP_FLIP_X = np.diag([1.0, -1.0, -1.0])


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle (numpy)."""
    R = np.asarray(R, dtype=np.float64)
    cos_th = (np.trace(R) - 1.0) / 2.0
    cos_th = max(-1.0, min(1.0, cos_th))
    theta = math.acos(cos_th)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2.0 * math.sin(theta))
    return (axis * theta).astype(np.float32)


def flip_romp_global_orient(global_orient: np.ndarray) -> np.ndarray:
    """Convert ROMP's Y-down root orientation into SMPL Y-up convention."""
    return matrix_to_axis_angle(ROMP_FLIP_X @ rodrigues_to_matrix(global_orient))


def yaw_only_axis_angle(global_orient: np.ndarray, axis: int = 1) -> np.ndarray:
    """Keep only the rotation about `axis` (SMPL is Y-up => yaw about Y).

    Returns an axis-angle vector with a single component on `axis`.
    """
    R = rodrigues_to_matrix(global_orient)
    # Yaw about Y: for R about Y, yaw = atan2(R[0,2], R[0,0]).
    if axis == 1:
        yaw = math.atan2(R[0, 2], R[0, 0])
        return np.array([0.0, yaw, 0.0], dtype=np.float32)
    if axis == 2:
        yaw = math.atan2(R[1, 0], R[0, 0])
        return np.array([0.0, 0.0, yaw], dtype=np.float32)
    # axis == 0
    yaw = math.atan2(R[2, 1], R[1, 1])
    return np.array([yaw, 0.0, 0.0], dtype=np.float32)


def select_largest_person(cam: np.ndarray) -> int:
    """ROMP `cam` is [N,3] = (scale, tx, ty). Largest scale == closest person."""
    return int(np.argmax(np.asarray(cam)[:, 0]))
