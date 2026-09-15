"""Compose Unitree arm motion with an official SONIC IDLE trajectory."""

from __future__ import annotations

import numpy as np

from humanoid_lab.controllers.sonic import SONIC_REFERENCE_JOINT_ORDER

from .schema import CanonicalEpisode
from .timeline import finite_difference, linear_resample, uniform_timeline

ARM_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def compose_unitree_standing_reference(
    timestamps: np.ndarray,
    arms: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    *,
    idle_timestamps: np.ndarray,
    idle_joint_pos: np.ndarray,
    idle_joint_vel: np.ndarray,
    idle_body_pos: np.ndarray,
    idle_body_quat_wxyz: np.ndarray,
) -> CanonicalEpisode:
    if np.asarray(arms).shape[1:] != (14,):
        raise ValueError("Unitree arm action must be 14D")
    target = uniform_timeline(timestamps)
    idle_ts = np.asarray(idle_timestamps, dtype=np.float64)
    idle_target = idle_ts[0] + (target - target[0])
    if idle_target[-1] > idle_ts[-1] + 1e-9:
        raise ValueError("SONIC IDLE trajectory is shorter than the resampled demonstration")
    q = linear_resample(idle_ts, idle_joint_pos, idle_target)
    qd = linear_resample(idle_ts, idle_joint_vel, idle_target)
    arm_ids = [SONIC_REFERENCE_JOINT_ORDER.index(name) for name in ARM_NAMES]
    arm_q = linear_resample(timestamps, arms, target)
    q[:, arm_ids] = arm_q
    qd[:, arm_ids] = finite_difference(arm_q)

    quat_source = np.asarray(idle_body_quat_wxyz, dtype=np.float64).copy()
    if quat_source.shape != (len(idle_ts), 4):
        raise ValueError("idle_body_quat_wxyz must be [frames, 4]")
    # Keep interpolation on one quaternion hemisphere, then renormalize.
    for index in range(1, len(quat_source)):
        if np.dot(quat_source[index - 1], quat_source[index]) < 0:
            quat_source[index] *= -1
    body_quat = linear_resample(idle_ts, quat_source, idle_target)
    norms = np.linalg.norm(body_quat, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("SONIC IDLE trajectory contains an invalid quaternion")
    body_quat = (body_quat / norms).astype(np.float32)
    episode = CanonicalEpisode(
        timestamps=target,
        joint_pos=q,
        joint_vel=qd,
        body_quat_wxyz=body_quat,
        body_pos=linear_resample(idle_ts, idle_body_pos, idle_target),
        left_hand_joints=linear_resample(timestamps, left_hand, target),
        right_hand_joints=linear_resample(timestamps, right_hand, target),
    )
    episode.validate()
    return episode
