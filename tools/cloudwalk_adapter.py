"""CloudWalk-specific constants and compatibility exports.

The sole action, protocol, scheduling, lifecycle, and hand-boundary implementation
is :mod:`sonic_isaac_inspire_adapter`.
"""
from __future__ import annotations

from typing import Sequence

from sonic_isaac_inspire_adapter import (  # re-exported for existing callers
    ACTION_HORIZON, ACTION_RATE_HZ, ACTION_SIZE, HAND_ACTION_SIZE as HAND_SIZE,
    INFERENCE_RATE_HZ, MOTION_TOKEN_SIZE, ContractError, V4Action, V4Action as ActionStep,
    pack_protocol_v4, split_groot_action_chunk as split_action_chunk,
)

PROMPT = "grab the bottle"
EMBODIMENT = "UNITREE_G1_SONIC"
CHECKPOINT_REVISION = "f980fd880c92b984d65dcc3f9dc82b98bc8009c"
G1_BODY_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
LEFT_INSPIRE_FTP_JOINTS = ("left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint")
RIGHT_INSPIRE_FTP_JOINTS = tuple(name.replace("left_", "right_", 1) for name in LEFT_INSPIRE_FTP_JOINTS)
G1_INSPIRE_FTP_JOINTS = G1_BODY_JOINTS[:22] + LEFT_INSPIRE_FTP_JOINTS + G1_BODY_JOINTS[22:] + RIGHT_INSPIRE_FTP_JOINTS


def validate_observation(image: object, state: Sequence[float], prompt: str) -> None:
    shape = tuple(getattr(image, "shape", ()))
    if not shape and isinstance(image, Sequence) and image and isinstance(image[0], Sequence) and image[0] and isinstance(image[0][0], Sequence):
        shape = (len(image), len(image[0]), len(image[0][0]))
    if shape != (480, 640, 3):
        raise ContractError(f"ego RGB must have shape [480,640,3], got {shape}")
    if len(state) != 43:
        raise ContractError(f"G1 Inspire FTP state must have 43 values, got {len(state)}")
    V4Action((0.125,) * 64, tuple(state[22:29]), tuple(state[36:43]))
    if prompt != PROMPT:
        raise ContractError(f"CloudWalk checkpoint requires literal prompt {PROMPT!r}")
