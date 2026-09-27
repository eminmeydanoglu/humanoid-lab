"""Joint orders shared by the simulator's Unitree topics and the G1 URDF.

The simulator's 29 body motors arrive in `/lowstate` already in the URDF's own
joint order, so the body needs no remapping.  The Dex3 hands do: the wire slots
are numbered from the physical finger layout, while the URDF names the left hand
middle-then-index and the right hand index-then-middle, so each side keeps its
own list here.

These names are the simulator's (`humanoid_lab.controllers.sonic`); the
repository test `tests/test_flux_sim_viz.py` fails if they ever drift apart.
"""

BODY_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

LEFT_HAND_JOINT_ORDER: tuple[str, ...] = (
    "thumb_0_joint",
    "thumb_1_joint",
    "thumb_2_joint",
    "middle_0_joint",
    "middle_1_joint",
    "index_0_joint",
    "index_1_joint",
)

RIGHT_HAND_JOINT_ORDER: tuple[str, ...] = (
    "thumb_0_joint",
    "thumb_1_joint",
    "thumb_2_joint",
    "index_0_joint",
    "index_1_joint",
    "middle_0_joint",
    "middle_1_joint",
)

HAND_MOTOR_SLOTS = 7


def hand_joint_names(side: str) -> tuple[str, ...]:
    """The URDF joint names of one Dex3 hand, in the wire's motor-slot order."""
    if side == "left":
        prefix, order = "left_hand", LEFT_HAND_JOINT_ORDER
    elif side == "right":
        prefix, order = "right_hand", RIGHT_HAND_JOINT_ORDER
    else:
        raise ValueError(f"side must be left or right, got {side!r}")
    return tuple(f"{prefix}_{name}" for name in order)


def measured_joint_state(body_q, left_q=None, right_q=None):
    """Names and positions of the measured state, ready for `sensor_msgs/JointState`.

    A hand whose slots have not arrived yet is left out rather than guessed at,
    so the first messages carry the body only.
    """
    names = list(BODY_JOINT_ORDER)
    positions = [float(value) for value in body_q]
    for side, hand_q in (("left", left_q), ("right", right_q)):
        if hand_q is None or len(hand_q) != HAND_MOTOR_SLOTS:
            continue
        names.extend(hand_joint_names(side))
        positions.extend(float(value) for value in hand_q)
    return names, positions
