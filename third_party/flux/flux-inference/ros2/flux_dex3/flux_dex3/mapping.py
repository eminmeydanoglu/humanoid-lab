"""The measured/action order shared by the Dex3 index and Unitree motor arrays."""

import math

JOINT_NAMES = (
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
)
ARM_MOTORS = tuple(range(15, 29))
HAND_MOTORS = tuple(range(7))


def measured_state(lowstate, left, right):
    """Extract measured q values, with no motor command or unit conversion."""
    if len(lowstate.motor_state) < 29 or len(left.motor_state) != 7 or len(right.motor_state) != 7:
        raise ValueError("expected 29 body and seven motors per hand")
    values = tuple(lowstate.motor_state[i].q for i in ARM_MOTORS)
    values += tuple(left.motor_state[i].q for i in HAND_MOTORS)
    values += tuple(right.motor_state[i].q for i in HAND_MOTORS)
    if len(values) != 28 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("invalid measured joint position")
    return values


def split_action(action):
    """Return named motor-index targets for the three command topics."""
    if len(action) != 28 or not all(math.isfinite(float(value)) for value in action):
        raise ValueError("expected 28 finite joint targets")
    return (
        dict(zip(ARM_MOTORS, action[:14])),
        dict(zip(HAND_MOTORS, action[14:21])),
        dict(zip(HAND_MOTORS, action[21:])),
    )
