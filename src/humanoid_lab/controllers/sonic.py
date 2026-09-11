"""Pinned facts about the upstream SONIC deployment.

Everything in this module is copied from the pinned SONIC source
(``NVlabs/GR00T-WholeBodyControl`` at the commit in ``versions.lock.yaml``).
The numbers are not ours to tune: the deployment binary computes the very same
values internally, and changing them here would silently break simulator parity
with the official MuJoCo loop.
"""

from __future__ import annotations

# Canonical body order used by the deployment's motor commands (Unitree
# hardware / MuJoCo order).  Verified against
# gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml.
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

# Dex3 hand order, 7 joints per side, from the same pinned MJCF.
HAND_JOINT_ORDER: tuple[str, ...] = (
    "thumb_0_joint",
    "thumb_1_joint",
    "thumb_2_joint",
    "middle_0_joint",
    "middle_1_joint",
    "index_0_joint",
    "index_1_joint",
)

# Effort limits the official MuJoCo loop clips body torques to
# (motor_effort_limit_list in gear_sonic/utils/mujoco_sim/wbc_configs/
# g1_29dof_sonic_model12.yaml).  That 43-entry list follows the MJCF actuator
# order: 22 left/body joints, 7 left-hand joints, 7 right-arm joints, then 7
# right-hand joints.  The body limits therefore are *not* its first 29 entries.
BODY_EFFORT_LIMIT_NM: tuple[float, ...] = (
    88.0,
    88.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    88.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    50.0,
    50.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
)

HAND_EFFORT_LIMIT_NM: tuple[float, ...] = (
    2.45,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
)
RIGHT_HAND_EFFORT_LIMIT_NM = HAND_EFFORT_LIMIT_NM

# Passive joint dynamics in the compiled SONIC MuJoCo model. Isaac Sim 5.x
# expresses static/dynamic joint friction directly as an effort, so these map
# to MuJoCo's ``frictionloss`` values without a unit conversion.
MUJOCO_JOINT_ARMATURE = 0.01
MUJOCO_JOINT_VISCOUS_FRICTION = 0.05
MUJOCO_BODY_FRICTION_LOSS_NM: tuple[float, ...] = (
    *((0.2,) * 20),
    0.1,
    0.1,
    *((0.2,) * 5),
    0.1,
    0.1,
)
MUJOCO_HAND_FRICTION_LOSS_NM: tuple[float, ...] = (0.1,) * 7

# Unitree DDS topics touched by the official sim2sim loop.  The simulator stands
# in for the robot, so it publishes state and consumes commands.
LOW_STATE_TOPIC = "rt/lowstate"
LOW_COMMAND_TOPIC = "rt/lowcmd"
SECONDARY_IMU_TOPIC = "rt/secondary_imu"
LEFT_HAND_COMMAND_TOPIC = "rt/dex3/left/cmd"
RIGHT_HAND_COMMAND_TOPIC = "rt/dex3/right/cmd"
LEFT_HAND_STATE_TOPIC = "rt/dex3/left/state"
RIGHT_HAND_STATE_TOPIC = "rt/dex3/right/state"

# The deployment reads its planner version token from the path, not the file.
PLANNER_PATH_TOKENS: tuple[str, ...] = ("V0", "V1", "V2")
# Documented upstream switch for the simulation loop: no real motor produces the
# CRC a physical G1 would.
SIMULATION_ONLY_FLAGS: tuple[str, ...] = ("--disable-crc-check",)

# The deployment in versions.lock.yaml (models.sonic_deploy) was built from this
# commit with the simulation DDS domain moved off domain 0, which a physical G1
# uses.  Runs are loopback-only, so the same host cannot reach a real robot.
DEFAULT_DOMAIN_ID = 42
DEFAULT_INTERFACE = "lo"

# The deployment ramps the robot to this standing pose before accepting control
# (default_angles in policy_parameters.hpp), expressed in body order above.
DEFAULT_STANDING_POSE_RAD: dict[str, float] = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Motor family per body joint, in body order.  The deployment derives its
# position and damping gains from the armature of each family
# (policy_parameters.hpp: kps, kds, ARMATURE_*, NATURAL_FREQ, DAMPING_RATIO).
BODY_MOTOR_FAMILY: tuple[str, ...] = (
    "7520_22",
    "7520_22",
    "7520_14",
    "7520_22",
    "5020",
    "5020",
    "7520_22",
    "7520_22",
    "7520_14",
    "7520_22",
    "5020",
    "5020",
    "7520_14",
    "5020",
    "5020",
    "5020",
    "5020",
    "5020",
    "5020",
    "5020",
    "4010",
    "4010",
    "5020",
    "5020",
    "5020",
    "5020",
    "5020",
    "4010",
    "4010",
)

# The deployment doubles both Kp and Kd for the ankle pitch/roll joints and
# waist roll/pitch (policy_parameters.hpp).  These factors do not alter DDS
# commands received from the binary; they keep the local scripted controller
# and parity checks faithful to those commands.
BODY_GAIN_MULTIPLIER: tuple[float, ...] = (
    1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
    1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
    1.0, 2.0, 2.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
)

MOTOR_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
NATURAL_FREQUENCY_HZ = 10.0
DAMPING_RATIO = 2.0


def deploy_gains() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Position and damping gains the deployment sends with every low command."""
    import math

    omega = 2.0 * math.pi * NATURAL_FREQUENCY_HZ
    kp = tuple(
        multiplier * MOTOR_ARMATURE[family] * omega * omega
        for family, multiplier in zip(BODY_MOTOR_FAMILY, BODY_GAIN_MULTIPLIER)
    )
    kd = tuple(
        multiplier * 2.0 * DAMPING_RATIO * MOTOR_ARMATURE[family] * omega
        for family, multiplier in zip(BODY_MOTOR_FAMILY, BODY_GAIN_MULTIPLIER)
    )
    return kp, kd


def standing_pose() -> dict[str, float]:
    return dict(DEFAULT_STANDING_POSE_RAD)


# The root height of the standing pose, in metres. The pinned reference motions
# place the pelvis at 0.79 m (frame 0 of reference/example/squat_001__A359,
# body_pos.csv). Spawning at this height with the standing pose above puts the
# soles exactly on the ground, which is the state the deployment ramps to and
# the state the policy expects to take over from.
STANDING_ROOT_HEIGHT_M = 0.792563

NAMED_POSES: dict[str, tuple[dict[str, float], float]] = {
    "sonic_standing": (DEFAULT_STANDING_POSE_RAD, STANDING_ROOT_HEIGHT_M),
}
def named_pose(name: str) -> tuple[dict[str, float], float]:
    """Return a declared start-up pose and the root height it was authored for."""
    try:
        pose, height = NAMED_POSES[name]
    except KeyError as error:
        raise ValueError(f"unknown named pose {name!r}; expected one of {tuple(NAMED_POSES)}") from error
    return dict(pose), height


def hand_joint_names(side: str) -> tuple[str, ...]:
    prefix = "left_hand" if side == "left" else "right_hand"
    return tuple(f"{prefix}_{name}" for name in HAND_JOINT_ORDER)


def effort_limits_for(side: str) -> tuple[float, ...]:
    """Return the pinned effort limits for one hand group."""
    if side == "left":
        return HAND_EFFORT_LIMIT_NM
    if side == "right":
        return RIGHT_HAND_EFFORT_LIMIT_NM
    raise ValueError(f"unknown hand side {side!r}")
