"""Runtime gate between passive Isaac physics and CloudWalk controller actuation."""
from __future__ import annotations

from enum import Enum


NATURAL_FREQUENCY = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
MOTOR_5020 = (0.003609725, 25.0)
MOTOR_7520_14 = (0.010177520, 88.0)
MOTOR_7520_22 = (0.025101925, 139.0)
MOTOR_4010 = (0.00425, 5.0)


def _sonic_motor(motor: tuple[float, float], gain_multiplier: float = 1.0) -> tuple[float, float, float, float]:
    armature, effort = motor
    stiffness = gain_multiplier * armature * NATURAL_FREQUENCY**2
    damping = gain_multiplier * 2.0 * DAMPING_RATIO * armature * NATURAL_FREQUENCY
    return stiffness, damping, armature, effort


def configure_sonic_actuators(robot_cfg: object) -> None:
    """Match the pinned SONIC G1 motor model while retaining Inspire hand actuators."""
    motors = {
        "7520_22": _sonic_motor(MOTOR_7520_22),
        "7520_14": _sonic_motor(MOTOR_7520_14),
        "5020": _sonic_motor(MOTOR_5020),
        "5020_double": _sonic_motor(MOTOR_5020, 2.0),
        "4010": _sonic_motor(MOTOR_4010),
    }

    def values(field: int, assignments: tuple[tuple[str, str], ...]) -> dict[str, float]:
        return {pattern: motors[motor][field] for pattern, motor in assignments}

    legs = robot_cfg.actuators["legs"]
    leg_assignments = (
        (".*_hip_pitch_joint", "7520_22"),
        (".*_hip_roll_joint", "7520_22"),
        (".*_hip_yaw_joint", "7520_14"),
        (".*_knee_joint", "7520_22"),
    )
    legs.stiffness = values(0, leg_assignments)
    legs.damping = values(1, leg_assignments)
    legs.armature = values(2, leg_assignments)
    legs.effort_limit = values(3, leg_assignments)
    legs.saturation_effort = MOTOR_7520_22[1]

    feet = robot_cfg.actuators["feet"]
    foot_assignments = ((".*_ankle_pitch_joint", "5020_double"), (".*_ankle_roll_joint", "5020_double"))
    feet.stiffness = values(0, foot_assignments)
    feet.damping = values(1, foot_assignments)
    feet.armature = values(2, foot_assignments)
    feet.effort_limit = {pattern: MOTOR_5020[1] for pattern, _ in foot_assignments}
    feet.saturation_effort = MOTOR_5020[1]

    waist = robot_cfg.actuators["waist"]
    waist_assignments = (("waist_yaw_joint", "7520_14"), ("waist_roll_joint", "5020_double"), ("waist_pitch_joint", "5020_double"))
    waist.stiffness = values(0, waist_assignments)
    waist.damping = values(1, waist_assignments)
    waist.armature = values(2, waist_assignments)
    waist.effort_limit = {
        "waist_yaw_joint": MOTOR_7520_14[1],
        "waist_roll_joint": MOTOR_5020[1],
        "waist_pitch_joint": MOTOR_5020[1],
    }

    arms = robot_cfg.actuators["arms"]
    arm_assignments = (
        (".*_shoulder_pitch_joint", "5020"),
        (".*_shoulder_roll_joint", "5020"),
        (".*_shoulder_yaw_joint", "5020"),
        (".*_elbow_joint", "5020"),
        (".*_wrist_roll_joint", "5020"),
        (".*_wrist_pitch_joint", "4010"),
        (".*_wrist_yaw_joint", "4010"),
    )
    arms.stiffness = values(0, arm_assignments)
    arms.damping = values(1, arm_assignments)
    arms.armature = values(2, arm_assignments)
    arms.effort_limit = values(3, arm_assignments)


class ActuationMode(Enum):
    PASSIVE = "passive"
    CONTROLLED = "controlled"


class ActuationGate:
    """Reversibly disables both explicit actuator models and PhysX joint drives."""

    def __init__(self, robot: object):
        self.robot = robot
        self.mode = ActuationMode.CONTROLLED
        self._joint_stiffness = robot.data.default_joint_stiffness.clone()
        self._joint_damping = robot.data.default_joint_damping.clone()
        self._actuator_gains = {
            name: (actuator.stiffness.clone(), actuator.damping.clone())
            for name, actuator in robot.actuators.items()
        }

    def set_passive(self) -> None:
        self.mode = ActuationMode.PASSIVE
        self.enforce()

    def set_controlled(self) -> None:
        self.mode = ActuationMode.CONTROLLED
        self.enforce()

    def enforce(self) -> None:
        """Reapply the selected mode after Timeline transitions rebuild simulator state."""
        passive = self.mode is ActuationMode.PASSIVE
        for name, actuator in self.robot.actuators.items():
            stiffness, damping = self._actuator_gains[name]
            if passive:
                actuator.stiffness.zero_()
                actuator.damping.zero_()
            else:
                actuator.stiffness.copy_(stiffness)
                actuator.damping.copy_(damping)
        stiffness = self._joint_stiffness.clone()
        damping = self._joint_damping.clone()
        if passive:
            stiffness.zero_()
            damping.zero_()
        self.robot.write_joint_stiffness_to_sim(stiffness)
        self.robot.write_joint_damping_to_sim(damping)
        self.robot.set_joint_effort_target(self.robot.data.default_joint_pos.clone().zero_())
