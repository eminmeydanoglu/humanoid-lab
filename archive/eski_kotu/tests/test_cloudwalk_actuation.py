import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cloudwalk_actuation import (
    MOTOR_4010,
    MOTOR_5020,
    MOTOR_7520_14,
    MOTOR_7520_22,
    ActuationGate,
    ActuationMode,
    configure_sonic_actuators,
)


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def clone(self):
        return FakeTensor(self.values)

    def zero_(self):
        self.values = [0.0 for _ in self.values]
        return self

    def copy_(self, other):
        self.values = list(other.values)
        return self


class FakeRobot:
    def __init__(self):
        self.data = SimpleNamespace(
            default_joint_stiffness=FakeTensor([100.0, 20.0]),
            default_joint_damping=FakeTensor([5.0, 1.0]),
            default_joint_pos=FakeTensor([0.0, 0.0]),
        )
        self.actuators = {
            "explicit_leg": SimpleNamespace(stiffness=FakeTensor([100.0]), damping=FakeTensor([5.0])),
            "implicit_arm": SimpleNamespace(stiffness=FakeTensor([20.0]), damping=FakeTensor([1.0])),
        }
        self.sim_stiffness = None
        self.sim_damping = None
        self.effort = None

    def write_joint_stiffness_to_sim(self, value):
        self.sim_stiffness = value.clone()

    def write_joint_damping_to_sim(self, value):
        self.sim_damping = value.clone()

    def set_joint_effort_target(self, value):
        self.effort = value.clone()


class SonicActuatorConfigTests(unittest.TestCase):
    def test_sonic_motor_model_replaces_body_gains_and_preserves_hands(self):
        def actuator():
            return SimpleNamespace(stiffness=None, damping=None, armature=None, effort_limit=None, saturation_effort=None)

        hands = actuator()
        cfg = SimpleNamespace(actuators={"legs": actuator(), "feet": actuator(), "waist": actuator(), "arms": actuator(), "hands": hands})

        configure_sonic_actuators(cfg)

        self.assertIs(cfg.actuators["hands"], hands)
        self.assertEqual(cfg.actuators["legs"].armature[".*_hip_pitch_joint"], MOTOR_7520_22[0])
        self.assertEqual(cfg.actuators["legs"].effort_limit[".*_hip_yaw_joint"], MOTOR_7520_14[1])
        self.assertEqual(cfg.actuators["legs"].saturation_effort, MOTOR_7520_22[1])
        self.assertEqual(cfg.actuators["feet"].armature[".*_ankle_roll_joint"], MOTOR_5020[0])
        self.assertEqual(cfg.actuators["feet"].saturation_effort, MOTOR_5020[1])
        self.assertEqual(cfg.actuators["waist"].effort_limit["waist_roll_joint"], MOTOR_5020[1])
        self.assertEqual(cfg.actuators["arms"].armature[".*_wrist_pitch_joint"], MOTOR_4010[0])
        self.assertLess(cfg.actuators["arms"].stiffness[".*_shoulder_pitch_joint"], 20.0)
        self.assertLess(cfg.actuators["arms"].damping[".*_shoulder_pitch_joint"], 2.0)


class ActuationGateTests(unittest.TestCase):
    def test_passive_disables_explicit_models_and_physx_drives(self):
        robot = FakeRobot()
        gate = ActuationGate(robot)

        gate.set_passive()

        self.assertIs(gate.mode, ActuationMode.PASSIVE)
        self.assertEqual(robot.actuators["explicit_leg"].stiffness.values, [0.0])
        self.assertEqual(robot.actuators["implicit_arm"].damping.values, [0.0])
        self.assertEqual(robot.sim_stiffness.values, [0.0, 0.0])
        self.assertEqual(robot.sim_damping.values, [0.0, 0.0])
        self.assertEqual(robot.effort.values, [0.0, 0.0])

    def test_passive_enforcement_clears_gains_restored_by_timeline_transition(self):
        robot = FakeRobot()
        gate = ActuationGate(robot)
        gate.set_passive()
        robot.actuators["explicit_leg"].stiffness.values = [100.0]
        robot.actuators["implicit_arm"].damping.values = [1.0]

        gate.enforce()

        self.assertEqual(robot.actuators["explicit_leg"].stiffness.values, [0.0])
        self.assertEqual(robot.actuators["implicit_arm"].damping.values, [0.0])
        self.assertEqual(robot.sim_stiffness.values, [0.0, 0.0])
        self.assertEqual(robot.sim_damping.values, [0.0, 0.0])

    def test_controlled_restores_all_actuator_gains(self):
        robot = FakeRobot()
        gate = ActuationGate(robot)
        gate.set_passive()

        gate.set_controlled()

        self.assertIs(gate.mode, ActuationMode.CONTROLLED)
        self.assertEqual(robot.actuators["explicit_leg"].stiffness.values, [100.0])
        self.assertEqual(robot.actuators["explicit_leg"].damping.values, [5.0])
        self.assertEqual(robot.actuators["implicit_arm"].stiffness.values, [20.0])
        self.assertEqual(robot.sim_stiffness.values, [100.0, 20.0])
        self.assertEqual(robot.sim_damping.values, [5.0, 1.0])


if __name__ == "__main__":
    unittest.main()
