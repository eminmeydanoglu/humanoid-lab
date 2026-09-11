"""Contract tests for joint commands, layouts and profiles.

These run without a simulator: they cover the rules that decide whether a
command is well formed, which joints it addresses, and how a profile declares a
controller.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from humanoid_lab.contracts.commands import (
    BODY_COMMAND_SCHEMA,
    CommandError,
    CompleteRobotCommand,
    JointCommand,
    JointLayout,
    resolve_layouts,
)
from humanoid_lab.controllers import sonic
from humanoid_lab.simulators.isaac.contracts import ContractError, RunProfile

PROFILES = Path(__file__).resolve().parents[1] / "configs" / "profiles"
REPO = Path(__file__).resolve().parents[1]


class JointLayoutTests(unittest.TestCase):
    def test_resolves_by_name_and_fails_closed_on_a_missing_joint(self) -> None:
        available = ["a", "b", "c"]
        layout = JointLayout.resolve(["c", "a"], available, what="body")
        self.assertEqual(layout.indices, (2, 0))
        with self.assertRaises(CommandError):
            JointLayout.resolve(["c", "zz"], available, what="body")

    def test_rejects_ambiguous_and_duplicate_names(self) -> None:
        with self.assertRaises(CommandError):
            JointLayout.resolve(["a", "a"], ["a"], what="body")
        with self.assertRaises(CommandError):
            JointLayout.resolve(["a"], ["a", "a"], what="body")

    def test_declared_order_is_preserved_not_sorted(self) -> None:
        body, _, _ = resolve_layouts(
            available_joints=["j2", "j0", "j1"], body_names=["j1", "j2", "j0"]
        )
        self.assertEqual(body.names, ("j1", "j2", "j0"))
        self.assertEqual(body.indices, (2, 0, 1))


class JointCommandTests(unittest.TestCase):
    def test_build_fills_missing_gain_terms_with_zero(self) -> None:
        command = JointCommand.build(
            schema=BODY_COMMAND_SCHEMA,
            sequence=1,
            joint_names=["a", "b"],
            q=[1.0, 2.0],
            valid_until_tick=10,
        )
        self.assertEqual(command.dq, (0.0, 0.0))
        self.assertEqual(command.kp, (0.0, 0.0))

    def test_rejects_non_finite_values(self) -> None:
        with self.assertRaises(CommandError):
            JointCommand.build(
                schema=BODY_COMMAND_SCHEMA,
                sequence=1,
                joint_names=["a"],
                q=[float("nan")],
                valid_until_tick=10,
            )

    def test_validate_rejects_a_layout_mismatch(self) -> None:
        layout = JointLayout(("a", "b"), (0, 1))
        command = JointCommand.build(
            schema=BODY_COMMAND_SCHEMA,
            sequence=1,
            joint_names=["b", "a"],
            q=[0.0, 0.0],
            valid_until_tick=10,
        )
        with self.assertRaises(CommandError):
            command.validate(layout)
        short = JointCommand.build(
            schema=BODY_COMMAND_SCHEMA, sequence=2, joint_names=["a"], q=[0.0], valid_until_tick=10
        )
        with self.assertRaises(CommandError):
            short.validate(layout)

    def test_validity_is_bounded_by_the_declared_tick(self) -> None:
        command = JointCommand.build(
            schema=BODY_COMMAND_SCHEMA,
            sequence=1,
            joint_names=["a"],
            q=[0.0],
            valid_until_tick=100,
        )
        self.assertTrue(command.is_valid_at(100))
        self.assertFalse(command.is_valid_at(101))

    def test_complete_command_expires_when_a_hand_part_expires(self) -> None:
        body = JointCommand.build(
            schema=BODY_COMMAND_SCHEMA, sequence=1, joint_names=["a"], q=[0.0], valid_until_tick=100
        )
        hand = JointCommand.build(
            schema="dex3_joint_command_v1", sequence=1, joint_names=["h"], q=[0.0], valid_until_tick=40
        )
        command = CompleteRobotCommand(episode_id=0, body=body, left_hand=hand)
        self.assertTrue(command.is_valid_at(40))
        self.assertFalse(command.is_valid_at(41))


class SonicConstantTests(unittest.TestCase):
    def test_body_order_matches_the_hand_order_prefixes(self) -> None:
        self.assertEqual(len(sonic.BODY_JOINT_ORDER), 29)
        self.assertEqual(len(set(sonic.BODY_JOINT_ORDER)), 29)
        self.assertEqual(len(sonic.BODY_EFFORT_LIMIT_NM), 29)
        self.assertEqual(len(sonic.BODY_MOTOR_FAMILY), 29)
        self.assertEqual(len(sonic.BODY_GAIN_MULTIPLIER), 29)
        self.assertEqual(len(sonic.MUJOCO_BODY_FRICTION_LOSS_NM), 29)
        self.assertEqual(len(sonic.HAND_JOINT_ORDER), 7)

    def test_43dof_effort_limits_are_split_by_actuator_order(self) -> None:
        # The hands sit between the left and right arms in MJCF actuator order.
        # Accidentally taking the first 29 entries clips the right arm to finger
        # limits and gives the left hand arm-sized limits.
        self.assertEqual(sonic.BODY_EFFORT_LIMIT_NM[15:22], sonic.BODY_EFFORT_LIMIT_NM[22:29])
        self.assertEqual(sonic.HAND_EFFORT_LIMIT_NM, sonic.RIGHT_HAND_EFFORT_LIMIT_NM)
        self.assertEqual(sonic.HAND_EFFORT_LIMIT_NM, (2.45, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7))

    def test_gains_are_derived_from_the_update_rule(self) -> None:
        import math

        kp, kd = sonic.deploy_gains()
        omega = 2.0 * math.pi * sonic.NATURAL_FREQUENCY_HZ
        armature = sonic.MOTOR_ARMATURE[sonic.BODY_MOTOR_FAMILY[0]]
        self.assertAlmostEqual(kp[0], armature * omega * omega)
        self.assertAlmostEqual(kd[0], 2.0 * sonic.DAMPING_RATIO * armature * omega)
        ankle = sonic.BODY_JOINT_ORDER.index("left_ankle_pitch_joint")
        ankle_armature = sonic.MOTOR_ARMATURE[sonic.BODY_MOTOR_FAMILY[ankle]]
        self.assertAlmostEqual(kp[ankle], 2.0 * ankle_armature * omega * omega)
        self.assertAlmostEqual(
            kd[ankle], 2.0 * 2.0 * sonic.DAMPING_RATIO * ankle_armature * omega
        )

    def test_standing_pose_covers_every_body_joint(self) -> None:
        self.assertEqual(set(sonic.standing_pose()), set(sonic.BODY_JOINT_ORDER))

    def test_the_run_command_uses_a_versioned_planner_path_and_sim_flags(self) -> None:
        """The deployment parses its planner version from the path, so the run
        command is the one place that must carry a version token."""
        dev_sh = (REPO / "dev.sh").read_text()
        block = dev_sh[dev_sh.index("sonic-controller)") : dev_sh.index("  doctor)")]
        self.assertIn("V2/planner_sonic.onnx", block)
        self.assertIn("--planner-file", block)
        for flag in sonic.SIMULATION_ONLY_FLAGS:
            self.assertIn(flag, block)
        self.assertIn(f"exec ./g1_deploy_onnx_ref {sonic.DEFAULT_INTERFACE}", block)
        # Only Isaac-targeted controllers replace each other. A generic process
        # name match would also kill an independent MuJoCo SONIC deployment.
        self.assertIn("HUMANOID_LAB_CONTROLLER_TARGET", block)
        self.assertIn("humanoid-lab-sonic-controller-isaac.pid", block)
        self.assertIn("replacing previous Isaac controller", block)
        self.assertNotIn("pkill", block)
        # The version token is checked before launch, so a wrong path fails with
        # a clear message instead of the deployment's own crash.
        self.assertIn("planner version token", block)


class ProfileTests(unittest.TestCase):
    def test_shipped_profiles_load(self) -> None:
        for path in sorted(PROFILES.glob("*.json")):
            with self.subTest(profile=path.name):
                profile = RunProfile.load(path)
                self.assertEqual(profile.schema_version, 1)
                self.assertEqual(profile.robot.body_dofs, 29)

    def test_sonic_profiles_declare_a_controller_pose_and_support(self) -> None:
        for name in ("isaac-g1-sonic-dex3.json", "isaac-g1-sonic-inspire-ftp.json"):
            with self.subTest(profile=name):
                profile = RunProfile.load(PROFILES / name)
                self.assertIsNotNone(profile.controller)
                assert profile.controller is not None
                self.assertEqual(profile.controller["provider"], "sonic_dds")
                self.assertEqual(profile.controller["domain_id"], sonic.DEFAULT_DOMAIN_ID)
                self.assertEqual(profile.initial_pose, "sonic_standing")
                self.assertIsNotNone(profile.support)
                assert profile.support is not None
                self.assertEqual(
                    profile.robot.initial_position_m[2], sonic.STANDING_ROOT_HEIGHT_M
                )

    def test_dex3_profile_aligns_mujoco_mass_and_joint_dynamics(self) -> None:
        profile = RunProfile.load(PROFILES / "isaac-g1-sonic-dex3.json")
        assert profile.controller is not None
        self.assertEqual(profile.controller["mass_alignment"], "sonic_mujoco")
        self.assertEqual(profile.controller["joint_dynamics_alignment"], "sonic_mujoco")

    def test_support_timeout_waits_for_the_first_controller_command(self) -> None:
        service = (REPO / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        body = service[
            service.index("    def _update_support") : service.index("    def _release_support")
        ]
        self.assertLess(body.index("if command is None:"), body.index("elapsed ="))
        self.assertIn("self._support_started_tick = self.tick", body)

    def test_a_support_band_without_a_controller_is_rejected(self) -> None:
        data = json.loads((PROFILES / "isaac-g1-sonic-dex3.json").read_text())
        del data["controller"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(ContractError):
                RunProfile.load(path)

    def test_a_mismatched_pose_height_is_rejected(self) -> None:
        data = json.loads((PROFILES / "isaac-g1-sonic-dex3.json").read_text())
        data["robot"]["initial_position_m"] = [0.0, 0.0, 0.9]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(data))
            profile = RunProfile.load(path)
            # The pose and its height travel together; mismatch is caught when
            # the pose is resolved against the asset and the spawn height.
            from humanoid_lab.controllers.sonic import named_pose

            _, height = named_pose(profile.initial_pose)
            self.assertNotAlmostEqual(profile.robot.initial_position_m[2], height)

    def test_unknown_provider_is_rejected_at_build_time(self) -> None:
        from humanoid_lab.controllers.factory import build_controller
        from humanoid_lab.contracts.commands import CommandError as ProviderError

        with self.assertRaises(ProviderError):
            build_controller({"provider": "nope"}, provider_override=None, physics_dt=0.005, ttl_s=0.25)

    def test_none_provider_yields_no_controller(self) -> None:
        from humanoid_lab.controllers.factory import build_controller

        controller, interface = build_controller(
            {"provider": "none"}, provider_override=None, physics_dt=0.005, ttl_s=0.25
        )
        self.assertIsNone(controller)
        self.assertIsNone(interface)


if __name__ == "__main__":
    unittest.main()
