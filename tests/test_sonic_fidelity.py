"""The primary A/B/C replay gate must expose missing tails and wrong commands."""

from __future__ import annotations

import unittest
import runpy
import tempfile
from pathlib import Path

import numpy as np

from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER
from humanoid_lab.datasets.sonic.fidelity import compare_reference_command_response, simulation_frame_due
from humanoid_lab.datasets.sonic.joints import reorder


class SonicFidelityTest(unittest.TestCase):
    def test_receiver_log_gate_detects_a_missing_middle_frame(self):
        script = Path(__file__).resolve().parents[1] / "scripts/check-sonic-token-delivery.py"
        check = runpy.run_path(str(script))["receiver_coverage"]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "controller.log"
            log.write_text("\n".join(
                f"[ZMQEndpointInterface] Protocol v4: Received 64D token, frame_index: {index}"
                for index in (0, 1, 3, 3)
            ), encoding="utf-8")
            result = check(log, 4)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_frames"], [2])

    def test_receiver_gate_accepts_upstream_interleaved_log_line(self):
        script = Path(__file__).resolve().parents[1] / "scripts/check-sonic-token-delivery.py"
        check = runpy.run_path(str(script))["receiver_coverage"]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "controller.log"
            log.write_text(
                "Protocol v4: Received 64D token, frame_index: 0\n"
                "Protocol v4: Received Reset init reference data root rotation\n"
                "0.1250, frame_index: 1\n",
                encoding="utf-8",
            )
            result = check(log, 2)
        self.assertTrue(result["complete"])

    def test_quantized_simulation_clock_does_not_skip_a_frame(self):
        # Binary floating point represents 26.1 below 25.94 + 8/50.
        self.assertTrue(simulation_frame_due(26.1, 25.94, 8))
        self.assertFalse(simulation_frame_due(26.08, 25.94, 8))

    def fixture(self, *, final_frame: int = 3, wrong_left_arm: bool = False):
        body = np.zeros((4, 29), dtype=np.float32)
        body[:, BODY_JOINT_ORDER.index("left_shoulder_roll_joint")] = [0.0, 0.2, 0.4, 0.6]
        body[:, BODY_JOINT_ORDER.index("right_shoulder_roll_joint")] = -0.4
        reference = {
            "joint_pos": reorder(body, BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER),
            "left_hand_joints": np.zeros((4, 7), dtype=np.float32),
            "right_hand_joints": np.zeros((4, 7), dtype=np.float32),
        }
        command = body[: final_frame + 1].copy()
        if wrong_left_arm:
            command[:, BODY_JOINT_ORDER.index("left_shoulder_roll_joint")] += 1.0
        tracking = {
            "sim_s": np.arange(final_frame + 1) * 0.02,
            "wall_time_ns": np.arange(final_frame + 1, dtype=np.int64) * 20_000_000 + 5_000_000,
            "support_active": np.zeros(final_frame + 1, dtype=bool),
            "body_target": command,
            "body_measured": command.copy(),
            "left_hand_target": np.zeros((final_frame + 1, 7)),
            "left_hand_measured": np.zeros((final_frame + 1, 7)),
            "right_hand_target": np.zeros((final_frame + 1, 7)),
            "right_hand_measured": np.zeros((final_frame + 1, 7)),
        }
        timeline = {
            "frame_index": np.arange(4),
            "sent_wall_time_ns": np.arange(4, dtype=np.int64) * 20_000_000,
        }
        return reference, tracking, timeline

    def test_complete_matching_motion_passes(self):
        result = compare_reference_command_response(*self.fixture())
        self.assertEqual(result["result"], "PASS")
        self.assertTrue(result["coverage"]["full_motion"])
        self.assertEqual(result["pairs"]["reference_to_command"]["body_mae_rad"], 0.0)

    def test_decoder_q_target_is_diagnostic_not_a_latent_fidelity_gate(self):
        reference, tracking, timeline = self.fixture(wrong_left_arm=True)
        tracking["body_measured"] = reorder(reference["joint_pos"], SONIC_REFERENCE_JOINT_ORDER, BODY_JOINT_ORDER)
        result = compare_reference_command_response(reference, tracking, timeline)
        self.assertGreater(result["pairs"]["command_to_response"]["body_mae_rad"], 0.0)
        self.assertEqual(result["result"], "PASS")
        self.assertLess(result["pairs"]["reference_to_command"]["body_mae_rad"], 0.35)
        self.assertGreater(result["pairs"]["reference_to_command"]["body_worst_joint_mae_rad"], 0.50)
        self.assertGreater(result["pairs"]["reference_to_command"]["left_arm_worst_joint_mae_rad"], 0.45)

    def test_mid_motion_cutoff_fails_even_with_zero_error(self):
        reference, tracking, timeline = self.fixture(final_frame=1)
        result = compare_reference_command_response(reference, tracking, timeline)
        self.assertEqual(result["pairs"]["reference_to_command"]["body_mae_rad"], 0.0)
        self.assertFalse(result["coverage"]["full_motion"])
        self.assertEqual(result["result"], "FAIL")

    def test_robot_response_is_scored_separately_from_sonic_command(self):
        reference, tracking, timeline = self.fixture()
        joint = BODY_JOINT_ORDER.index("left_shoulder_roll_joint")
        tracking["body_measured"][:, joint] += 1.0
        result = compare_reference_command_response(reference, tracking, timeline)
        self.assertEqual(result["pairs"]["reference_to_command"]["body_mae_rad"], 0.0)
        self.assertEqual(result["result"], "FAIL")
        self.assertGreater(result["pairs"]["reference_to_response"]["left_arm_worst_joint_mae_rad"], 0.55)

    def test_simulation_clock_is_used_when_available(self):
        reference, tracking, timeline = self.fixture()
        # Wall timestamps are deliberately shifted away from the send times;
        # the shared simulation clock still aligns all four frames correctly.
        tracking["wall_time_ns"] += 2_000_000_000
        timeline["sent_sim_s"] = np.arange(4, dtype=np.float64) * 0.02
        result = compare_reference_command_response(reference, tracking, timeline)
        self.assertEqual(result["result"], "PASS")
        self.assertEqual(result["coverage"]["alignment"], "latest publisher send by Isaac simulation clock")

    def test_full_sonic_command_components_are_reported(self):
        reference, tracking, timeline = self.fixture()
        shape = tracking["body_target"].shape
        for name in ("body_velocity_target", "body_feedforward_torque", "body_kp", "body_kd",
                     "body_applied_torque", "body_measured_velocity"):
            tracking[name] = np.zeros(shape)
        result = compare_reference_command_response(reference, tracking, timeline)
        self.assertEqual(result["sonic_action_components"]["left_arm_mean_abs_torque_reconstruction_error_nm"], 0.0)


if __name__ == "__main__":
    unittest.main()
