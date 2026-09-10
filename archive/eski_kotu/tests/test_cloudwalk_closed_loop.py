#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cloudwalk_closed_loop import BODY_PACKET_SIZE, DEFAULT_ANGLES, ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB, STATE_PACKET_SIZE, BodyCommand, SonicState, native_decoder_state
from sonic_isaac_inspire_adapter import ContractError


class ClosedLoopContractTests(unittest.TestCase):
    def test_live_state_and_body_packets_are_exact_and_round_trip(self):
        state = SonicState(7, 100, (0.1, 0.2, 0.3), tuple(range(29)), tuple(range(29)), tuple(range(29)), (0.0, 0.0, -1.0))
        command = BodyCommand(7, 101, tuple(range(29)), tuple(value / 10 for value in range(29)))
        self.assertEqual(len(state.pack()), STATE_PACKET_SIZE)
        self.assertEqual(len(command.pack()), BODY_PACKET_SIZE)
        unpacked_state = SonicState.unpack(state.pack())
        self.assertEqual((unpacked_state.sequence, unpacked_state.monotonic_ns), (state.sequence, state.monotonic_ns))
        self.assertEqual(tuple(round(value, 6) for value in unpacked_state.base_angular_velocity), tuple(round(value, 6) for value in state.base_angular_velocity))
        unpacked_command = BodyCommand.unpack(command.pack())
        self.assertEqual((unpacked_command.sequence, unpacked_command.monotonic_ns), (command.sequence, command.monotonic_ns))
        self.assertEqual(tuple(round(value, 6) for value in unpacked_command.positions), tuple(round(value, 6) for value in command.positions))
        self.assertEqual(tuple(round(value, 6) for value in unpacked_command.raw_actions), tuple(round(value, 6) for value in command.raw_actions))

    def test_malformed_or_nonfinite_live_packets_fail_closed(self):
        with self.assertRaisesRegex(ContractError, "exactly 29"):
            BodyCommand(0, 0, (0.0,) * 28, (0.0,) * 29)
        with self.assertRaisesRegex(ContractError, "non-finite"):
            SonicState(0, 0, (math.nan, 0.0, 0.0), (0.0,) * 29, (0.0,) * 29, (0.0,) * 29, (0.0, 0.0, -1.0))
        with self.assertRaises(ContractError):
            SonicState.unpack(b"bad")

    def test_native_decoder_state_matches_upstream_order_and_default_pose(self):
        q = tuple(DEFAULT_ANGLES[index] + index / 100 for index in range(29))
        qd = tuple(index / 10 for index in range(29))
        relative, velocity = native_decoder_state(q, qd)
        self.assertEqual(tuple(round(value, 10) for value in relative), tuple(round(MUJOCO_TO_ISAACLAB[index] / 100, 10) for index in range(29)))
        self.assertEqual(velocity, tuple(qd[index] for index in MUJOCO_TO_ISAACLAB))
        self.assertEqual(tuple(MUJOCO_TO_ISAACLAB[index] for index in ISAACLAB_TO_MUJOCO), tuple(range(29)))

    def test_constants_match_pinned_upstream_policy_parameters_when_available(self):
        root = Path(os.environ.get("SONIC_ROOT", "/opt/src/sonic"))
        source = root / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
        if not source.is_file():
            self.skipTest("pinned SONIC source is available on Raider")
        text = source.read_text()
        def array(name: str) -> tuple[float, ...]:
            match = re.search(rf"{name}[^=]*=\s*\{{([^}}]+)\}}", text, re.S)
            self.assertIsNotNone(match)
            return tuple(float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", re.sub(r"//.*", "", match.group(1))))
        self.assertEqual(tuple(int(value) for value in array("isaaclab_to_mujoco")), ISAACLAB_TO_MUJOCO)
        self.assertEqual(tuple(int(value) for value in array("mujoco_to_isaaclab")), MUJOCO_TO_ISAACLAB)
        self.assertEqual(array("default_angles"), DEFAULT_ANGLES)

    def test_decoder_observation_order_matches_pinned_model_config_when_available(self):
        path = Path(os.environ.get("SONIC_MODELS_ROOT", "/data/runtime/sonic-deploy-models/sonic_v1_1")) / "observation_config.yaml"
        if not path.is_file():
            self.skipTest("pinned SONIC model config is available on Raider")
        names = re.findall(r'^  - name: "([^"]+)"', path.read_text(), re.M)
        self.assertEqual(names[:6], ["token_state", "his_base_angular_velocity_10frame_step1", "his_body_joint_positions_10frame_step1", "his_body_joint_velocities_10frame_step1", "his_last_actions_10frame_step1", "his_gravity_dir_10frame_step1"])

    def test_shipped_path_uses_real_boundaries_and_applies_verified_hands(self):
        root = Path(__file__).resolve().parents[1]
        runner = (root / "scripts" / "run-cloudwalk-isaac.py").read_text()
        worker = (root / "scripts" / "cloudwalk-vla-worker.py").read_text()
        native = (root / "tests/native/sonic_closed_loop_harness.cpp").read_text()
        launcher = (root / "scripts" / "run-cloudwalk-closed-loop.sh").read_text()
        self.assertIn("PolicyClient", worker)
        self.assertIn("prepare_observation_for_eval", worker)
        self.assertIn("load_upstream_packer", runner)
        self.assertIn("G1_BODY_JOINTS", runner)
        self.assertIn("verified_24_joint_normalized_mapper", runner)
        self.assertIn("InspireFTPGripMapper", runner)
        self.assertIn("--rollout-metrics-path", runner)
        self.assertIn("hand_object_distance_m", runner)
        self.assertIn("stable_grasp_frames", runner)
        self.assertIn("max_grasp_associated_lift_m", runner)
        self.assertIn("bottle_fell", runner)
        self.assertIn("--video-path", launcher)
        self.assertIn("ZMQPackedMessageSubscriber", native)
        self.assertIn("kDecoderInput", native)
        self.assertIn("run_gr00t_server.py", launcher)
        self.assertIn("SONIC_NATIVE_HARNESS=closed-loop", launcher)


if __name__ == "__main__":
    unittest.main()
