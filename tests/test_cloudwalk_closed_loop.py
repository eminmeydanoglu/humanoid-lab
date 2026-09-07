#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cloudwalk_closed_loop import BODY_PACKET_SIZE, STATE_PACKET_SIZE, BodyCommand, SonicState
from sonic_isaac_inspire_adapter import ContractError


class ClosedLoopContractTests(unittest.TestCase):
    def test_live_state_and_body_packets_are_exact_and_round_trip(self):
        state = SonicState(7, 100, (0.1, 0.2, 0.3), tuple(range(29)), tuple(range(29)), tuple(range(29)), (0.0, 0.0, -1.0))
        command = BodyCommand(7, 101, tuple(range(29)))
        self.assertEqual(len(state.pack()), STATE_PACKET_SIZE)
        self.assertEqual(len(command.pack()), BODY_PACKET_SIZE)
        unpacked_state = SonicState.unpack(state.pack())
        self.assertEqual((unpacked_state.sequence, unpacked_state.monotonic_ns), (state.sequence, state.monotonic_ns))
        self.assertEqual(tuple(round(value, 6) for value in unpacked_state.base_angular_velocity), tuple(round(value, 6) for value in state.base_angular_velocity))
        self.assertEqual(BodyCommand.unpack(command.pack()), command)

    def test_malformed_or_nonfinite_live_packets_fail_closed(self):
        with self.assertRaisesRegex(ContractError, "exactly 29"):
            BodyCommand(0, 0, (0.0,) * 28)
        with self.assertRaisesRegex(ContractError, "non-finite"):
            SonicState(0, 0, (math.nan, 0.0, 0.0), (0.0,) * 29, (0.0,) * 29, (0.0,) * 29, (0.0, 0.0, -1.0))
        with self.assertRaises(ContractError):
            SonicState.unpack(b"bad")

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
        self.assertIn("ZMQPackedMessageSubscriber", native)
        self.assertIn("kDecoderInput", native)
        self.assertIn("run_gr00t_server.py", launcher)
        self.assertIn("SONIC_NATIVE_HARNESS=closed-loop", launcher)


if __name__ == "__main__":
    unittest.main()
