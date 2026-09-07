#!/usr/bin/env python3
from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sonic_isaac_inspire_adapter import *  # noqa: F403

def action(seed: float = 0.25) -> V4Action: return V4Action(tuple(seed + i for i in range(64)), tuple(seed + i for i in range(7)), tuple(seed + 10 + i for i in range(7)))
def reference() -> SafeReference: return SafeReference(action(), SONIC_V1_1_DECODER_SHA256, "recorded-upstream-sonic-initial-poses.py@a0732b642c0333077e127a2f56ab0014c196bca4")

class AdapterTests(unittest.TestCase):
    def test_ragged_nonnumeric_and_nonfinite_chunks_fail_cleanly(self):
        valid = [[0.1] * 78 for _ in range(40)]
        for bad in (valid[:-1], valid[:1] + [[0.1] * 77] + valid[2:], valid[:1] + ["not-a-row"] + valid[2:]):
            with self.assertRaisesRegex(ContractError, r"\[40,78\]"): split_groot_action_chunk(bad)
        valid[1][2] = float("nan")
        with self.assertRaisesRegex(ContractError, "non-finite"): split_groot_action_chunk(valid)

    def test_v4_packet_is_schema_exact(self):
        packet = pack_protocol_v4(action(), 17); decoded, frame = unpack_protocol_v4(packet)
        self.assertEqual((frame, decoded), (17, action()))
        self.assertEqual(len(packet), 4 + HEADER_SIZE + 320)
        with self.assertRaises(ContractError): unpack_protocol_v4(packet[:-1])

    def test_verified_thumb_first_mapping_scales_normalized_closures_to_joint_limits(self):
        closures = V4Action((0.0,) * 64, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), (0.7, 0.8, 0.9, 1.0, 0.0, 0.1, 0.2))
        mapper = InspireFTPGripMapper()
        with self.assertRaisesRegex(ContractError, "verified channel order and right-thumb closure sign"):
            mapper.normalized_targets(closures)
        mapper = InspireFTPGripMapper(True, True)
        limits = tuple((-1.0, 1.0) for _ in INSPIRE_HAND_JOINTS)
        mapped = dict(zip(INSPIRE_HAND_JOINTS, mapper.targets(closures, limits)))
        self.assertEqual(mapped["L_thumb_proximal_yaw_joint"], -1.0)
        self.assertEqual(mapped["L_index_proximal_joint"], -0.4)
        self.assertAlmostEqual(mapped["L_ring_intermediate_joint"], 0.2)
        self.assertEqual(mapped["R_thumb_distal_joint"], 0.8)
        self.assertEqual(len(mapped), 24)

    def test_hand_mapping_rejects_non_normalized_values_and_incomplete_limits(self):
        mapper = InspireFTPGripMapper(True, True)
        with self.assertRaisesRegex(ContractError, "normalized"):
            mapper.normalized_targets(action())
        with self.assertRaisesRegex(ContractError, "exactly 24"):
            mapper.targets(V4Action((0.0,) * 64, (0.0,) * 7, (0.0,) * 7), ((0.0, 1.0),))

    def test_transition_and_validation_order_are_fail_closed(self):
        guard = LifecycleGuard(0.1)
        with self.assertRaises(ContractError): guard.start(0)
        guard.initialize(reference()); guard.start(1.0)
        endpoint = SimulatorV4Endpoint(guard, InspireFTPGripMapper())
        with self.assertRaises(ContractError): endpoint.submit(action(), 1.01)
        self.assertEqual(guard.last_action_time, 1.0)
        self.assertEqual(guard.tick(1.11), Lifecycle.TIMED_OUT)
        with self.assertRaises(ContractError): guard.pause()
        guard.stop(); guard.reset(); self.assertEqual(guard.lifecycle, Lifecycle.RESET)

    def test_scheduler_drops_stale_backlog(self):
        schedule = Scheduler(); schedule.install_chunk(10.0); self.assertTrue(schedule.action_due(10.0))
        schedule.consume_action(12.0); self.assertFalse(schedule.action_due(12.0)); self.assertTrue(schedule.action_due(12.02))
        self.assertFalse(schedule.inference_due(10.399)); self.assertTrue(schedule.inference_due(10.4)); schedule.mark_inference(10.4); self.assertFalse(schedule.inference_due(10.5))

    def test_manifest_errors_are_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"; path.write_text("{")
            with self.assertRaisesRegex(ContractError, "malformed safe-reference manifest"): SafeReference.load(path)
            path.write_text(json.dumps({"motion_token": []}))
            with self.assertRaisesRegex(ContractError, "malformed safe-reference manifest"): SafeReference.load(path)

if __name__ == "__main__": unittest.main()
