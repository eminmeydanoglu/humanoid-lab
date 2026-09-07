#!/usr/bin/env python3
"""Unit tests for the F310-to-Unitree packet mapping; no hardware required."""

import importlib.util
import pathlib
import struct
import sys
import unittest

MODULE = pathlib.Path(__file__).parents[1] / "tools" / "f310_unitree_bridge.py"
SPEC = importlib.util.spec_from_file_location("f310_bridge", MODULE)
bridge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


class F310MappingTests(unittest.TestCase):
    def test_face_and_system_buttons_follow_unitree_bit_order(self):
        state = bridge.F310State(buttons={0: 1, 1: 1, 6: 1, 7: 1, 8: 1})
        bits = struct.unpack_from("<H", state.remote_payload(), 2)[0]
        for bit in (2, 3, 6, 8, 9):
            self.assertTrue(bits & (1 << bit), bit)

    def test_bumper_chord_is_f1_without_l1_or_r1_leakage(self):
        state = bridge.F310State(buttons={4: 1, 5: 1})
        bits = struct.unpack_from("<H", state.remote_payload(now=10.0), 2)[0]
        self.assertTrue(bits & (1 << 6))
        self.assertFalse(bits & (1 << 0))
        self.assertFalse(bits & (1 << 1))

        state.buttons = {}
        self.assertEqual(struct.unpack_from("<H", state.remote_payload(now=10.5), 2)[0] & 0x43, 0)
        state.buttons = {4: 1}
        self.assertEqual(struct.unpack_from("<H", state.remote_payload(now=11.0), 2)[0] & 0x03, 0)
        self.assertTrue(struct.unpack_from("<H", state.remote_payload(now=11.2), 2)[0] & (1 << 1))

        state.buttons = {4: 1, 5: 1}
        self.assertTrue(struct.unpack_from("<H", state.remote_payload(now=11.21), 2)[0] & (1 << 6))
        state.buttons = {5: 1}
        self.assertEqual(struct.unpack_from("<H", state.remote_payload(now=11.4), 2)[0] & 0x43, 0)

    def test_summary_does_not_consume_the_f1_chord_before_transmission(self):
        state = bridge.F310State(buttons={4: 1, 5: 1})
        self.assertIn("buttons=0x0040", state.summary())
        bits = struct.unpack_from("<H", state.remote_payload(now=10.0), 2)[0]
        self.assertTrue(bits & (1 << 6))

    def test_axes_and_triggers_are_normalized_with_forward_positive(self):
        state = bridge.F310State(axes={0: 16384, 1: -16384, 2: 32767, 3: -16384, 4: 16384, 5: 32767})
        raw = state.remote_payload()
        lx, rx, ry, l2, ly = struct.unpack_from("<5f", raw, 4)
        self.assertAlmostEqual(lx, 0.5, places=2)
        self.assertAlmostEqual(ly, 0.5, places=2)
        self.assertAlmostEqual(rx, -0.5, places=2)
        self.assertAlmostEqual(ry, -0.5, places=2)
        self.assertAlmostEqual(l2, 1.0, places=3)

    def test_dpad_maps_from_hat_axes(self):
        state = bridge.F310State(axes={6: -32767, 7: 32767})
        bits = struct.unpack_from("<H", state.remote_payload(), 2)[0]
        self.assertTrue(bits & (1 << 15))
        self.assertTrue(bits & (1 << 14))

    def test_wire_frame_is_fixed_and_decodable(self):
        raw = bridge.F310State().remote_payload(now=0.0)
        frame = bridge.FRAME.pack(bridge.MAGIC, bridge.VERSION, 0, 0, 42, raw)
        self.assertEqual(len(frame), 56)
        magic, version, flags, reserved, sequence, decoded = bridge.FRAME.unpack(frame)
        self.assertEqual((magic, version, flags, reserved, sequence, decoded), (bridge.MAGIC, bridge.VERSION, 0, 0, 42, raw))


if __name__ == "__main__":
    unittest.main()
