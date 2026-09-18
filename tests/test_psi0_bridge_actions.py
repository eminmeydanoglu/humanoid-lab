"""Action adapter: 78/80D Ψ₀ action -> SONIC Protocol v4 ``pose`` message."""

from __future__ import annotations

import json
import unittest

import numpy as np

from humanoid_lab.datasets.sonic.protocol_v4 import HEADER_SIZE
from humanoid_lab.psi0_bridge.actions import ActionAdapter, ActionAdapterError, fsq_quantize

TOKEN_BYTES = 64 * 4
HAND_BYTES = 7 * 4


def parse_packed(packed: bytes) -> tuple[dict, bytes]:
    assert packed.startswith(b"pose"), packed[:8]
    header = json.loads(packed[4:4 + HEADER_SIZE].rstrip(b"\0"))
    return header, packed[4 + HEADER_SIZE:]


def field_map(header: dict) -> dict[str, dict]:
    return {field["name"]: field for field in header["fields"]}


def frame_index(packed: bytes) -> int:
    _, payload = parse_packed(packed)
    return int(np.frombuffer(payload[TOKEN_BYTES:TOKEN_BYTES + 8], dtype="<i8")[0])


class ProtocolPackingTest(unittest.TestCase):
    def test_78d_splits_into_token_and_hands_with_exact_wire_shape(self) -> None:
        action = np.arange(78, dtype=np.float32) / 100.0
        header, payload = parse_packed(ActionAdapter().pack(action))

        self.assertEqual(header["v"], 4)
        self.assertEqual(
            [field["name"] for field in header["fields"]],
            ["token_state", "frame_index", "left_hand_joints", "right_hand_joints"],
        )
        fields = field_map(header)
        self.assertEqual(fields["token_state"], {"name": "token_state", "dtype": "f32", "shape": [1, 64]})
        self.assertEqual(fields["frame_index"], {"name": "frame_index", "dtype": "i64", "shape": [1]})
        self.assertEqual(fields["left_hand_joints"], {"name": "left_hand_joints", "dtype": "f32", "shape": [1, 7]})
        self.assertEqual(fields["right_hand_joints"], {"name": "right_hand_joints", "dtype": "f32", "shape": [1, 7]})

        token = np.frombuffer(payload[:TOKEN_BYTES], dtype="<f4")
        np.testing.assert_array_equal(token, fsq_quantize(action[:64]))
        self.assertEqual(int(np.frombuffer(payload[TOKEN_BYTES:TOKEN_BYTES + 8], dtype="<i8")[0]), 0)
        left = np.frombuffer(payload[TOKEN_BYTES + 8:TOKEN_BYTES + 8 + HAND_BYTES], dtype="<f4")
        right = np.frombuffer(payload[TOKEN_BYTES + 8 + HAND_BYTES:], dtype="<f4")
        np.testing.assert_array_equal(left, action[64:71])
        np.testing.assert_array_equal(right, action[71:78])

    def test_hands_are_not_swapped(self) -> None:
        action = np.zeros(78, dtype=np.float32)
        action[64:71] = np.arange(1, 8)
        action[71:78] = np.arange(11, 18)
        adapted = ActionAdapter().adapt(action)
        np.testing.assert_array_equal(adapted.left_hand, np.arange(1, 8, dtype=np.float32))
        np.testing.assert_array_equal(adapted.right_hand, np.arange(11, 18, dtype=np.float32))
        self.assertIsNone(adapted.neck)
        self.assertEqual(adapted.source_dim, 78)

    def test_token_is_quantized_but_hands_are_verbatim(self) -> None:
        action = np.zeros(78, dtype=np.float32)
        action[:64] = 0.031  # rounds down to 0.0
        action[64] = 0.031
        adapted = ActionAdapter().adapt(action)
        self.assertTrue(np.all(adapted.token == np.float32(0.0)))
        self.assertEqual(adapted.left_hand[0], np.float32(0.031))

    def test_80d_zero_neck_is_accepted_as_a_no_op(self) -> None:
        action = np.zeros(80, dtype=np.float32)
        action[:64] = 0.2
        adapted = ActionAdapter().adapt(action)
        self.assertEqual(adapted.source_dim, 80)
        self.assertIsNotNone(adapted.neck)
        np.testing.assert_array_equal(adapted.neck, np.zeros(2, dtype=np.float32))

    def test_80d_meaningful_neck_is_rejected(self) -> None:
        action = np.zeros(80, dtype=np.float32)
        action[78] = 0.2
        with self.assertRaises(ActionAdapterError):
            ActionAdapter().adapt(action)

    def test_80d_neck_within_tolerance_is_accepted_and_above_is_not(self) -> None:
        within = np.zeros(80, dtype=np.float32)
        within[78:80] = 0.04
        self.assertIsNotNone(ActionAdapter(neck_tolerance=0.05).adapt(within).neck)
        above = np.zeros(80, dtype=np.float32)
        above[79] = 0.06
        with self.assertRaises(ActionAdapterError):
            ActionAdapter(neck_tolerance=0.05).adapt(above)

    def test_non_finite_neck_is_rejected(self) -> None:
        action = np.zeros(80, dtype=np.float32)
        action[78] = np.nan
        with self.assertRaises(ActionAdapterError):
            ActionAdapter().adapt(action)


class ActionValidationTest(unittest.TestCase):
    def test_wrong_widths_are_rejected(self) -> None:
        for width in (0, 64, 77, 79, 128):
            with self.subTest(width=width):
                with self.assertRaises(ActionAdapterError):
                    ActionAdapter().pack(np.zeros(width, dtype=np.float32))

    def test_non_finite_action_is_rejected(self) -> None:
        action = np.zeros(78, dtype=np.float32)
        action[3] = np.inf
        with self.assertRaises(ActionAdapterError):
            ActionAdapter().pack(action)

    def test_single_row_batch_is_accepted_and_multi_row_is_not(self) -> None:
        adapter = ActionAdapter()
        self.assertEqual(adapter.adapt(np.zeros((1, 78), dtype=np.float32)).token.shape, (64,))
        with self.assertRaises(ActionAdapterError):
            adapter.adapt(np.zeros((2, 78), dtype=np.float32))

    def test_frame_index_is_monotonic_and_a_rejection_consumes_nothing(self) -> None:
        adapter = ActionAdapter()
        self.assertEqual(frame_index(adapter.pack(np.zeros(78, dtype=np.float32))), 0)
        with self.assertRaises(ActionAdapterError):
            adapter.pack(np.zeros(77, dtype=np.float32))
        self.assertEqual(frame_index(adapter.pack(np.zeros(78, dtype=np.float32))), 1)

    def test_explicit_frame_index_is_honoured_and_must_advance(self) -> None:
        adapter = ActionAdapter()
        self.assertEqual(frame_index(adapter.pack(np.zeros(78, dtype=np.float32), frame_index=5)), 5)
        self.assertEqual(frame_index(adapter.pack(np.zeros(78, dtype=np.float32))), 6)
        with self.assertRaises(ActionAdapterError):
            adapter.pack(np.zeros(78, dtype=np.float32), frame_index=6)
        with self.assertRaises(ActionAdapterError):
            adapter.pack(np.zeros(78, dtype=np.float32), frame_index=-1)

    def test_declared_action_dim_pins_the_accepted_width(self) -> None:
        adapter = ActionAdapter(expected_dim=80)
        self.assertEqual(adapter.adapt(np.zeros(80, dtype=np.float32)).source_dim, 80)
        with self.assertRaises(ActionAdapterError):
            adapter.adapt(np.zeros(78, dtype=np.float32))

        adapter = ActionAdapter(expected_dim=78)
        self.assertEqual(adapter.adapt(np.zeros(78, dtype=np.float32)).source_dim, 78)
        with self.assertRaises(ActionAdapterError):
            adapter.adapt(np.zeros(80, dtype=np.float32))

        with self.assertRaises(ActionAdapterError):
            ActionAdapter(expected_dim=79)

    def test_reset_restarts_the_frame_sequence(self) -> None:
        # Pinned SONIC reads frame_index only for its debug log
        # (gear_sonic_deploy .../zmq_endpoint_interface.hpp, protocol v4), so a
        # new Start may begin at 0 again; each packed message is complete.
        adapter = ActionAdapter()
        adapter.pack(np.zeros(78, dtype=np.float32))
        adapter.reset()
        self.assertEqual(frame_index(adapter.pack(np.zeros(78, dtype=np.float32))), 0)


class FsqQuantizationTest(unittest.TestCase):
    def test_values_land_on_the_grid_and_are_clipped(self) -> None:
        values = np.array([-5.0, -0.03, 0.0, 0.03, 0.04, 0.62, 5.0], dtype=np.float32)
        quantized = fsq_quantize(values)
        expected = np.array([-0.625, 0.0, 0.0, 0.0, 0.0625, 0.625, 0.625], dtype=np.float32)
        np.testing.assert_allclose(quantized, expected, atol=1e-7)
        self.assertTrue(np.all(np.abs(quantized) <= 0.625 + 1e-7))
        np.testing.assert_allclose(quantized / 0.0625, np.round(quantized / 0.0625), atol=1e-4)
