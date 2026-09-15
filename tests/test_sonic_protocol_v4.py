"""SONIC latent Protocol v4 packing contract."""

from __future__ import annotations

import json
import unittest

import numpy as np

from humanoid_lab.datasets.sonic.protocol_v4 import HEADER_SIZE, pack_latent_action_message


class ProtocolV4Test(unittest.TestCase):
    def test_exact_field_order_shapes_and_payload(self) -> None:
        token = np.arange(64, dtype=np.float32)
        left = np.arange(7, dtype=np.float32) * 0.1
        right = -left
        packed = pack_latent_action_message(token, 12, left, right)
        self.assertTrue(packed.startswith(b"pose"))
        header_start = len(b"pose")
        header = json.loads(packed[header_start:header_start + HEADER_SIZE].rstrip(b"\0"))
        self.assertEqual(header["v"], 4)
        self.assertEqual(
            header["fields"],
            [
                {"name": "token_state", "dtype": "f32", "shape": [1, 64]},
                {"name": "frame_index", "dtype": "i64", "shape": [1]},
                {"name": "left_hand_joints", "dtype": "f32", "shape": [1, 7]},
                {"name": "right_hand_joints", "dtype": "f32", "shape": [1, 7]},
            ],
        )
        payload = packed[header_start + HEADER_SIZE:]
        offset = 0
        np.testing.assert_array_equal(np.frombuffer(payload[offset:offset + 256], dtype="<f4"), token)
        offset += 256
        self.assertEqual(int(np.frombuffer(payload[offset:offset + 8], dtype="<i8")[0]), 12)
        offset += 8
        np.testing.assert_allclose(np.frombuffer(payload[offset:offset + 28], dtype="<f4"), left)
        offset += 28
        np.testing.assert_allclose(np.frombuffer(payload[offset:offset + 28], dtype="<f4"), right)

    def test_hands_are_optional_and_bad_shapes_fail(self) -> None:
        packed = pack_latent_action_message(np.zeros(64, dtype=np.float32), np.array([0]))
        header = json.loads(packed[4:4 + HEADER_SIZE].rstrip(b"\0"))
        self.assertEqual([field["name"] for field in header["fields"]], ["token_state", "frame_index"])
        with self.assertRaises(ValueError):
            pack_latent_action_message(np.zeros(63), 0)
        with self.assertRaises(ValueError):
            pack_latent_action_message(np.zeros(64), [0, 1])
        with self.assertRaises(ValueError):
            pack_latent_action_message(np.zeros(64), 0, np.zeros(6))


if __name__ == "__main__":
    unittest.main()
