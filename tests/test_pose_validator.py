"""Production policy pose packets fail closed before reaching the SONIC port."""

from __future__ import annotations

import json
import unittest
from typing import Callable

import numpy as np

from humanoid_lab.datasets.sonic.protocol_v4 import HEADER_SIZE, pack_latent_action_message
from humanoid_lab.psi0_bridge.pose_validator import PosePacketError, validate_pose_packet


def good_packet() -> bytes:
    return pack_latent_action_message(
        np.linspace(-1, 1, 64, dtype=np.float32), 17,
        np.arange(7, dtype=np.float32), -np.arange(7, dtype=np.float32),
    )


def replace_header(packet: bytes, change: Callable[[dict], None]) -> bytes:
    header = json.loads(packet[4:4 + HEADER_SIZE].rstrip(b"\0"))
    change(header)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return b"pose" + encoded.ljust(HEADER_SIZE, b"\0") + packet[4 + HEADER_SIZE:]


class PoseValidatorTest(unittest.TestCase):
    def test_accepts_complete_packed_packet(self) -> None:
        self.assertEqual(validate_pose_packet(good_packet()), 17)

    def test_rejects_missing_hand_and_bad_metadata(self) -> None:
        no_hands = pack_latent_action_message(np.zeros(64, dtype=np.float32), 0)
        mutations = (
            no_hands,
            replace_header(good_packet(), lambda h: h.update(v=3)),
            replace_header(good_packet(), lambda h: h.update(endian="be")),
            replace_header(good_packet(), lambda h: h.update(count=2)),
            replace_header(good_packet(), lambda h: h["fields"][0].update(dtype="f64")),
            replace_header(good_packet(), lambda h: h["fields"][2].update(shape=[1, 6])),
            replace_header(good_packet(), lambda h: h["fields"].reverse()),
        )
        for packet in mutations:
            with self.subTest(packet=packet[:55]), self.assertRaises(PosePacketError):
                validate_pose_packet(packet)

    def test_rejects_bad_lengths_padding_and_nonfinite_values(self) -> None:
        packet = good_packet()
        for corrupted in (
            b"bad!" + packet[4:], packet[:-1], packet + b"\0",
            packet[:100] + b"X" + packet[101:],
            packet[:4 + HEADER_SIZE] + np.array([np.nan], dtype="<f4").tobytes()
            + packet[4 + HEADER_SIZE + 4:],
            packet[:4 + HEADER_SIZE + 256 + 8 + 28]
            + np.array([np.inf], dtype="<f4").tobytes()
            + packet[4 + HEADER_SIZE + 256 + 8 + 28 + 4:],
            packet[:4 + HEADER_SIZE + 256] + np.array([-1], dtype="<i8").tobytes()
            + packet[4 + HEADER_SIZE + 264:],
        ):
            with self.subTest(length=len(corrupted)), self.assertRaises(PosePacketError):
                validate_pose_packet(corrupted)
