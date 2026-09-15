from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER
from humanoid_lab.datasets.sonic.adapters.nvidia_apple_to_plate import require_hand_schema
from humanoid_lab.datasets.sonic.encoder_observation import build_g1_encoder_observation, quaternion_to_rotation_6d
from humanoid_lab.datasets.sonic.joints import permutation, reorder
from humanoid_lab.datasets.sonic.provenance import assert_processed_destination
from humanoid_lab.datasets.sonic.protocol_v1 import LOOKAHEAD_FRAMES, pack_command_message, pack_pose_message, staged_windows
from humanoid_lab.datasets.sonic.reference import compose_unitree_standing_reference
from humanoid_lab.datasets.sonic.schema import compose_action
from humanoid_lab.datasets.sonic.timeline import finite_difference, uniform_timeline


class SonicDatasetContractsTest(unittest.TestCase):
    @staticmethod
    def _compose(timestamps: np.ndarray, arms: np.ndarray, hands: np.ndarray):
        idle_timestamps = np.arange(101, dtype=np.float64) / 50.0
        idle_q = np.zeros((101, 29), dtype=np.float32)
        idle_q[:, :15] = idle_timestamps[:, None]
        idle_qd = np.zeros_like(idle_q)
        idle_qd[:, :15] = 1.0
        return compose_unitree_standing_reference(
            timestamps, arms, hands, hands,
            idle_timestamps=idle_timestamps,
            idle_joint_pos=idle_q,
            idle_joint_vel=idle_qd,
            idle_body_pos=np.column_stack((np.zeros(101), np.zeros(101), np.full(101, 0.8))),
            idle_body_quat_wxyz=np.tile([1, 0, 0, 0], (101, 1)),
        )

    def test_duplicate_and_missing_joints_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            permutation(["a", "a"], ["a"])
        with self.assertRaisesRegex(ValueError, "missing"):
            permutation(["a"], ["a", "b"])

    def test_30_to_50_hz_and_stationary_tail(self) -> None:
        source = np.arange(31) / 30.0
        target = uniform_timeline(source)
        self.assertEqual(len(target), 51)
        qdot = finite_difference(np.ones((4, 2), dtype=np.float32))
        np.testing.assert_array_equal(qdot, 0)

    def test_reference_and_exact_encoder_layout(self) -> None:
        timestamps = np.arange(31) / 30.0
        arms = np.tile(np.arange(14, dtype=np.float32), (31, 1))
        hands = np.zeros((31, 7), dtype=np.float32)
        episode = self._compose(timestamps, arms, hands)
        # Lower body is the IDLE time series; arms are absolute replacements.
        np.testing.assert_allclose(episode.joint_pos[:, 0], episode.timestamps)
        arm_names = (
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
            "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        )
        arm_ids = [SONIC_REFERENCE_JOINT_ORDER.index(name) for name in arm_names]
        np.testing.assert_allclose(episode.joint_pos[:, arm_ids], np.tile(arms[0], (len(episode.timestamps), 1)))
        np.testing.assert_allclose(episode.joint_vel[:, 0], 1.0)
        np.testing.assert_allclose(episode.joint_vel[:, arm_ids], 0.0)
        observation, clamp = build_g1_encoder_observation(episode)
        self.assertEqual(observation.shape, (51, 1751))
        np.testing.assert_array_equal(observation[:, :4], 0)
        np.testing.assert_array_equal(observation[:, 644:], 0)
        self.assertEqual(float(clamp[0]), 0.0)
        self.assertGreater(float(clamp[-1]), 0.0)

    def test_official_reference_and_hardware_orders_round_trip_by_name(self) -> None:
        self.assertNotEqual(BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER)
        marker = np.arange(29, dtype=np.float32)[None, :]
        hardware = reorder(marker, SONIC_REFERENCE_JOINT_ORDER, BODY_JOINT_ORDER)
        restored = reorder(hardware, BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER)
        np.testing.assert_array_equal(restored, marker)
        for name in SONIC_REFERENCE_JOINT_ORDER:
            self.assertEqual(
                hardware[0, BODY_JOINT_ORDER.index(name)],
                marker[0, SONIC_REFERENCE_JOINT_ORDER.index(name)],
                name,
            )

    def test_upright_rotation_6d(self) -> None:
        actual = quaternion_to_rotation_6d(np.array([[1, 0, 0, 0]], dtype=np.float32))
        np.testing.assert_array_equal(actual, [[1, 0, 0, 1, 0, 0]])

    def test_action_shape_and_apple_blocker(self) -> None:
        self.assertEqual(compose_action(np.zeros(64), np.zeros(7), np.zeros(7)).shape, (78,))
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            require_hand_schema()

    def test_raw_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "first_tur_ham"
            source.mkdir()
            with self.assertRaises(ValueError):
                assert_processed_destination(source, source / "bad")
            assert_processed_destination(source, Path(root) / "first_tur_processed")

    def test_protocol_v1_lookahead_and_tail_clamp(self) -> None:
        timestamps = np.arange(4) / 30.0
        episode = self._compose(timestamps, np.zeros((4, 14)), np.zeros((4, 7)))
        frame_index, window = next(staged_windows(episode))
        self.assertEqual(len(frame_index), LOOKAHEAD_FRAMES)
        np.testing.assert_array_equal(window.joint_pos[-1], episode.joint_pos[-1])
        self.assertTrue(pack_pose_message(window, frame_index).startswith(b"pose"))
        self.assertTrue(pack_command_message(start=True, stop=False, planner=False).startswith(b"command"))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            pack_pose_message(window, frame_index * 2)


if __name__ == "__main__":
    unittest.main()
