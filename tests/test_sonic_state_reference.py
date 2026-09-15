"""Narrow tests for the source-rate state references and the kinematic contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_lab.controllers.sonic import SONIC_REFERENCE_JOINT_ORDER
from humanoid_lab.datasets.sonic.adapters import nvidia_apple_to_plate, nvidia_fruits
from humanoid_lab.datasets.sonic.joints import hand_order
from humanoid_lab.datasets.sonic.quality import load_joint_limits
from humanoid_lab.datasets.sonic.state_reference import (
    StateReference,
    build_state_reference,
    hand_limit_report,
    load_state_reference,
    save_state_reference,
)

RAW_ROOT = Path("/data/datasets/first_tur_ham")
LIMITS_PATH = Path(__file__).resolve().parents[1] / "configs/datasets/sonic/g1_joint_limits.json"


def _synthetic(frames: int = 5) -> StateReference:
    timestamps = np.arange(frames, dtype=np.float64) / 20.0
    body = np.tile(np.arange(29, dtype=np.float64) * 0.01, (frames, 1))
    left = np.tile([-0.1, 0.05, 0.3, -0.4, -0.5, -0.2, -0.3], (frames, 1))
    right = np.tile([-0.1, -0.05, -0.2, 0.1, 0.2, -0.2, -0.1], (frames, 1))
    limits = load_joint_limits(LIMITS_PATH)
    return build_state_reference(
        timestamps, body, left, right,
        source_field="observation.state", source_fps=20.0, limits=limits,
        mapping={"test": True},
    )


class StateReferenceContractTest(unittest.TestCase):
    def test_round_trip_preserves_arrays_and_hands_flag(self) -> None:
        reference = _synthetic()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference_state.npz"
            save_state_reference(path, reference)
            restored = load_state_reference(path)
        np.testing.assert_allclose(restored.joint_pos, reference.joint_pos)
        np.testing.assert_allclose(restored.left_hand_joints, reference.left_hand_joints)
        self.assertEqual(restored.hands_applied, reference.hands_applied)
        self.assertEqual(restored.provenance["source_field"], "observation.state")

    def test_velocity_is_derived_at_source_rate(self) -> None:
        reference = _synthetic(6)
        expected = np.diff(reference.joint_pos, axis=0) * 20.0
        np.testing.assert_allclose(reference.joint_vel[:-1], expected, atol=1e-6)
        self.assertEqual(reference.joint_vel.shape, reference.joint_pos.shape)

    def test_kinematic_file_carries_every_key_the_replay_reads(self) -> None:
        payload = _synthetic().to_arrays()
        for key in ("timestamps", "joint_pos", "joint_vel", "left_hand_joints", "right_hand_joints", "hands_applied"):
            self.assertIn(key, payload)
        self.assertEqual(payload["joint_pos"].shape[1], 29)
        self.assertEqual(payload["left_hand_joints"].shape[1], 7)

    def test_sanitised_hand_channel_fails_the_limit_check(self) -> None:
        limits = load_joint_limits(LIMITS_PATH)
        bad = np.tile([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0], (4, 1))
        with self.assertRaises(ValueError):
            hand_limit_report(bad, bad, limits)

    def test_identity_order_is_the_canonical_dex3_order(self) -> None:
        self.assertEqual(hand_order("left")[0], "left_hand_thumb_0_joint")
        self.assertEqual(hand_order("right")[3], "right_hand_index_0_joint")


@unittest.skipUnless(RAW_ROOT.is_dir(), "raw collection is not mounted")
class RealStateReferenceTest(unittest.TestCase):
    fruits = RAW_ROOT / "nvidia-g1-fruits-1k/g1-pick-apple"
    apple = RAW_ROOT / "nvidia-gr00t-n1.7-apple-to-plate"

    def test_fruits_state_reference_is_source_rate_with_radian_hands(self) -> None:
        limits = load_joint_limits(LIMITS_PATH)
        reference = nvidia_fruits.load_state_reference(self.fruits, 0, limits=limits)
        self.assertEqual(reference.provenance["source_field"], "observation.state")
        self.assertEqual(reference.provenance["source_fps"], 20.0)
        self.assertTrue(reference.hands_applied)
        self.assertEqual(reference.joint_pos.shape, (111, 29))
        self.assertEqual(reference.left_hand_joints.shape, (111, 7))
        # Radian hands: the magnete is far above the normalized action envelope.
        self.assertGreater(float(np.abs(reference.left_hand_joints).max()), 0.5)

    def test_apple_state_hands_are_radians_in_canonical_order(self) -> None:
        limits = load_joint_limits(LIMITS_PATH)
        reference = nvidia_apple_to_plate.load_state_reference(self.apple, 179, limits=limits)
        self.assertEqual(reference.provenance["source_fps"], 30.0)
        self.assertTrue(reference.hands_applied)
        left = reference.left_hand_joints
        # Canonical Dex3 order signature: thumb_2 is a one-sided positive joint.
        self.assertGreaterEqual(float(left[:, 2].min()), 0.0)
        self.assertLessEqual(float(left[:, 3].max()), 0.0)
        self.assertLessEqual(float(left[:, 6].max()), 0.0)
        report = reference.provenance["hand_limit_report"]["sides"]["left"]
        self.assertEqual(report["beyond_tolerance_channels"], 0)

    def test_apple_action_source_reference_keeps_hands_blocked(self) -> None:
        reference = nvidia_apple_to_plate.load_action_source_reference(self.apple, 179)
        self.assertFalse(reference.hands_applied)
        np.testing.assert_array_equal(reference.left_hand_joints, 0.0)
        np.testing.assert_array_equal(reference.right_hand_joints, 0.0)
        self.assertEqual(reference.provenance["source_field"], "action")
        self.assertEqual(reference.joint_pos.shape, (249, 29))

    def test_fruits_action_source_reference_is_module_body_order(self) -> None:
        reference = nvidia_fruits.load_action_source_reference(self.fruits, 0)
        self.assertTrue(reference.hands_applied)
        self.assertEqual(reference.joint_pos.shape, (111, 29))
        self.assertEqual(len(SONIC_REFERENCE_JOINT_ORDER), 29)


if __name__ == "__main__":
    unittest.main()
