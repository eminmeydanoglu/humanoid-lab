"""Encoder observation semantics: layout, mode block, heading normalisation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_lab.datasets.sonic.encoder_observation import (
    ENCODER_INPUT_DIM,
    FUTURE_OFFSETS,
    G1_REQUIRED_OBSERVATIONS,
    PINNED_LAYOUT,
    OrientationPolicy,
    build_g1_encoder_observation,
    future_indices,
    load_encoder_layout,
    quaternion_to_rotation_6d,
    require_conversion_policy,
)
from humanoid_lab.datasets.sonic.heading import (
    calc_heading,
    calc_heading_quat,
    heading_relative_anchor_6d,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_to_rotation_matrix,
)
from humanoid_lab.datasets.sonic.schema import CanonicalEpisode

MODEL_DIR = Path("/data/models/sonic-isaac/sonic_v1_1")


def yaw(angle: float) -> np.ndarray:
    return quat_from_angle_axis(angle, (0.0, 0.0, 1.0))


def episode(frames: int, body_quat: np.ndarray, *, arms: float = 0.0) -> CanonicalEpisode:
    quaternions = np.asarray(body_quat, dtype=np.float64)
    if quaternions.ndim == 1:
        quaternions = np.tile(quaternions, (frames, 1))
    assert quaternions.shape == (frames, 4)
    timestamps = np.arange(frames, dtype=np.float64) / 50.0
    joint_pos = np.zeros((frames, 29), dtype=np.float32)
    joint_pos[:, 15:29] = arms
    return CanonicalEpisode(
        timestamps=timestamps,
        joint_pos=joint_pos,
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
        body_quat_wxyz=quaternions.astype(np.float32),
        body_pos=np.tile([0.0, 0.0, 0.79], (frames, 1)).astype(np.float32),
        left_hand_joints=np.zeros((frames, 7), dtype=np.float32),
        right_hand_joints=np.zeros((frames, 7), dtype=np.float32),
    )


class HeadingMathTest(unittest.TestCase):
    def test_heading_quat_extracts_yaw(self) -> None:
        self.assertAlmostEqual(float(calc_heading(yaw(0.7)[None, :])[0]), 0.7, places=6)
        heading = calc_heading_quat(yaw(0.7)[None, :])[0]
        np.testing.assert_allclose(heading, yaw(0.7), atol=1e-9)

    def test_heading_follows_yaw_only(self) -> None:
        # A rotation about the world x-axis leaves the reference x-axis untouched.
        self.assertAlmostEqual(float(calc_heading(quat_from_angle_axis(0.3, (1.0, 0.0, 0.0))[None, :])[0]), 0.0, places=9)
        self.assertAlmostEqual(float(calc_heading(yaw(1.1)[None, :])[0]), 1.1, places=9)

    def test_quat_mul_matches_hamilton_product(self) -> None:
        a = np.array([0.4, -0.3, 0.5, 0.7])
        b = np.array([0.6, 0.2, -0.1, 0.3])
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        expected = np.array(
            [
                a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
            ]
        )
        np.testing.assert_allclose(quat_mul(a, b), expected, atol=1e-12)
        np.testing.assert_allclose(quat_mul(quat_conjugate(a), a), [1, 0, 0, 0], atol=1e-12)

    def test_rotation_matrix_6d_packs_first_two_columns_row_wise(self) -> None:
        matrix = quat_to_rotation_matrix(yaw(0.4))
        six = quaternion_to_rotation_6d(yaw(0.4))
        np.testing.assert_allclose(six, [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[2, 0], matrix[2, 1]], atol=1e-6)

    def test_heading_relative_orientation_cancels_global_yaw(self) -> None:
        base = quat_from_angle_axis(0.25, (1.0, 0.0, 0.0))
        reference = np.stack([yaw(0.3), yaw(0.9), quat_mul(base, yaw(1.4))])
        rotated = np.stack([quat_mul(yaw(0.6), q) for q in reference])
        first = heading_relative_anchor_6d(np.broadcast_to(reference[0], reference.shape), reference)
        second = heading_relative_anchor_6d(np.broadcast_to(rotated[0], rotated.shape), rotated)
        np.testing.assert_allclose(first, second, atol=1e-6)


class EncoderLayoutTest(unittest.TestCase):
    def test_pinned_layout_offsets(self) -> None:
        self.assertEqual(PINNED_LAYOUT.total_dim, ENCODER_INPUT_DIM)
        self.assertEqual(PINNED_LAYOUT.span("encoder_mode_4"), slice(0, 4))
        self.assertEqual(PINNED_LAYOUT.span("motion_joint_positions_10frame_step5"), slice(4, 294))
        self.assertEqual(PINNED_LAYOUT.span("motion_joint_velocities_10frame_step5"), slice(294, 584))
        self.assertEqual(PINNED_LAYOUT.span("motion_anchor_orientation_heading_10frame_step5"), slice(584, 644))
        self.assertEqual(len(FUTURE_OFFSETS), 10)
        self.assertEqual(int(FUTURE_OFFSETS[-1]), 45)

    def test_layout_matches_pinned_observation_config(self) -> None:
        config = MODEL_DIR / "observation_config.yaml"
        if not config.is_file():
            self.skipTest("pinned model directory is not mounted")
        self.assertEqual(load_encoder_layout(config), PINNED_LAYOUT)

    def test_layout_rejects_a_different_config(self) -> None:
        config = MODEL_DIR / "observation_config.yaml"
        if not config.is_file():
            self.skipTest("pinned model directory is not mounted")
        document = config.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "observation_config.yaml"
            path.write_text(document.replace("motion_anchor_orientation_heading_10frame_step5", "motion_anchor_orientation_10frame_step5", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_encoder_layout(path)
            path.write_text(document.replace("mode_id: 0", "mode_id: 7", 1), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_encoder_layout(path)

    def test_required_g1_observations_are_the_zero_fill_set(self) -> None:
        self.assertEqual(len(G1_REQUIRED_OBSERVATIONS), 4)
        for name in G1_REQUIRED_OBSERVATIONS:
            self.assertIn(name, PINNED_LAYOUT.offsets)
        self.assertNotIn("smpl_joints_10frame_step1", G1_REQUIRED_OBSERVATIONS)


class OrientationPolicyTest(unittest.TestCase):
    def test_mode_block_uses_the_mode_id_not_one_hot(self) -> None:
        observation, _ = build_g1_encoder_observation(episode(4, yaw(0.0)))
        np.testing.assert_array_equal(observation[:, :4], np.zeros((4, 4), dtype=np.float32))
        np.testing.assert_array_equal(observation[:, 644:], np.zeros((4, ENCODER_INPUT_DIM - 644), dtype=np.float32))

    def test_conversion_policy_gate_is_fail_closed(self) -> None:
        self.assertEqual(
            require_conversion_policy(OrientationPolicy.REFERENCE_ROOT_CURRENT_FRAME),
            OrientationPolicy.REFERENCE_ROOT_CURRENT_FRAME,
        )
        for policy in (OrientationPolicy.LEGACY_RAW_WORLD, OrientationPolicy.MEASURED_BASE):
            with self.assertRaises(ValueError):
                require_conversion_policy(policy)

    def test_measured_base_policy_needs_a_measured_quaternion(self) -> None:
        with self.assertRaises(ValueError):
            build_g1_encoder_observation(episode(4, yaw(0.0)), policy=OrientationPolicy.MEASURED_BASE)

    def test_offline_policy_equals_refheading_semantics(self) -> None:
        """The offline assumption is upstream's refheading variant with identity delta heading."""
        frames = 6
        reference = np.stack([yaw(0.1 * index) for index in range(frames)])
        observation, _ = build_g1_encoder_observation(episode(frames, reference))
        block = observation[:, 584:644]
        for row in range(frames):
            manual = heading_relative_anchor_6d(np.broadcast_to(reference[row], (10, 4)), reference[np.minimum(row + FUTURE_OFFSETS, frames - 1)])
            np.testing.assert_allclose(block[row], manual.reshape(-1), atol=1e-6)

    def test_policy_changes_the_orientation_block(self) -> None:
        reference = np.stack([yaw(0.2 * index) for index in range(8)])
        heading_observation, _ = build_g1_encoder_observation(episode(8, reference))
        legacy_observation, _ = build_g1_encoder_observation(
            episode(8, reference), policy=OrientationPolicy.LEGACY_RAW_WORLD
        )
        self.assertGreater(np.abs(heading_observation[:, 584:644] - legacy_observation[:, 584:644]).max(), 0.1)


class FutureWindowTest(unittest.TestCase):
    def test_clamp_fraction_grows_towards_the_tail(self) -> None:
        indices, clamp = future_indices(60)
        self.assertEqual(indices.shape, (60, 10))
        self.assertEqual(float(clamp[0]), 0.0)
        np.testing.assert_array_equal(indices[0], FUTURE_OFFSETS)
        self.assertAlmostEqual(float(clamp[-1]), 0.9, places=6)
        np.testing.assert_array_equal(indices[-1], np.full(10, 59))

    def test_encoder_observation_is_finite_and_wide(self) -> None:
        observation, clamp = build_g1_encoder_observation(episode(12, yaw(0.3), arms=0.4))
        self.assertEqual(observation.shape, (12, ENCODER_INPUT_DIM))
        self.assertTrue(np.isfinite(observation).all())
        self.assertEqual(clamp.shape, (12,))
        # Joint blocks carry the reference joints, not zeros.
        np.testing.assert_allclose(observation[0, 4 + 15: 4 + 29], 0.4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
