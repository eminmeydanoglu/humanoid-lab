"""Source loaders: standing completion, name-based remaps, fail-closed gaps."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_lab.controllers.sonic import SONIC_REFERENCE_JOINT_ORDER
from humanoid_lab.datasets.sonic.adapters import nvidia_apple_to_plate, nvidia_fruits, unitree_dex3
from humanoid_lab.datasets.sonic.encoder_observation import (
    OrientationPolicy,
    build_g1_encoder_observation,
)
from humanoid_lab.datasets.sonic.joints import canonical_hand, hand_order, sonic_reference_body
from humanoid_lab.datasets.sonic.pilot import PilotSpec, map_source_episode
from humanoid_lab.datasets.sonic.quality import permutation_round_trip
from humanoid_lab.datasets.sonic.reference import (
    STANDING_COMPLETION_SCOPE,
    compose_unitree_static_completion,
    deployment_standing_pose,
    load_standing_pose,
)

RAW_ROOT = Path("/data/datasets/first_tur_ham")
FRUITS_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint", "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
)


class PilotConfigTest(unittest.TestCase):
    def test_bulk_conversion_policy_is_loaded_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "pilots.json"
            config.write_text(json.dumps({
                "apple": {
                    "dataset": "apple-dataset",
                    "episode": 7,
                    "bulk_conversion_status": "excluded",
                    "bulk_conversion_reason": "human review rejected the source",
                }
            }), encoding="utf-8")
            spec = PilotSpec.from_config(config, "apple", raw_root=Path("/raw"))
        self.assertEqual(spec.bulk_conversion_status, "excluded")
        self.assertEqual(spec.bulk_conversion_reason, "human review rejected the source")


class StaticCompletionTest(unittest.TestCase):
    def _compose(self, seconds: float):
        frames = int(seconds * 30.0) + 1
        timestamps = np.arange(frames, dtype=np.float64) / 30.0
        arms = np.tile(np.arange(14, dtype=np.float32), (frames, 1)) * 0.01
        hands = np.zeros((frames, 7), dtype=np.float32)
        return compose_unitree_static_completion(
            timestamps, arms, hands, hands, standing=deployment_standing_pose()
        )

    def test_lower_body_and_root_are_frozen_for_any_length(self) -> None:
        standing = deployment_standing_pose()
        arm_ids = [SONIC_REFERENCE_JOINT_ORDER.index(name) for name in __import__(
            "humanoid_lab.datasets.sonic.reference", fromlist=["ARM_NAMES"]
        ).ARM_NAMES]
        for episode in (self._compose(2.0), self._compose(30.0)):
            frozen = [index for index in range(29) if index not in arm_ids]
            np.testing.assert_allclose(
                episode.joint_pos[:, frozen], np.tile(standing.joint_pos[frozen], (len(episode.timestamps), 1)), atol=1e-6
            )
            np.testing.assert_allclose(episode.joint_vel[:, frozen], 0.0, atol=1e-6)
            np.testing.assert_allclose(episode.body_pos, np.tile(standing.body_pos, (len(episode.timestamps), 1)), atol=1e-9)
            np.testing.assert_array_equal(episode.body_quat_wxyz, np.tile(standing.body_quat_wxyz, (len(episode.timestamps), 1)))

    def test_long_episodes_do_not_need_a_capture(self) -> None:
        """The 30 s episode exists precisely because no captured trajectory covers it."""
        episode = self._compose(30.0)
        self.assertGreater(len(episode.timestamps), 800)
        self.assertTrue(np.isfinite(episode.joint_pos).all())
        observation, clamp = build_g1_encoder_observation(episode)
        self.assertTrue(np.isfinite(observation).all())
        self.assertGreater(float(clamp[-1]), 0.0)

    def test_velocities_are_rederived_from_the_composed_positions(self) -> None:
        episode = self._compose(2.0)
        np.testing.assert_allclose(episode.joint_vel[:-1], np.diff(episode.joint_pos, axis=0) * 50.0, atol=1e-6)
        np.testing.assert_allclose(episode.joint_vel[-1], episode.joint_vel[-2], atol=1e-6)

    def test_standing_pose_loader_refuses_a_moving_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savetxt(directory / "joint_pos.csv", np.zeros((3, 29)), delimiter=",")
            np.savetxt(directory / "body_pos.csv", np.zeros((3, 3)), delimiter=",")
            np.savetxt(directory / "body_quat.csv", np.tile([1.0, 0, 0, 0], (3, 1)), delimiter=",")
            (directory / "provenance.json").write_text(json.dumps({"lower_body_is_time_series": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_standing_pose(directory)


class NameMappingTest(unittest.TestCase):
    def test_unitree_hand_fields_are_explicit_and_complete(self) -> None:
        for side, expected in (
            ("left", ("kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2", "kLeftHandMiddle0",
                      "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1")),
            ("right", ("kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2", "kRightHandIndex0",
                       "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1")),
        ):
            self.assertEqual(unitree_dex3.HAND_SOURCE_NAMES[side], expected)
            self.assertTrue(set(expected) <= set(unitree_dex3.UNITREE_ACTION_NAMES))

    def test_fruits_hand_names_round_trip_into_canonical_dex3_order(self) -> None:
        values = np.arange(7, dtype=np.float32)[None, :]
        names = FRUITS_NAMES[22:29]
        canonical = canonical_hand(values, names, "left")
        # Source order is index_0, index_1, middle_0, middle_1, thumb_0/1/2;
        # canonical Dex3 order is thumb_0/1/2, middle_0/1, index_0/1.
        np.testing.assert_array_equal(canonical[0], [4, 5, 6, 2, 3, 0, 1])
        self.assertTrue(permutation_round_trip(tuple(names), hand_order("left"))["ok"])

    def test_fruits_body_names_round_trip_into_reference_order(self) -> None:
        body_names = FRUITS_NAMES[:22] + FRUITS_NAMES[29:36]
        values = np.arange(29, dtype=np.float32)[None, :]
        reordered = sonic_reference_body(values, body_names)
        self.assertEqual(reordered.shape, (1, 29))
        self.assertTrue(permutation_round_trip(body_names, SONIC_REFERENCE_JOINT_ORDER)["ok"])
        # left_hip_pitch stays first, right_leg follows, waist is interleaved.
        self.assertEqual(float(reordered[0, 0]), 0.0)
        self.assertEqual(float(reordered[0, 1]), float(values[0, body_names.index("right_hip_pitch_joint")]))

    def test_permutation_round_trip_detects_a_bad_order(self) -> None:
        report = permutation_round_trip(("a", "b"), ("a", "c"))
        self.assertFalse(report["ok"])


class AppleFailClosedTest(unittest.TestCase):
    def test_body_order_is_verified_so_the_legacy_flag_is_not_required(self) -> None:
        """The block-interior order is verified now; the old flag is accepted but not required."""
        self.assertTrue(nvidia_apple_to_plate.BODY_ORDER_VERIFIED)
        self.assertEqual(nvidia_apple_to_plate.BODY_ORDER, "unitree_mujoco_order_within_each_modality_block")
        self.assertTrue(nvidia_apple_to_plate.BODY_ORDER_EVIDENCE)
        with self.assertRaises(FileNotFoundError):
            nvidia_apple_to_plate.load_body_pilot(
                Path("/nonexistent"), 0,
                standing=deployment_standing_pose(),
                assume_unitree_mujoco_body_order=False,
            )

    def test_hand_schema_is_still_blocked(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            nvidia_apple_to_plate.require_hand_schema()

    def test_body_order_evidence_is_recorded(self) -> None:
        self.assertIn("joint-limit", " ".join(nvidia_apple_to_plate.BODY_ORDER_EVIDENCE))

    def test_hand_blocks_are_already_in_canonical_dex3_motor_order(self) -> None:
        """Both blocks are canonical per side, so the slot map is the identity.

        The earlier index-first reading was wrong; this pins the verified order.
        """
        self.assertEqual(nvidia_apple_to_plate.HAND_BLOCK_IDENTITY_MAP, (0, 1, 2, 3, 4, 5, 6))
        self.assertEqual(
            nvidia_apple_to_plate.LEFT_HAND_ORDER,
            ("thumb_0_joint", "thumb_1_joint", "thumb_2_joint", "middle_0_joint", "middle_1_joint",
             "index_0_joint", "index_1_joint"),
        )
        self.assertEqual(
            nvidia_apple_to_plate.RIGHT_HAND_ORDER,
            ("thumb_0_joint", "thumb_1_joint", "thumb_2_joint", "index_0_joint", "index_1_joint",
             "middle_0_joint", "middle_1_joint"),
        )
        self.assertNotEqual(nvidia_apple_to_plate.LEFT_HAND_ORDER[0], "index_0_joint")
        self.assertEqual(
            nvidia_apple_to_plate.HAND_BLOCK_SLICES["left"], slice(29, 36)
        )
        self.assertEqual(
            nvidia_apple_to_plate.HAND_BLOCK_SLICES["right"], slice(36, 43)
        )

    def test_schema_validation_accepts_the_documented_evidence(self) -> None:
        nvidia_apple_to_plate.validate_hand_schema(_evidence())

    def test_schema_validation_is_fail_closed(self) -> None:
        for patch, message in (
            ({"right_nonzero_frames": 1, "right_max_abs": 0.2}, "nonzero right"),
            ({"left_pair_max_abs_error": 0.1}, "pairs"),
            ({"left_min": [-1.0, -1.0, -1.0, -1.0, -0.5, -0.1, 0.0]}, "envelope"),
            ({"closed_frames": 0}, "never closes"),
        ):
            with self.subTest(patch=patch):
                with self.assertRaisesRegex(ValueError, message):
                    nvidia_apple_to_plate.validate_hand_schema(_evidence(**patch))

    def test_hand_command_is_declared_normalized_not_radians(self) -> None:
        self.assertEqual(nvidia_apple_to_plate.DOCUMENTED_CLOSED_TUPLE, (-1.0, -1.0, -1.0, -1.0, 0.0, 0.4, 0.7))
        self.assertFalse(any(value > 1.0 for value in nvidia_apple_to_plate.LEFT_NORMALIZED_ENVELOPE[0]))


def _evidence(**overrides) -> dict:
    base = {
        "episodes": 1,
        "frames": 249,
        "left_min": [-1.0, -1.0, -1.0, -1.0, -0.5, 0.0, 0.0],
        "left_max": [-0.0, -0.0, -0.0, -0.0, 0.5, 0.4, 0.7],
        "left_non_finite": 0,
        "left_pair_max_abs_error": 0.0,
        "left_negative_counts": [100, 100, 90, 90, 20, 0, 0],
        "right_nonzero_frames": 0,
        "right_max_abs": 0.0,
        "closed_frames": 55,
        "closed_mean_tuple": [-1.0, -1.0, -0.96, -0.96, -0.02, 0.4, 0.7],
        "documented_closed_tuple": list(nvidia_apple_to_plate.DOCUMENTED_CLOSED_TUPLE),
        "interior_order": list(nvidia_apple_to_plate.LEFT_HAND_ORDER),
        "source_to_canonical_slot": list(nvidia_apple_to_plate.HAND_BLOCK_IDENTITY_MAP),
    }
    base.update(overrides)
    return base


@unittest.skipUnless(RAW_ROOT.is_dir(), "raw collection is not mounted")
class RealEpisodeTest(unittest.TestCase):
    unitree_dataset = RAW_ROOT / "unitree-g1-dex3/G1_Dex3_ObjectPlacement_Dataset"
    fruits_dataset = RAW_ROOT / "nvidia-g1-fruits-1k/g1-pick-apple"
    apple_dataset = RAW_ROOT / "nvidia-gr00t-n1.7-apple-to-plate"

    def test_unitree_episode_is_static_lower_body_and_measured_arms(self) -> None:
        spec = PilotSpec("unitree", self.unitree_dataset, 74, "unitree_pilot_test")
        build = map_source_episode(spec, deployment_standing_pose())
        self.assertEqual(build.hand_schema_status, "verified")
        self.assertEqual(build.provenance["standing_completion_scope"], STANDING_COMPLETION_SCOPE)
        self.assertEqual(build.provenance["standing_completion_policy"], "static_stable_frame")
        episode = build.episode
        self.assertEqual(episode.joint_pos.shape[1], 29)
        self.assertGreater(len(episode.timestamps), 100)
        self.assertEqual(build.provenance["state_sources"]["left_leg"], "synthetic_static_standing")

    def test_fruits_episode_resamples_20hz_to_50hz(self) -> None:
        spec = PilotSpec("fruits", self.fruits_dataset, 0, "fruits_pilot_test")
        build = map_source_episode(spec, deployment_standing_pose())
        self.assertEqual(build.provenance["source_fps"], 20.0)
        self.assertEqual(len(build.episode.timestamps), 276)
        self.assertTrue(np.isfinite(build.episode.joint_pos).all())
        self.assertGreater(float(np.abs(build.episode.left_hand_joints).max()), 0.0)

    def test_apple_episode_stays_a_body_latent_pilot(self) -> None:
        spec = PilotSpec("apple", self.apple_dataset, 179, "apple_pilot_test",
                         assume_unitree_mujoco_body_order=True)
        build = map_source_episode(spec, deployment_standing_pose())
        self.assertFalse(build.final_action_allowed())
        np.testing.assert_array_equal(build.episode.left_hand_joints, 0.0)
        self.assertTrue(build.provenance["body_joint_order_verified"])
        with self.assertRaises(ValueError):
            nvidia_fruits.action_slices(self.apple_dataset)

    def test_apple_raw_hand_command_is_preserved_outside_the_canonical_slots(self) -> None:
        spec = PilotSpec("apple", self.apple_dataset, 179, "apple_pilot_test",
                         assume_unitree_mujoco_body_order=True)
        build = map_source_episode(spec, deployment_standing_pose())
        raw = build.raw_hand_command
        self.assertIsNotNone(raw)
        self.assertEqual(raw["left"].shape[1], 7)
        self.assertEqual(raw["right"].shape[1], 7)
        self.assertEqual(raw["left"].shape[0], raw["right"].shape[0])
        # The raw command is normalized: it must not look like the radian slots.
        self.assertLessEqual(float(np.abs(raw["left"]).max()), 1.0 + 1e-6)
        np.testing.assert_array_equal(raw["right"], 0.0)
        self.assertIs(build.provenance["hand_command_is_radians"], False)
        self.assertTrue(build.provenance["hand_command_evidence"]["closed_frames"] > 0)


if __name__ == "__main__":
    unittest.main()
