"""Gate 0: the frozen conversion contract and the deterministic split.

These tests read the raw collection metadata under the data root, which is the
same input the split is built from. Nothing here writes into a source.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_lab.datasets.psi0_dex3.contract import (
    CANONICAL_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_PATH,
    LEGS_WAIST_SLICE,
    SOURCE_ARM_HAND_FIELDS,
    assert_source_state_names,
    assert_state_layout,
    load_config,
    source_state_permutation,
    standing_lower_body,
)
from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import HAND_SOURCE_NAMES, UNITREE_ACTION_NAMES
from humanoid_lab.datasets.psi0_dex3.split import (
    SCHEMA_VERSION,
    build_split,
    load_split_manifest,
    write_split_manifest,
)

TOTAL_USABLE_EPISODES = 3150
COLLECTIONS = 13


class ContractConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_contract_is_self_consistent(self) -> None:
        self.config.assert_contract()
        self.assertEqual(self.config.name, "psi0-unitree-dex3-sonic-v1")
        self.assertEqual(self.config.dataset_fps, 30)
        self.assertEqual(self.config.sonic_fps, 50)
        self.assertEqual(self.config.resample_method, "nearest_timestamp")
        self.assertEqual(self.config.output_root.name, "psi0-unitree-dex3-sonic-v1")
        self.assertEqual((self.config.train_repo, self.config.val_repo), ("train", "val"))

    def test_config_is_machine_readable(self) -> None:
        raw = self.config.raw
        self.assertEqual(raw["schema_version"], 1)
        for key in ("source", "camera", "fps", "resample", "action", "state", "model_contract", "split", "tasks"):
            self.assertIn(key, raw)

    def test_train_val_split_is_declared_but_not_yet_realised(self) -> None:
        self.assertAlmostEqual(self.config.val_fraction, 0.05, places=6)
        self.assertEqual(self.config.min_val_per_collection, 1)
        self.assertIsInstance(self.config.split_seed, int)

    def test_only_the_left_high_camera_is_a_training_input(self) -> None:
        self.assertEqual(self.config.camera_source_key, "observation.images.cam_left_high")
        self.assertEqual(self.config.camera_target_key, "observation.images.egocentric")
        self.assertEqual(self.config.camera_excluded_keys, ("observation.images.cam_right_high",))

    def test_13_collections_and_13_canonical_instructions(self) -> None:
        self.assertEqual(len(self.config.collections), COLLECTIONS)
        self.assertEqual(set(self.config.tasks), set(self.config.collection_names))
        for instruction in self.config.tasks.values():
            self.assertTrue(instruction.endswith("."), instruction)
            self.assertEqual(instruction, instruction.strip())

    def test_grasp_square_instruction_is_corrected(self) -> None:
        """The upstream label of GraspSquare is CameraPackaging's, not a grasp."""
        instruction = self.config.tasks["G1_Dex3_GraspSquare_Dataset"]
        self.assertNotEqual(instruction.lower(), "camera packaging")
        self.assertIn("square", instruction.lower())
        self.assertNotEqual(instruction, self.config.tasks["G1_Dex3_CameraPackaging_Dataset"])

    def test_no_two_collections_share_an_instruction(self) -> None:
        instructions = list(self.config.tasks.values())
        self.assertEqual(len(set(instructions)), len(instructions))

    def test_excluded_episodes_are_exactly_the_two_frozen_ones(self) -> None:
        excluded = {collection.name: collection.excluded for collection in self.config.collections if collection.excluded}
        self.assertEqual(
            excluded,
            {"G1_Dex3_ObjectPlacement_Dataset": (0,), "G1_Dex3_ToastedBread_Dataset": (396,)},
        )
        for collection in self.config.collections:
            if collection.excluded:
                self.assertTrue(collection.exclusion_reason)


class CanonicalStateLayoutTest(unittest.TestCase):
    def test_names_are_43_and_unique(self) -> None:
        assert_state_layout()
        self.assertEqual(len(CANONICAL_STATE_NAMES), 43)
        self.assertEqual(len(set(CANONICAL_STATE_NAMES)), 43)

    def test_group_boundaries_match_the_plan(self) -> None:
        self.assertEqual(CANONICAL_STATE_NAMES[LEGS_WAIST_SLICE.start], "left_hip_pitch_joint")
        self.assertEqual(CANONICAL_STATE_NAMES[LEGS_WAIST_SLICE.stop], "left_shoulder_pitch_joint")
        self.assertEqual(CANONICAL_STATE_NAMES[15], "left_shoulder_pitch_joint")
        self.assertEqual(CANONICAL_STATE_NAMES[29], "left_hand_thumb_0_joint")
        self.assertEqual(CANONICAL_STATE_NAMES[28], "right_wrist_yaw_joint")
        self.assertEqual(len(CANONICAL_HAND_NAMES), 14)

    def test_first_state_is_a_leg_and_hands_start_at_29(self) -> None:
        """The Psi0 SONIC contract: qpos(29) hardware order, then the hands."""
        self.assertTrue(CANONICAL_STATE_NAMES[0].startswith("left_hip"))
        self.assertTrue(all("_hand_" not in name for name in CANONICAL_STATE_NAMES[:29]))
        self.assertTrue(all("_hand_" in name for name in CANONICAL_STATE_NAMES[29:]))

    def test_standing_proxy_is_the_15d_lower_body(self) -> None:
        standing = standing_lower_body()
        self.assertEqual(standing.shape, (15,))
        self.assertTrue(np.isfinite(standing).all())
        self.assertEqual(
            standing.tolist(),
            [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0],
        )


class SourceNameMappingTest(unittest.TestCase):
    def test_raw_28d_order_maps_identity_by_name(self) -> None:
        permutation = source_state_permutation(UNITREE_ACTION_NAMES)
        self.assertEqual(permutation, tuple(range(28)))

    def test_declared_raw_fields_match_the_source_schema(self) -> None:
        self.assertEqual(SOURCE_ARM_HAND_FIELDS[0], "kLeftShoulderPitch")
        self.assertEqual(SOURCE_ARM_HAND_FIELDS[13], "kRightWristYaw")
        self.assertEqual(SOURCE_ARM_HAND_FIELDS[14], "kLeftHandThumb0")
        self.assertEqual(SOURCE_ARM_HAND_FIELDS[-1], "kRightHandMiddle1")
        self.assertEqual(set(SOURCE_ARM_HAND_FIELDS), set(UNITREE_ACTION_NAMES))

    def test_hand_field_map_agrees_with_the_sonic_adapter(self) -> None:
        for side in ("left", "right"):
            declared = tuple(SOURCE_ARM_HAND_FIELDS[14:21] if side == "left" else SOURCE_ARM_HAND_FIELDS[21:28])
            self.assertEqual(declared, HAND_SOURCE_NAMES[side])

    def test_a_reordered_source_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "28D Unitree Dex3"):
            assert_source_state_names(tuple(reversed(UNITREE_ACTION_NAMES)))
        with self.assertRaisesRegex(ValueError, "28D Unitree Dex3"):
            assert_source_state_names(UNITREE_ACTION_NAMES[:-1])


class SplitManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.manifest = build_split(cls.config)

    def test_totals_are_the_full_usable_corpus(self) -> None:
        totals = self.manifest["totals"]
        self.assertEqual(totals["usable"], TOTAL_USABLE_EPISODES)
        self.assertEqual(totals["train"] + totals["val"], TOTAL_USABLE_EPISODES)
        self.assertAlmostEqual(totals["val_fraction_realized"], 0.05, delta=0.002)

    def test_descending_from_3152_source_episodes(self) -> None:
        declared = sum(entry["total"] for entry in self.manifest["collections"].values())
        self.assertEqual(declared, 3152)
        self.assertEqual(declared - TOTAL_USABLE_EPISODES, 2)

    def test_every_collection_is_represented_in_validation(self) -> None:
        for name, entry in self.manifest["collections"].items():
            self.assertTrue(entry["val"], name)
            self.assertGreaterEqual(len(entry["val"]), self.config.min_val_per_collection, name)

    def test_splits_are_disjoint_and_cover_the_usable_episodes(self) -> None:
        for name, entry in self.manifest["collections"].items():
            train, val = entry["train"], entry["val"]
            self.assertFalse(set(train) & set(val), name)
            self.assertEqual(sorted(train + val), sorted(self.config.usable_episodes(name)), name)
            self.assertFalse(set(train + val) & set(entry["excluded"]), name)

    def test_excluded_episodes_are_in_no_split(self) -> None:
        for name, entry in self.manifest["collections"].items():
            for excluded in entry["excluded"]:
                self.assertNotIn(excluded, entry["train"] + entry["val"], f"{name} episode {excluded}")
        placement = self.manifest["collections"]["G1_Dex3_ObjectPlacement_Dataset"]
        bread = self.manifest["collections"]["G1_Dex3_ToastedBread_Dataset"]
        self.assertNotIn(0, placement["train"] + placement["val"])
        self.assertNotIn(396, bread["train"] + bread["val"])
        self.assertEqual(len(placement["train"]) + len(placement["val"]), 209)
        self.assertEqual(len(bread["train"]) + len(bread["val"]), 417)

    def test_seed_and_instructions_are_recorded(self) -> None:
        self.assertEqual(self.manifest["seed"], self.config.split_seed)
        self.assertEqual(self.manifest["task_instructions"], self.config.tasks)
        self.assertEqual(self.manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(self.manifest["config"]["sha256"], self.config.sha256)

    def test_validation_normalizes_with_the_train_statistics(self) -> None:
        self.assertEqual(self.manifest["normalization_stats"], "train/meta/stats_psi0.json")

    def test_building_twice_is_identical(self) -> None:
        self.assertEqual(build_split(self.config), self.manifest)

    def test_one_collection_does_not_disturb_another(self) -> None:
        """Per-collection seeds keep the split stable when the corpus grows."""
        trimmed = build_split(self.config)
        for name, entry in trimmed["collections"].items():
            original = self.manifest["collections"][name]
            self.assertEqual(entry["val"], original["val"], name)

    def test_manifest_round_trips_through_disk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_split_manifest(self.config, Path(directory) / "split_manifest.json")
            self.assertEqual(load_split_manifest(path), self.manifest)

    def test_wrong_seed_is_refused(self) -> None:
        from humanoid_lab.datasets.psi0_dex3.split import assert_manifest_matches_config

        tampered = dict(self.manifest)
        tampered["seed"] = self.manifest["seed"] + 1
        with self.assertRaisesRegex(ValueError, "seed"):
            assert_manifest_matches_config(tampered, self.config)

    def test_other_contract_is_refused(self) -> None:
        from humanoid_lab.datasets.psi0_dex3.split import assert_manifest_matches_config

        tampered = dict(self.manifest)
        tampered["config"] = {"path": "other.yaml", "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "rebuild the split"):
            assert_manifest_matches_config(tampered, self.config)


if __name__ == "__main__":
    unittest.main()
