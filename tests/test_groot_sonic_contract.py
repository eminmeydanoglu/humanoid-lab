"""The GR00T Unitree Dex3 / SONIC contract: layout, hand orders, modality, split.

These are pure tests: they need no data root and no parquet. The two hand
vocabularies are the point of the file -- the state order, the official actuated
order and the corpus motor order are three different orders, and every mapping
is name-based and fail-closed.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from humanoid_lab.datasets.groot_sonic.contract import (  # noqa: E402
    ACTION_HAND_ORDER,
    CANONICAL_ACTION_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    DEFAULT_CONFIG_PATH,
    EMBODIMENT_TAG,
    FPS,
    GRAVITY_FIELD,
    HORIZON,
    LEFT_HAND_FIELD,
    MEASURED_STATE_FIELDS,
    MODEL_STATE_DIM,
    MOTION_TOKEN_FIELD,
    PROJECTED_GRAVITY,
    RIGHT_HAND_FIELD,
    STATE_BLOCKS,
    STATE_DIM,
    STATE_HAND_ORDER,
    STATE_MODEL_KEYS,
    STATE_STORAGE_KEYS,
    TAIL_ROWS,
    assert_source_names,
    corpus_hand_permutation,
    load_config,
    official_stats_command,
    source_action_hand_permutation,
    source_state_permutation,
    standing_lower_body,
)
from humanoid_lab.datasets.groot_sonic.modality import (  # noqa: E402
    ModalityError,
    assert_modality_payload,
    modality_payload,
)
from humanoid_lab.datasets.groot_sonic.split import (  # noqa: E402
    SplitError,
    assert_instructions_match_psi0,
    assert_manifest_matches_config,
    load_split_manifest,
    selection,
)
from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import UNITREE_ACTION_NAMES  # noqa: E402

#: The raw 28D layout, restated so a change in the pinned order is visible here.
#: Stored order is the official exporter's: left arm, left hand, right arm, right
#: hand, with the left hand block reordered from the source's thumb-first order
#: into the state's index, middle, thumb order.
EXPECTED_STATE_PERMUTATION = (
    0, 1, 2, 3, 4, 5, 6,
    19, 20, 17, 18, 14, 15, 16,
    7, 8, 9, 10, 11, 12, 13,
    24, 25, 26, 27, 21, 22, 23,
)
EXPECTED_ACTION_PERMUTATION = (14, 15, 16, 19, 20, 17, 18, 21, 22, 23, 24, 25, 26, 27)


def source_matrix(frames: int = 4) -> np.ndarray:
    """A measured block whose every column carries its own index."""
    return np.tile(np.arange(28, dtype=np.float64), (frames, 1))


class StateLayoutTest(unittest.TestCase):
    def test_official_43d_order(self) -> None:
        self.assertEqual(len(CANONICAL_STATE_NAMES), STATE_DIM)
        self.assertEqual(CANONICAL_STATE_NAMES[0], "left_hip_pitch_joint")
        self.assertEqual(CANONICAL_STATE_NAMES[15], "left_shoulder_pitch_joint")
        self.assertEqual(CANONICAL_STATE_NAMES[22], "left_hand_index_0_joint")
        self.assertEqual(CANONICAL_STATE_NAMES[29], "right_shoulder_pitch_joint")
        self.assertEqual(CANONICAL_STATE_NAMES[36], "right_hand_index_0_joint")
        self.assertEqual(
            {key: (block.start, block.stop) for key, block in STATE_BLOCKS.items()},
            {
                "left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15), "left_arm": (15, 22),
                "left_hand": (22, 29), "right_arm": (29, 36), "right_hand": (36, 43),
            },
        )
        self.assertEqual(STATE_STORAGE_KEYS, tuple(STATE_BLOCKS))
        self.assertEqual(
            STATE_STORAGE_KEYS, ("left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand")
        )
        # The registered concatenation order is not the stored order.
        self.assertEqual(STATE_MODEL_KEYS[:7], ("left_leg", "right_leg", "waist", "left_arm", "right_arm",
                                                "left_hand", "right_hand"))
        self.assertNotEqual(STATE_MODEL_KEYS[:7], STATE_STORAGE_KEYS)

    def test_state_hands_are_index_middle_thumb_on_both_sides(self) -> None:
        for side in ("left", "right"):
            block = CANONICAL_STATE_NAMES[STATE_BLOCKS[f"{side}_hand"]]
            self.assertEqual(
                block,
                tuple(f"{side}_hand_{stem}" for stem in ("index_0_joint", "index_1_joint", "middle_0_joint",
                                                        "middle_1_joint", "thumb_0_joint", "thumb_1_joint",
                                                        "thumb_2_joint")),
            )

    def test_action_hands_are_thumb_index_middle_on_both_sides(self) -> None:
        for side in ("left", "right"):
            self.assertEqual(
                ACTION_HAND_ORDER[side],
                ("thumb_0_joint", "thumb_1_joint", "thumb_2_joint", "index_0_joint", "index_1_joint",
                 "middle_0_joint", "middle_1_joint"),
            )
        self.assertEqual(len(CANONICAL_ACTION_HAND_NAMES), 14)
        self.assertEqual(CANONICAL_ACTION_HAND_NAMES[0], "left_hand_thumb_0_joint")
        self.assertEqual(CANONICAL_ACTION_HAND_NAMES[7], "right_hand_thumb_0_joint")

    def test_the_two_hand_vocabularies_differ(self) -> None:
        self.assertNotEqual(STATE_HAND_ORDER["left"], ACTION_HAND_ORDER["left"])
        self.assertNotEqual(STATE_HAND_ORDER["right"], ACTION_HAND_ORDER["right"])

    def test_standing_lower_body_is_the_declared_15d_proxy(self) -> None:
        lower = standing_lower_body()
        self.assertEqual(lower.shape, (15,))
        self.assertEqual(lower[0], lower[6])  # both hips pitch, symmetric stance
        self.assertEqual(lower[12:].tolist(), [0.0, 0.0, 0.0])


class NameMappingTest(unittest.TestCase):
    def test_source_permutations_are_the_declared_ones(self) -> None:
        self.assertEqual(source_state_permutation(UNITREE_ACTION_NAMES), EXPECTED_STATE_PERMUTATION)
        self.assertEqual(source_action_hand_permutation(UNITREE_ACTION_NAMES), EXPECTED_ACTION_PERMUTATION)

    def test_mapping_by_name_moves_the_declared_channels(self) -> None:
        matrix = source_matrix()
        mapped = matrix[:, list(source_state_permutation(UNITREE_ACTION_NAMES))]
        self.assertEqual(mapped[0].tolist(), list(EXPECTED_STATE_PERMUTATION))
        hands = matrix[:, list(source_action_hand_permutation(UNITREE_ACTION_NAMES))]
        self.assertEqual(hands[0].tolist(), list(EXPECTED_ACTION_PERMUTATION))

    def test_left_and_right_source_blocks_are_asymmetric(self) -> None:
        permutation = source_action_hand_permutation(UNITREE_ACTION_NAMES)
        # The right block is already actuated order; the left block is not.
        self.assertEqual(permutation[7:], tuple(range(21, 28)))
        self.assertNotEqual(permutation[:7], tuple(range(14, 21)))

    def test_corpus_hand_permutation_reorders_the_left_block_only(self) -> None:
        permutation = corpus_hand_permutation()
        self.assertEqual(permutation, (0, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12, 13))
        self.assertNotEqual(permutation[:7], tuple(range(7)))
        self.assertEqual(permutation[7:], tuple(range(7, 14)))

    def test_measured_fields_are_the_declared_source_channels(self) -> None:
        self.assertEqual(len(MEASURED_STATE_FIELDS), 28)
        self.assertEqual(MEASURED_STATE_FIELDS[0], "kLeftShoulderPitch")
        self.assertEqual(MEASURED_STATE_FIELDS[7], "kLeftHandIndex0")
        self.assertEqual(
            MEASURED_STATE_FIELDS[7:14],
            ("kLeftHandIndex0", "kLeftHandIndex1", "kLeftHandMiddle0", "kLeftHandMiddle1",
             "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2"),
        )
        self.assertEqual(MEASURED_STATE_FIELDS[14], "kRightShoulderPitch")
        self.assertEqual(MEASURED_STATE_FIELDS[21], "kRightHandIndex0")
        self.assertEqual(set(MEASURED_STATE_FIELDS) - set(UNITREE_ACTION_NAMES), set())

    def test_fail_closed_on_a_missing_channel(self) -> None:
        broken = UNITREE_ACTION_NAMES[:14] + ("kLeftHandUnknown0",) + UNITREE_ACTION_NAMES[15:]
        with self.assertRaisesRegex(ValueError, "not the declared 28D"):
            source_state_permutation(broken)

    def test_fail_closed_on_a_duplicated_channel(self) -> None:
        broken = UNITREE_ACTION_NAMES[:-1] + (UNITREE_ACTION_NAMES[0],)
        with self.assertRaisesRegex(ValueError, "not the declared 28D"):
            assert_source_names(broken)

    def test_fail_closed_on_a_reordered_release(self) -> None:
        """A release that swaps two channels is refused, not silently remapped."""
        broken = list(UNITREE_ACTION_NAMES)
        broken[0], broken[1] = broken[1], broken[0]
        with self.assertRaises(ValueError):
            source_action_hand_permutation(tuple(broken))


class ModalityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.payload = modality_payload(cls.config)

    def test_state_keys_and_slices(self) -> None:
        state = self.payload["state"]
        self.assertEqual(tuple(state), (
            "left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand",
            "projected_gravity",
        ))
        self.assertEqual(state["left_leg"], {"start": 0, "end": 6})
        self.assertEqual(state["left_arm"], {"start": 15, "end": 22})
        self.assertEqual(state["left_hand"], {"start": 22, "end": 29})
        self.assertEqual(state["right_arm"], {"start": 29, "end": 36})
        self.assertEqual(state["right_hand"], {"start": 36, "end": 43})
        self.assertEqual(
            state["projected_gravity"], {"start": 0, "end": 3, "original_key": GRAVITY_FIELD}
        )
        self.assertEqual(sum(entry["end"] - entry["start"] for entry in state.values()), MODEL_STATE_DIM)

    def test_action_original_keys(self) -> None:
        self.assertEqual(
            self.payload["action"],
            {
                "motion_token": {"start": 0, "end": 64, "original_key": MOTION_TOKEN_FIELD},
                "left_hand_joints": {"start": 0, "end": 7, "original_key": LEFT_HAND_FIELD},
                "right_hand_joints": {"start": 0, "end": 7, "original_key": RIGHT_HAND_FIELD},
            },
        )

    def test_video_and_annotation(self) -> None:
        self.assertEqual(self.payload["video"], {"ego_view": {"original_key": "observation.images.ego_view"}})
        self.assertEqual(
            self.payload["annotation"],
            {"human.task_description": {"original_key": "task_index"}},
        )

    def test_contract_check_accepts_the_payload(self) -> None:
        assert_modality_payload(self.payload, self.config)

    def test_contract_check_rejects_the_arms_then_hands_storage_order(self) -> None:
        """A box that still stores right_arm before left_hand is not this pack."""
        broken = json.loads(json.dumps(self.payload))
        broken["state"]["right_arm"] = {"start": 22, "end": 29}
        broken["state"]["left_hand"] = {"start": 29, "end": 36}
        with self.assertRaisesRegex(ModalityError, "must address"):
            assert_modality_payload(broken, self.config)

    def test_contract_check_rejects_a_missing_original_key(self) -> None:
        broken = json.loads(json.dumps(self.payload))
        del broken["action"]["left_hand_joints"]["original_key"]
        with self.assertRaises(ModalityError):
            assert_modality_payload(broken, self.config)

    def test_contract_check_rejects_a_gravity_slice_of_the_state_column(self) -> None:
        broken = json.loads(json.dumps(self.payload))
        broken["state"]["projected_gravity"] = {"start": 43, "end": 46}
        with self.assertRaises(ModalityError):
            assert_modality_payload(broken, self.config)

    def test_contract_check_rejects_an_extra_state_key(self) -> None:
        broken = json.loads(json.dumps(self.payload))
        broken["state"]["neck"] = {"start": 43, "end": 46}
        with self.assertRaises(ModalityError):
            assert_modality_payload(broken, self.config)


class ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_frozen_numbers(self) -> None:
        self.assertEqual(self.config.dataset_fps, FPS)
        self.assertEqual(self.config.trailing_invalid_rows, TAIL_ROWS)
        self.assertEqual(self.config.horizon, HORIZON)
        self.assertEqual(self.config.repair_invalid_abs_rad, 3.0)
        # The audit-derived bound (longest invalid run in the frozen corpus),
        # not an arbitrary number: see the config's repair comment.
        self.assertEqual(self.config.repair_max_gap_frames, 51)
        self.assertEqual(self.config.camera_target_key, "observation.images.ego_view")
        self.assertEqual((self.config.camera_width, self.config.camera_height), (640, 480))
        self.assertEqual(PROJECTED_GRAVITY, (0.0, 0.0, -1.0))
        self.assertEqual(self.config.raw["state"]["projected_gravity"], [0.0, 0.0, -1.0])

    def test_declared_orders_match_the_module(self) -> None:
        for side in ("left", "right"):
            self.assertEqual(tuple(self.config.raw["state"]["hand_order"][side]), STATE_HAND_ORDER[side])
            self.assertEqual(tuple(self.config.raw["action"]["hand_order"][side]), ACTION_HAND_ORDER[side])

    def test_official_stats_command_points_at_the_pinned_tool(self) -> None:
        command = official_stats_command(self.config, Path("/tmp/pack/train"))
        self.assertEqual(command[1], "/opt/src/isaac-groot/gr00t/data/stats.py")
        self.assertEqual(command[2:4], ["--dataset-path", "/tmp/pack/train"])
        # The pinned tool's tyro CLI requires the tag, and only the enum member
        # name is a valid choice (the lower-case value is rejected).
        self.assertEqual(command[4:], ["--embodiment-tag", EMBODIMENT_TAG])
        self.assertEqual(EMBODIMENT_TAG, "UNITREE_G1_SONIC")

    def test_declared_embodiment_tag_is_the_pinned_member_name(self) -> None:
        self.assertEqual(self.config.stats_embodiment_tag, "UNITREE_G1_SONIC")

    def test_output_root_and_split_manifest(self) -> None:
        self.assertEqual(str(self.config.output_root), "/data/datasets/groot/unitree-dex3-sonic-v1")
        self.assertEqual(
            str(self.config.split_manifest),
            "/data/datasets/psi0-unitree-dex3-sonic-v1/split_manifest.json",
        )
        self.assertEqual(self.config.split_name, "psi0-unitree-dex3-sonic-v1")

    def test_a_relaxed_contract_is_refused(self) -> None:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["trim"]["trailing_invalid_rows"] = 40
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relaxed.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clamped tail"):
                load_config(path)

    def test_a_mismatched_hand_order_is_refused(self) -> None:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["action"]["hand_order"]["left"] = list(reversed(raw["action"]["hand_order"]["left"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "swapped.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "official schema|pinned Dex3"):
                load_config(path)


def synthetic_manifest(config) -> dict:
    """The declared shape of the Psi0 manifest; episodes 1 and 2 are excluded nowhere."""
    collections = {
        collection.name: {
            "total": 3,
            "usable": 3 - len(collection.excluded),
            "excluded": list(collection.excluded),
            "instruction": collection.instruction,
            "train": [1],
            "val": [2],
        }
        for collection in config.collections
    }
    return {
        "schema_version": 1,
        "name": config.split_name,
        "strategy": "episode_level_stratified_by_collection",
        "seed": 1,
        "collections": collections,
    }


class SplitManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_accepts_the_declared_manifest_shape(self) -> None:
        assert_manifest_matches_config(synthetic_manifest(self.config), self.config)

    def test_rejects_another_dataset(self) -> None:
        manifest = synthetic_manifest(self.config)
        manifest["name"] = "psi0-something-else"
        with self.assertRaisesRegex(SplitError, "belongs to"):
            assert_manifest_matches_config(manifest, self.config)

    def test_rejects_a_collection_set_that_disagrees(self) -> None:
        manifest = synthetic_manifest(self.config)
        del manifest["collections"]["G1_Dex3_Pouring_Dataset"]
        with self.assertRaisesRegex(SplitError, "collections disagree"):
            assert_manifest_matches_config(manifest, self.config)

    def test_rejects_a_dropped_exclusion(self) -> None:
        manifest = synthetic_manifest(self.config)
        manifest["collections"]["G1_Dex3_ObjectPlacement_Dataset"]["excluded"] = []
        with self.assertRaisesRegex(SplitError, "excludes"):
            assert_manifest_matches_config(manifest, self.config)

    def test_rejects_an_excluded_episode_inside_the_split(self) -> None:
        manifest = synthetic_manifest(self.config)
        manifest["collections"]["G1_Dex3_ToastedBread_Dataset"]["train"] = [396]
        with self.assertRaisesRegex(SplitError, "excluded episode appears"):
            assert_manifest_matches_config(manifest, self.config)

    def test_rejects_a_drifted_instruction(self) -> None:
        manifest = synthetic_manifest(self.config)
        manifest["collections"]["G1_Dex3_PickApple_Dataset"]["instruction"] = "Do something else."
        with self.assertRaisesRegex(SplitError, "instruction disagrees"):
            assert_manifest_matches_config(manifest, self.config)

    def test_instructions_agree_with_the_psi0_contract(self) -> None:
        assert_instructions_match_psi0(self.config)

    def test_mini_and_full_selection(self) -> None:
        manifest = synthetic_manifest(self.config)
        mini = selection(manifest, self.config, train_per_task=1, val_per_task=1)
        self.assertEqual(mini["train"]["G1_Dex3_PickApple_Dataset"], [1])
        self.assertEqual(mini["val"]["G1_Dex3_PickApple_Dataset"], [2])
        full = selection(manifest, self.config, all_episodes=True)
        self.assertEqual(full["train"]["G1_Dex3_PickApple_Dataset"], [1])
        with self.assertRaisesRegex(SplitError, "cannot be combined"):
            selection(manifest, self.config, train_per_task=1, all_episodes=True)

    def test_load_split_manifest_rejects_a_missing_file(self) -> None:
        with self.assertRaisesRegex(SplitError, "missing"):
            load_split_manifest(Path("/nonexistent/split_manifest.json"))


if __name__ == "__main__":
    unittest.main()
