"""Gate 1: 50->30 Hz mapping, 43D state, split actions and strict validity.

The pure tests are synthetic and need no data root. The end-to-end test converts
real episodes and validates the produced dataset; it skips when the frozen
sources are not mounted.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_lab.datasets.psi0_dex3.contract import (
    MASK_KEY,
    ANCHOR_MASK_KEY,
    CANONICAL_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    ACTION_MODEL_DIM,
    load_config,
    standing_lower_body,
)
from humanoid_lab.datasets.psi0_dex3.convert import (
    RawEpisode,
    anchor_validity,
    assert_mapping,
    build_state,
    convert_episode,
    nearest_indices,
    read_raw_episode,
)
from humanoid_lab.datasets.psi0_dex3.split import build_split
from humanoid_lab.datasets.psi0_dex3.validate import validate_dataset
from humanoid_lab.datasets.psi0_dex3.writer import SplitWriter
from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import UNITREE_ACTION_NAMES

CHUNK = 30


def synthetic_sources(frames: int = 100, source_frames: int = 167) -> tuple[RawEpisode, np.ndarray, np.ndarray, np.ndarray]:
    """A raw episode, its 50 Hz action and a mask whose last 45 frames are clamped."""
    state = np.arange(frames * 28, dtype=np.float32).reshape(frames, 28) / 977.0
    action = np.arange(source_frames * 78, dtype=np.float32).reshape(source_frames, 78) / 1013.0
    valid = np.ones(source_frames, dtype=bool)
    valid[-45:] = False
    raw = RawEpisode(
        collection="synthetic",
        episode_index=0,
        state=state,
        timestamp=np.arange(frames, dtype=np.float64) / 30.0,
        state_names=UNITREE_ACTION_NAMES,
        data_file="data/chunk-000/file-000.parquet",
        video_file=Path("synthetic.mp4"),
        video_start_s=0.0,
        video_stop_s=frames / 30.0,
    )
    return raw, action, np.arange(source_frames, dtype=np.float64) / 50.0, valid


class NearestTimestampTest(unittest.TestCase):
    def test_picks_the_nearest_sample(self) -> None:
        source = np.arange(11, dtype=np.float64) / 50.0
        target = np.array([0.0, 0.021, 0.2])
        self.assertEqual(nearest_indices(source, target).tolist(), [0, 1, 10])

    def test_ties_pick_the_lower_index(self) -> None:
        source = np.array([0.0, 0.02])
        self.assertEqual(nearest_indices(source, np.array([0.01])).tolist(), [0])

    def test_selection_is_monotone(self) -> None:
        source = np.arange(196, dtype=np.float64) / 50.0
        target = np.arange(118, dtype=np.float64) / 30.0
        index = nearest_indices(source, target)
        self.assertTrue((np.diff(index) >= 0).all())

    def test_decreasing_source_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "increasing"):
            nearest_indices(np.array([1.0, 0.5]), np.array([0.6]))


class StrictAnchorValidityTest(unittest.TestCase):
    def test_anchor_requires_the_whole_window(self) -> None:
        valid = np.ones(40, dtype=bool)
        valid[35] = False
        anchor = anchor_validity(valid, 10)
        self.assertEqual(np.flatnonzero(anchor).tolist(), list(range(0, 26)))

    def test_short_episode_has_no_anchor(self) -> None:
        anchor = anchor_validity(np.ones(29, dtype=bool), CHUNK)
        self.assertFalse(anchor.any())
        self.assertEqual(anchor.shape, (29,))

    def test_invalid_tail_removes_anchors(self) -> None:
        valid = np.ones(45, dtype=bool)
        valid[43:] = False
        anchor = anchor_validity(valid, CHUNK)
        self.assertEqual(int(np.flatnonzero(anchor)[-1]), 13)


class BuildStateTest(unittest.TestCase):
    def test_state_is_standing_proxy_plus_measured(self) -> None:
        measured = np.arange(5 * 28, dtype=np.float32).reshape(5, 28) / 100.0
        state = build_state(measured, tuple(range(28)))
        self.assertEqual(state.shape, (5, 43))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_allclose(state[:, :15], np.tile(standing_lower_body(), (5, 1)), rtol=0, atol=1e-6)
        np.testing.assert_allclose(state[:, 15:], measured, rtol=0, atol=1e-6)
        self.assertTrue((state[:, :15] == state[0, :15]).all())

    def test_wrong_source_width_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation.state"):
            build_state(np.zeros((4, 27)), tuple(range(28)))


class ConvertEpisodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.raw, action, timestamp, valid = synthetic_sources()
        cls.episode = convert_episode(cls.raw, action, timestamp, valid, cls.config)

    def test_timeline_is_the_source_30hz_grid(self) -> None:
        np.testing.assert_allclose(
            self.episode.timestamp, np.arange(self.raw.frames) / 30.0, rtol=0, atol=1e-12
        )

    def test_body_token_and_hand_action_are_separate_fields(self) -> None:
        self.assertEqual(self.episode.body_token.shape, (self.raw.frames, 64))
        self.assertEqual(self.episode.hand_action.shape, (self.raw.frames, 14))
        self.assertEqual(self.config.body_token_field, "action.body_token_v1_1")
        self.assertEqual(self.config.action_field, "action")

    def test_actions_are_the_selected_source_rows(self) -> None:
        action = np.arange(167 * 78, dtype=np.float32).reshape(167, 78) / 1013.0
        selected = action[self.episode.source_index]
        np.testing.assert_array_equal(self.episode.body_token, selected[:, :64])
        np.testing.assert_array_equal(self.episode.hand_action, selected[:, 64:78])

    def test_state_channels(self) -> None:
        self.assertEqual(self.episode.state.shape, (self.raw.frames, 43))
        np.testing.assert_allclose(self.episode.state[:, 15:], self.raw.state, rtol=0, atol=1e-6)
        self.assertTrue((self.episode.state[:, :15] == self.episode.state[0, :15]).all())

    def test_action_mask_is_80d_and_matches_frame_validity(self) -> None:
        mask = self.episode.action_mask
        self.assertEqual(mask.shape, (self.raw.frames, ACTION_MODEL_DIM))
        self.assertTrue(np.isin(mask, (0.0, 1.0)).all())
        np.testing.assert_array_equal(
            mask[:, :78],
            np.broadcast_to(np.asarray(self.episode.training_valid_mask, dtype=np.float32)[:, None], (self.raw.frames, 78)),
        )
        self.assertTrue((mask[:, 78:] == 0).all())

    def test_invalid_tail_never_reaches_a_valid_anchor(self) -> None:
        anchors = np.flatnonzero(self.episode.anchor_valid)
        self.assertGreater(len(anchors), 0)
        for anchor in anchors:
            self.assertTrue(self.episode.training_valid_mask[anchor : anchor + CHUNK].all(), int(anchor))
            self.assertTrue((self.episode.action_mask[anchor : anchor + CHUNK, 0] == 1).all())

    def test_valid_tail_frames_are_not_supervision(self) -> None:
        """Frames whose window is clamped may be valid targets but not anchors."""
        anchors = np.flatnonzero(self.episode.anchor_valid)
        self.assertLess(int(anchors[-1]), int(np.flatnonzero(self.episode.training_valid_mask)[-1]))

    def test_changing_the_invalid_tail_does_not_change_a_valid_anchor(self) -> None:
        raw, action, timestamp, valid = synthetic_sources()
        first = convert_episode(raw, action, timestamp, valid, self.config)
        poisoned = action.copy()
        poisoned[~valid] = 1e9
        second = convert_episode(raw, poisoned, timestamp, valid, self.config)
        last = int(np.flatnonzero(first.anchor_valid)[-1]) + CHUNK
        np.testing.assert_array_equal(first.body_token[:last], second.body_token[:last])
        np.testing.assert_array_equal(first.hand_action[:last], second.hand_action[:last])
        np.testing.assert_array_equal(first.anchor_valid, second.anchor_valid)

    def test_changing_the_invalid_tail_does_change_the_masked_tail(self) -> None:
        """The test above is not vacuous: the tail really is part of the file."""
        raw, action, timestamp, valid = synthetic_sources()
        first = convert_episode(raw, action, timestamp, valid, self.config)
        poisoned = action.copy()
        poisoned[~valid] = 1e9
        second = convert_episode(raw, poisoned, timestamp, valid, self.config)
        self.assertFalse(np.array_equal(first.body_token[-1], second.body_token[-1]))

    def test_mismatched_source_width_is_refused(self) -> None:
        raw, action, timestamp, valid = synthetic_sources()
        with self.assertRaisesRegex(ValueError, r"\[T, 78\]"):
            convert_episode(raw, action[:, :64], timestamp, valid, self.config)

    def test_non_uniform_source_timeline_is_refused(self) -> None:
        raw, action, timestamp, valid = synthetic_sources()
        broken = RawEpisode(**{**raw.__dict__, "timestamp": raw.timestamp * 1.01})
        with self.assertRaisesRegex(ValueError, "uniform"):
            convert_episode(broken, action, timestamp, valid, self.config)


class MappingBoundTest(unittest.TestCase):
    """An episode long enough that a coarse tail sample sits outside the windows."""

    def setUp(self) -> None:
        self.config = load_config()
        self.frames = CHUNK + 10
        self.index = np.arange(self.frames)
        self.anchor = np.zeros(self.frames, dtype=bool)
        self.anchor[0] = True  # its 30-target window is frames 0..29

    def test_tail_error_beyond_the_last_anchor_is_allowed(self) -> None:
        error = np.zeros(self.frames)
        error[CHUNK + 5] = 0.02
        assert_mapping(error, self.index, self.anchor, CHUNK, self.config, "label")

    def test_tail_error_above_the_tail_bound_is_refused(self) -> None:
        error = np.zeros(self.frames)
        error[CHUNK + 5] = 0.5
        with self.assertRaisesRegex(ValueError, "tail bound"):
            assert_mapping(error, self.index, self.anchor, CHUNK, self.config, "label")

    def test_coarse_sample_inside_a_valid_anchor_is_refused(self) -> None:
        error = np.zeros(self.frames)
        error[0] = 0.0133  # inside the first anchor's 30-frame window
        with self.assertRaisesRegex(ValueError, "supervised action target"):
            assert_mapping(error, self.index, self.anchor, CHUNK, self.config, "label")

    def test_non_monotone_selection_is_refused(self) -> None:
        index = self.index.copy()
        index[5] = 2  # jumps backwards
        with self.assertRaisesRegex(ValueError, "monotone"):
            assert_mapping(np.zeros(self.frames), index, self.anchor, CHUNK, self.config, "label")

    def test_repeated_sample_is_still_monotone(self) -> None:
        """A repeated sample is legitimate; only a backwards jump is not."""
        index = self.index.copy()
        index[5] = 4
        assert_mapping(np.zeros(self.frames), index, self.anchor, CHUNK, self.config, "label")


@unittest.skipUnless(
    Path("/data/datasets/unitree-sonic-v1.1-78d").is_dir()
    and Path("/data/datasets/first_tur_ham/unitree-g1-dex3").is_dir()
    and shutil.which("ffmpeg") is not None,
    "the frozen raw and 50 Hz SONIC corpora are not mounted",
)
class MiniDatasetEndToEndTest(unittest.TestCase):
    """Convert one episode per task and validate the produced LeRobot dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.manifest = build_split(cls.config)
        cls.temporary = tempfile.TemporaryDirectory(prefix="psi0-dex3-mini-")
        cls.root = Path(cls.temporary.name)
        for split in ("train", "val"):
            writer = SplitWriter(cls.root / split, cls.config, split)
            for name in cls.config.collection_names:
                episode = cls.manifest["collections"][name][split][0]
                raw = read_raw_episode(cls.config, name, episode)
                path = cls.config.sonic_root / name / "episodes" / f"episode_{episode:06d}" / "action.npz"
                with np.load(path, allow_pickle=False) as payload:
                    writer.add(
                        convert_episode(
                            raw,
                            payload["action"],
                            payload["timestamp"],
                            payload["training_valid_mask"],
                            cls.config,
                        )
                    )
            writer.finalize()
        cls.report = validate_dataset(cls.config, cls.root, cls.manifest, check_lerobot=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_dataset_validates(self) -> None:
        failed = [entry for entry in self.report["failed"]]
        self.assertEqual(failed, [], f"{len(failed)} failing checks, first: {failed[:1]}")
        self.assertEqual(self.report["status"], "PASS")

    def test_every_task_is_covered_by_both_splits(self) -> None:
        for split in ("train", "val"):
            self.assertEqual(
                len(list((self.root / split / "data" / "chunk-000").glob("episode_*.parquet"))),
                len(self.config.collections),
            )

    def test_lerobot_opens_both_splits(self) -> None:
        names = {entry["check"] for entry in self.report["checks"] if entry["status"] == "PASS"}
        self.assertIn("train.lerobot_load", names)
        self.assertIn("val.lerobot_load", names)

    def test_episode_and_frame_count_match_the_selection(self) -> None:
        self.assertEqual(self.report["episodes_checked"], 2 * len(self.config.collections))
        self.assertGreater(self.report["frames_checked"], 0)

    def test_declared_state_names_match_the_canonical_order(self) -> None:
        import json

        info = json.loads((self.root / "train/meta/info.json").read_text(encoding="utf-8"))
        self.assertEqual(tuple(info["features"]["observation.state"]["names"]), CANONICAL_STATE_NAMES)
        self.assertEqual(tuple(info["features"]["action"]["names"]), CANONICAL_HAND_NAMES)
        self.assertEqual(info["features"][MASK_KEY]["shape"], [ACTION_MODEL_DIM])
        self.assertIn(ANCHOR_MASK_KEY, info["features"])


if __name__ == "__main__":
    unittest.main()
