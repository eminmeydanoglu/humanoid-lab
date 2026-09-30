"""Conversion: tail trim, interpolation, corruption repair, rebuilt hand action.

Pure tests on synthetic source episodes; no data root, no parquet, no video. The
hand orders are checked against the values a channel carries, not against a
formula, so a swapped middle/index block cannot pass.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from humanoid_lab.datasets.groot_sonic.contract import (  # noqa: E402
    CANONICAL_ACTION_HAND_NAMES,
    CANONICAL_STATE_NAMES,
    FPS,
    HORIZON,
    MEASURED_STATE_FIELDS,
    STATE_BLOCKS,
    TAIL_ROWS,
    corpus_hand_permutation,
    load_config,
    source_action_hand_permutation,
    source_state_permutation,
    standing_lower_body,
)
from humanoid_lab.datasets.groot_sonic.convert import (  # noqa: E402
    ConversionError,
    RawEpisode,
    SonicEpisode,
    assert_action_hands_clean,
    assert_measured_finite,
    build_state,
    convert_episode,
    corrupt_hand_columns,
    linear_resample,
    repair_measured_hands,
    retained_rows,
)
from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import UNITREE_ACTION_NAMES  # noqa: E402

SOURCE_FRAMES = 100
CORPUS_FRAMES = 167
ROWS = CORPUS_FRAMES - TAIL_ROWS


def raw_timeline(frames: int = SOURCE_FRAMES) -> np.ndarray:
    return np.arange(frames, dtype=np.float64) / 30.0


def corpus_timeline(frames: int = CORPUS_FRAMES) -> np.ndarray:
    return np.arange(frames, dtype=np.float64) / float(FPS)


def smooth_source(frames: int = SOURCE_FRAMES) -> np.ndarray:
    """A finite, strictly moving [T, 28] source block far inside the repair bound."""
    ramp = np.linspace(0.2, 1.1, frames)
    channels = np.arange(1, 29, dtype=np.float64) / 28.0
    return np.outer(ramp, channels)


def make_raw(
    collection: str = "G1_Dex3_Synthetic_Dataset",
    episode: int = 0,
    frames: int = SOURCE_FRAMES,
    state: np.ndarray | None = None,
    action: np.ndarray | None = None,
) -> RawEpisode:
    block = smooth_source(frames)
    return RawEpisode(
        collection=collection,
        episode_index=episode,
        state=block if state is None else state,
        action=block if action is None else action,
        timestamp=raw_timeline(frames),
        state_names=UNITREE_ACTION_NAMES,
        action_names=UNITREE_ACTION_NAMES,
        data_file="data/chunk-000/file-000.parquet",
        video_file=Path("/nonexistent/source.mp4"),
        video_start_s=0.0,
        video_stop_s=frames / 30.0,
    )


def make_corpus(
    raw: RawEpisode,
    frames: int = CORPUS_FRAMES,
    *,
    corrupt_left_block: bool = False,
    tail_rows: int = TAIL_ROWS,
    interior_invalid: tuple[int, ...] = (),
) -> SonicEpisode:
    """A corpus built the way the real one was: same-row desired action, 50 Hz.

    The hand block is written in the *corpus* order -- the left hand in the source
    motor order -- which is exactly the trap the conversion must not copy.
    """
    timestamps = corpus_timeline(frames)
    token = np.arange(frames * 64, dtype=np.float32).reshape(frames, 64) / 977.0
    hands = np.stack(
        [np.interp(timestamps, raw.timestamp, raw.action[:, column]) for column in range(14, 28)],
        axis=1,
    ).astype(np.float32)
    if corrupt_left_block:
        hands[:, :7] = hands[:, [0, 1, 2, 5, 6, 3, 4]]
    mask = np.ones(frames, dtype=bool)
    mask[frames - tail_rows :] = False
    for index in interior_invalid:
        mask[index] = False
    return SonicEpisode(
        path=Path("/nonexistent/action.npz"),
        action=np.concatenate([token, hands], axis=1),
        timestamp=timestamps,
        training_valid_mask=mask,
    )


class RetainedRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_drops_exactly_the_clamped_tail(self) -> None:
        mask = np.ones(CORPUS_FRAMES, dtype=bool)
        mask[-TAIL_ROWS:] = False
        self.assertEqual(retained_rows(mask, self.config, "label"), CORPUS_FRAMES - TAIL_ROWS)

    def test_refuses_a_shorter_tail(self) -> None:
        mask = np.ones(CORPUS_FRAMES, dtype=bool)
        mask[-44:] = False
        with self.assertRaisesRegex(ConversionError, "trailing 45"):
            retained_rows(mask, self.config, "label")

    def test_refuses_an_interior_invalid_row(self) -> None:
        mask = np.ones(CORPUS_FRAMES, dtype=bool)
        mask[-TAIL_ROWS:] = False
        mask[10] = False
        with self.assertRaisesRegex(ConversionError, "trailing 45"):
            retained_rows(mask, self.config, "label")

    def test_refuses_an_episode_shorter_than_the_horizon(self) -> None:
        mask = np.ones(HORIZON + TAIL_ROWS - 1, dtype=bool)
        mask[-TAIL_ROWS:] = False
        with self.assertRaisesRegex(ConversionError, "horizon"):
            retained_rows(mask, self.config, "label")


class LinearResampleTest(unittest.TestCase):
    def test_values_at_the_nodes_are_preserved(self) -> None:
        source = raw_timeline(10)
        values = np.arange(20, dtype=np.float64).reshape(10, 2)
        result = linear_resample(source, values, source, "label")
        np.testing.assert_array_equal(result, values)

    def test_midpoints_are_the_mean_of_their_neighbours(self) -> None:
        source = np.array([0.0, 1.0])
        values = np.array([[0.0, 2.0], [1.0, 4.0]])
        result = linear_resample(source, values, np.array([0.25, 0.75]), "label")
        np.testing.assert_allclose(result, [[0.25, 2.5], [0.75, 3.5]], rtol=0, atol=1e-12)

    def test_a_target_outside_the_source_span_is_refused(self) -> None:
        with self.assertRaisesRegex(ConversionError, "outside the source span"):
            linear_resample(np.array([0.0, 1.0]), np.zeros((2, 1)), np.array([1.5]), "label")

    def test_a_non_monotone_source_timeline_is_refused(self) -> None:
        with self.assertRaisesRegex(ConversionError, "increasing"):
            linear_resample(np.array([0.0, 0.5, 0.4]), np.zeros((3, 1)), np.array([0.2]), "label")

    def test_a_non_finite_source_value_is_refused(self) -> None:
        values = np.array([[0.0], [np.nan]])
        with self.assertRaisesRegex(ConversionError, "NaN or Inf"):
            linear_resample(np.array([0.0, 1.0]), values, np.array([0.5]), "label")


class RepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_gross_values_are_interpolated_per_channel(self) -> None:
        state = smooth_source()
        clean = state.copy()
        # kLeftHandThumb0 is source column 14; kRightHandMiddle1 is column 27.
        state[10, 14] = 5.0
        state[20, 27] = np.nan
        repaired, ledger = repair_measured_hands(state, UNITREE_ACTION_NAMES, self.config, "label")
        expected = np.interp(10, np.array([9, 11]), clean[[9, 11], 14])
        self.assertAlmostEqual(repaired[10, 14], expected, places=12)
        self.assertEqual(ledger.channels, {"left_hand_thumb_0_joint": 1, "right_hand_middle_1_joint": 1})
        self.assertEqual(ledger.samples, 2)
        self.assertEqual(ledger.threshold_rad, 3.0)
        self.assertTrue(np.isfinite(repaired).all())
        np.testing.assert_array_equal(repaired[:, :14], clean[:, :14])
        np.testing.assert_array_equal(repaired[:, 15:27], clean[:, 15:27])
        np.testing.assert_equal(repaired[[9, 11], 14], clean[[9, 11], 14])

    def test_a_clean_episode_reports_nothing(self) -> None:
        repaired, ledger = repair_measured_hands(smooth_source(), UNITREE_ACTION_NAMES, self.config, "label")
        self.assertEqual(ledger.channels, {})
        self.assertEqual(ledger.samples, 0)
        np.testing.assert_array_equal(repaired, smooth_source())

    def test_values_inside_the_bound_are_kept_verbatim(self) -> None:
        """No clipping: a legal but large reading survives unchanged."""
        state = smooth_source()
        state[5, 18] = 2.9
        repaired, ledger = repair_measured_hands(state, UNITREE_ACTION_NAMES, self.config, "label")
        self.assertEqual(ledger.channels, {})
        self.assertEqual(repaired[5, 18], 2.9)

    def test_a_boundary_corruption_is_refused(self) -> None:
        state = smooth_source()
        state[0, 14] = 4.0
        with self.assertRaisesRegex(ConversionError, "boundary"):
            repair_measured_hands(state, UNITREE_ACTION_NAMES, self.config, "label")

    def test_a_channel_without_valid_samples_is_refused(self) -> None:
        state = smooth_source()
        state[:, 20] = 9.0
        with self.assertRaisesRegex(ConversionError, "no valid measured sample"):
            repair_measured_hands(state, UNITREE_ACTION_NAMES, self.config, "label")

    def test_too_much_corruption_in_one_channel_is_refused(self) -> None:
        state = smooth_source()
        state[[5, 15, 25, 35, 45, 55], 14] = 4.0
        with self.assertRaisesRegex(ConversionError, "above the declared"):
            repair_measured_hands(state, UNITREE_ACTION_NAMES, self.config, "label")

    def test_a_run_at_the_gap_bound_is_repaired(self) -> None:
        """The audited corpus maximum (51 source frames) is repaired, not refused."""
        state = smooth_source(frames=1200)
        clean = state.copy()
        state[20:71, 14] = np.nan
        repaired, ledger = repair_measured_hands(state, UNITREE_ACTION_NAMES, self.config, "label")
        self.assertEqual(self.config.repair_max_gap_frames, 51)
        self.assertEqual(ledger.channels, {"left_hand_thumb_0_joint": 51})
        np.testing.assert_allclose(repaired[20:71, 14], clean[20:71, 14], rtol=0, atol=1e-12)

    def test_a_long_invalid_run_is_refused(self) -> None:
        state = smooth_source(frames=1200)
        state[20:72, 14] = np.nan
        with self.assertRaisesRegex(ConversionError, "gap bound"):
            repair_measured_hands(state, UNITREE_ACTION_NAMES, self.config, "label")

    def test_corruption_scan_only_covers_the_hands(self) -> None:
        state = smooth_source()
        state[3, 2] = 6.0  # an arm channel: not in the repair scope
        self.assertEqual(corrupt_hand_columns(state, UNITREE_ACTION_NAMES, 3.0), {})

    def test_a_corrupt_action_hand_is_refused_not_repaired(self) -> None:
        action = smooth_source()
        action[7, 16] = 4.0
        with self.assertRaisesRegex(ConversionError, "action targets are declared clean"):
            assert_action_hands_clean(action, UNITREE_ACTION_NAMES, self.config, "label")

    def test_a_non_finite_arm_is_refused(self) -> None:
        state = smooth_source()
        state[9, 1] = np.nan
        with self.assertRaisesRegex(ConversionError, "non-finite"):
            assert_measured_finite(state[:, :14], UNITREE_ACTION_NAMES[:14], "label")


class BuildStateTest(unittest.TestCase):
    def test_lower_body_is_the_standing_proxy(self) -> None:
        measured = np.zeros((4, 28))
        state = build_state(measured, tuple(range(28)), 4)
        self.assertEqual(state.shape, (4, 43))
        np.testing.assert_allclose(state[:, :15], np.tile(standing_lower_body(), (4, 1)), rtol=0, atol=1e-6)
        self.assertTrue((state[:, :15] == state[0, :15]).all())

    def test_a_short_permutation_is_refused(self) -> None:
        with self.assertRaisesRegex(ConversionError, "28"):
            build_state(np.zeros((4, 28)), tuple(range(27)), 4)


class ConvertEpisodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.raw = make_raw()
        cls.corpus = make_corpus(cls.raw)
        cls.episode = convert_episode(cls.raw, cls.corpus, cls.config)

    def test_rows_are_the_corpus_rows_minus_the_clamped_tail(self) -> None:
        self.assertEqual(self.episode.frames, ROWS)
        self.assertEqual(self.episode.corpus_rows, CORPUS_FRAMES)
        self.assertEqual(self.episode.trimmed_rows, TAIL_ROWS)
        self.assertEqual(self.episode.frames, len(self.episode.timestamp))
        self.assertEqual(self.episode.state.shape, (ROWS, 43))
        self.assertEqual(self.episode.motion_token.shape, (ROWS, 64))
        self.assertEqual(self.episode.left_hand.shape, (ROWS, 7))
        self.assertEqual(self.episode.right_hand.shape, (ROWS, 7))

    def test_the_timeline_is_the_corpus_timeline(self) -> None:
        np.testing.assert_array_equal(self.episode.timestamp, corpus_timeline()[:ROWS])
        self.assertAlmostEqual(float(self.episode.timestamp[0]), 0.0)

    def test_the_token_is_the_corpus_token_verbatim(self) -> None:
        np.testing.assert_array_equal(self.episode.motion_token, self.corpus.action[:ROWS, :64])

    def test_the_state_is_the_standing_proxy_plus_the_measured_block(self) -> None:
        measured = linear_resample(
            self.raw.timestamp, self.raw.state, self.episode.timestamp, "test measured"
        )[:, list(source_state_permutation(UNITREE_ACTION_NAMES))]
        np.testing.assert_allclose(self.episode.state[:, :15], np.tile(standing_lower_body(), (ROWS, 1)), rtol=0, atol=1e-6)
        np.testing.assert_allclose(self.episode.state[:, 15:], measured, rtol=0, atol=1e-6)

    def test_the_hand_action_is_rebuilt_from_the_raw_desired_action(self) -> None:
        desired = linear_resample(
            self.raw.timestamp,
            self.raw.action[:, list(source_action_hand_permutation(UNITREE_ACTION_NAMES))],
            self.episode.timestamp,
            "test desired",
        )
        np.testing.assert_allclose(self.episode.hand_action, desired, rtol=0, atol=1e-6)
        self.assertLessEqual(self.episode.corpus_hand_max_error, self.config.corpus_hand_tolerance_rad)

    def test_gravity_is_the_synthetic_upright(self) -> None:
        np.testing.assert_array_equal(self.episode.gravity, np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (ROWS, 1)))

    def test_no_repair_is_reported_for_a_clean_source(self) -> None:
        self.assertEqual(self.episode.repair.channels, {})
        self.assertEqual(self.episode.repair.samples, 0)

    def test_the_two_hand_orders_are_applied_independently(self) -> None:
        """Distinct source values must land in distinctly ordered output channels."""
        frames = SOURCE_FRAMES
        block = np.zeros((frames, 28))
        # Source left hand order is thumb0, thumb1, thumb2, middle0, middle1, index0, index1.
        block[:, 14:21] = np.arange(10, 17, dtype=np.float64) / 100.0
        block[:, 21:28] = np.arange(20, 27, dtype=np.float64) / 100.0
        raw = make_raw(state=block, action=block)
        episode = convert_episode(raw, make_corpus(raw), self.config)
        left_slice, right_slice = STATE_BLOCKS["left_hand"], STATE_BLOCKS["right_hand"]
        state_hands = episode.state[
            0, [*range(left_slice.start, left_slice.stop), *range(right_slice.start, right_slice.stop)]
        ].astype(np.float64)
        # State: index, index, middle, middle, thumb, thumb, thumb.
        np.testing.assert_allclose(state_hands[:7], [0.15, 0.16, 0.13, 0.14, 0.10, 0.11, 0.12], rtol=0, atol=1e-6)
        # The source right block is thumb, thumb, thumb, index, index, middle, middle.
        np.testing.assert_allclose(state_hands[7:], [0.23, 0.24, 0.25, 0.26, 0.20, 0.21, 0.22], rtol=0, atol=1e-6)
        # Action: thumb, thumb, thumb, index, index, middle, middle.
        np.testing.assert_allclose(episode.left_hand[0].astype(np.float64), [0.10, 0.11, 0.12, 0.15, 0.16, 0.13, 0.14], rtol=0, atol=1e-6)
        np.testing.assert_allclose(episode.right_hand[0].astype(np.float64), [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26], rtol=0, atol=1e-6)
        self.assertEqual(CANONICAL_ACTION_HAND_NAMES[3], "left_hand_index_0_joint")
        self.assertEqual(CANONICAL_STATE_NAMES.index("left_hand_thumb_0_joint"), left_slice.start + 4)
        self.assertEqual(CANONICAL_STATE_NAMES.index("right_hand_index_0_joint"), right_slice.start)
        self.assertEqual(episode.repair.samples, 0)

    def test_a_corpus_whose_left_block_is_not_permuted_by_name_is_refused(self) -> None:
        """The corpus left hand is in the source motor order; copying it would be wrong."""
        raw = make_raw()
        corpus = make_corpus(raw, corrupt_left_block=True)
        with self.assertRaisesRegex(ConversionError, "name-permuted corpus block"):
            convert_episode(raw, corpus, self.config)

    def test_a_corpus_with_another_tail_length_is_refused(self) -> None:
        raw = make_raw()
        with self.assertRaisesRegex(ConversionError, "invalid"):
            convert_episode(raw, make_corpus(raw, tail_rows=44), self.config)

    def test_a_corpus_with_an_interior_invalid_row_is_refused(self) -> None:
        raw = make_raw()
        with self.assertRaisesRegex(ConversionError, "invalid"):
            convert_episode(raw, make_corpus(raw, interior_invalid=(11,)), self.config)

    def test_a_corpus_that_does_not_fit_the_camera_segment_is_refused(self) -> None:
        raw = make_raw(frames=60)
        with self.assertRaisesRegex(ConversionError, "camera segment"):
            convert_episode(raw, make_corpus(raw), self.config)

    def test_a_corpus_without_enough_rows_is_refused(self) -> None:
        raw = make_raw()
        with self.assertRaisesRegex(ConversionError, "horizon"):
            convert_episode(raw, make_corpus(raw, frames=HORIZON + TAIL_ROWS - 1), self.config)

    def test_corruption_is_repaired_before_the_resampling(self) -> None:
        """A repaired source sample is the interpolation of its *source* neighbours."""
        clean = smooth_source()
        broken = clean.copy()
        broken[10, 14] = 6.0
        raw = make_raw(state=broken)
        episode = convert_episode(raw, make_corpus(raw), self.config)
        self.assertEqual(episode.repair.channels, {"left_hand_thumb_0_joint": 1})
        channel = CANONICAL_STATE_NAMES.index("left_hand_thumb_0_joint")
        repaired_source = clean.copy()
        repaired_source[10, 14] = np.interp(10, np.array([9, 11]), clean[[9, 11], 14])
        expected = np.interp(episode.timestamp, raw.timestamp, repaired_source[:, 14])
        untouched = np.interp(episode.timestamp, raw.timestamp, broken[:, 14])
        np.testing.assert_allclose(episode.state[:, channel].astype(np.float64), expected, rtol=0, atol=1e-5)
        index = int(np.argmin(np.abs(episode.timestamp - 10 / 30.0)))
        self.assertAlmostEqual(float(episode.state[index, channel]), float(expected[index]), places=5)
        self.assertGreater(abs(float(untouched[index] - expected[index])), 1.0)

    def test_corpus_hand_permutation_is_a_real_permutation(self) -> None:
        permutation = corpus_hand_permutation()
        self.assertEqual(sorted(permutation), list(range(14)))
        self.assertEqual(len(MEASURED_STATE_FIELDS), 28)


if __name__ == "__main__":
    unittest.main()
