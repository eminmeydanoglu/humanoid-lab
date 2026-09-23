"""Physical-bound guard: non-physical measured samples never reach the normaliser.

Background. The raw Unitree Dex3 collections contain isolated ``observation.state``
spikes (up to 3363 rad) on four right-hand channels.  Psi0 normalises states by
``min``/``max``, so those spikes collapsed those channels to ~0.02% of their
usable range in the shipped pack.  These tests pin the three guards that now
prevent it, plus the property that a legitimate overshoot of a channel's own
nominal limit is *not* treated as corruption.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_lab.datasets.psi0_dex3.contract import (
    CANONICAL_STATE_NAMES,
    PHYSICAL_HAND_SAMPLE_BOUND_RAD,
    SOURCE_STATE_DIM,
    _physical_bound,
)
from humanoid_lab.datasets.psi0_dex3.convert import (
    MAX_REPAIRABLE_SAMPLES_PER_EPISODE,
    RawEpisode,
    repair_nonphysical_samples,
)
from humanoid_lab.datasets.psi0_dex3.writer import verify_stats_are_physical

#: The real spike that produced the shipped pack's -3363 rad bound.
REAL_SPIKE = -3363.48291015625


def source_names() -> tuple[str, ...]:
    """The raw 28D Unitree field order, read from the contract's own map."""
    from humanoid_lab.datasets.psi0_dex3.contract import SOURCE_ARM_HAND_FIELDS

    return SOURCE_ARM_HAND_FIELDS


class PhysicalBoundTest(unittest.TestCase):
    def test_bound_is_inside_the_pinned_model_limits(self) -> None:
        """A value the model cannot reach must never pass the guard."""
        worst = _physical_bound()
        self.assertLessEqual(PHYSICAL_HAND_SAMPLE_BOUND_RAD, worst)
        self.assertGreater(PHYSICAL_HAND_SAMPLE_BOUND_RAD, 0.0)

    def test_bound_is_the_robot_wide_limit_not_one_channel(self) -> None:
        """The bound must exceed every individual joint's limit.

        A per-channel guard would flag normal teleoperation: measured Dex3
        fingers legitimately reach 2.93 rad against a 1.5708 rad URDF limit.
        """
        limits = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/datasets/sonic/g1_joint_limits.json").read_text()
        )["joints"]
        self.assertGreaterEqual(PHYSICAL_HAND_SAMPLE_BOUND_RAD, 2.9293)
        self.assertIn("right_hand_middle_0_joint", limits)


class RepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.names = source_names()
        self.clean = np.zeros((40, SOURCE_STATE_DIM), dtype=np.float64)
        self.clean[:, :] = np.linspace(-1.0, 1.0, SOURCE_STATE_DIM)

    def test_clean_episode_is_returned_untouched(self) -> None:
        repaired, repair = repair_nonphysical_samples(self.clean, self.names)
        np.testing.assert_array_equal(repaired, self.clean)
        self.assertFalse(repair.repaired)
        self.assertEqual(repair.invalid_source_samples, 0)
        self.assertEqual(repair.policy, "none")

    def test_real_spike_is_interpolated_between_valid_neighbours(self) -> None:
        state = self.clean.copy()
        channel = self.names.index("kRightHandMiddle0")
        state[20, channel] = REAL_SPIKE
        repaired, repair = repair_nonphysical_samples(state, self.names)
        self.assertTrue(repair.repaired)
        self.assertEqual(repair.invalid_source_samples, 1)
        self.assertEqual(repair.channels, {"kRightHandMiddle0": 1})
        self.assertLess(abs(repaired[20, channel]), PHYSICAL_HAND_SAMPLE_BOUND_RAD)
        # Interpolation over the two neighbouring valid samples.
        expected = 0.5 * (self.clean[19, channel] + self.clean[21, channel])
        self.assertAlmostEqual(repaired[20, channel], expected, places=9)
        # Every other channel and frame is bit-identical.
        keep = np.ones(state.shape[1], dtype=bool)
        keep[channel] = False
        np.testing.assert_array_equal(repaired[:, keep], state[:, keep])

    def test_leading_and_trailing_runs_use_the_nearest_valid_value(self) -> None:
        state = self.clean.copy()
        channel = self.names.index("kRightHandIndex0")
        state[0:3, channel] = 500.0
        state[-2:, channel] = -900.0
        repaired, repair = repair_nonphysical_samples(state, self.names)
        self.assertEqual(repair.invalid_source_samples, 5)
        self.assertAlmostEqual(repaired[0, channel], self.clean[3, channel], places=9)
        self.assertAlmostEqual(repaired[2, channel], self.clean[3, channel], places=9)
        self.assertAlmostEqual(repaired[-1, channel], self.clean[-3, channel], places=9)

    def test_repair_record_round_trips_for_the_manifest(self) -> None:
        state = self.clean.copy()
        channel = self.names.index("kRightHandMiddle1")
        state[10, channel] = REAL_SPIKE
        state[11, channel] = REAL_SPIKE
        _, repair = repair_nonphysical_samples(state, self.names)
        record = repair.as_dict()
        self.assertEqual(record["policy"], "linear_interpolation_over_valid_source_samples")
        self.assertEqual(record["threshold_rad"], PHYSICAL_HAND_SAMPLE_BOUND_RAD)
        self.assertEqual(record["channels"], {"kRightHandMiddle1": 2})
        self.assertEqual(record["invalid_source_samples"], 2)
        json.dumps(record)  # must be JSON-serialisable for episodes.jsonl

    def test_fully_invalid_channel_is_refused(self) -> None:
        state = self.clean.copy()
        state[:, self.names.index("kRightHandIndex1")] = 99.0
        with self.assertRaises(ValueError) as ctx:
            repair_nonphysical_samples(state, self.names)
        self.assertIn("no valid sample", str(ctx.exception))

    def test_corpus_scale_defect_is_refused_not_imputed(self) -> None:
        """A whole-corpus spike burst must fail closed, not be mass-imputed."""
        frames = MAX_REPAIRABLE_SAMPLES_PER_EPISODE * 2
        state = np.zeros((frames, SOURCE_STATE_DIM), dtype=np.float64)
        state[:, 24:28] = 100.0
        # Keep one valid sample so the refusal comes from the volume check and
        # not from a fully-invalid channel.
        state[0, 24:28] = 0.0
        with self.assertRaises(ValueError) as ctx:
            repair_nonphysical_samples(state, self.names)
        self.assertIn("refusing to impute", str(ctx.exception))
        self.assertIn(str(MAX_REPAIRABLE_SAMPLES_PER_EPISODE), str(ctx.exception))

    def test_legitimate_overshoot_of_a_nominal_limit_is_kept(self) -> None:
        """2.93 rad is real measured data on a 1.5708 rad joint; keep it."""
        state = self.clean.copy()
        channel = self.names.index("kRightHandMiddle0")
        state[5, channel] = 2.9293
        repaired, repair = repair_nonphysical_samples(state, self.names)
        self.assertFalse(repair.repaired)
        self.assertEqual(repaired[5, channel], 2.9293)

    def test_name_count_must_match_columns(self) -> None:
        with self.assertRaises(ValueError):
            repair_nonphysical_samples(self.clean, self.names[:-1])


class StatsGuardTest(unittest.TestCase):
    def _config(self):
        from humanoid_lab.datasets.psi0_dex3.contract import load_config

        return load_config()

    def _stats(self, low: float, high: float) -> dict:
        width = len(CANONICAL_STATE_NAMES)
        block = {
            "min": [-1.0] * width,
            "max": [1.0] * width,
            "mean": [0.0] * width,
            "std": [0.1] * width,
            "count": [10],
        }
        # Channels 15: are the measured ones; corrupt the last measured channel.
        block["min"][42] = low
        block["max"][42] = high
        return {"observation.state": block}

    def test_physical_stats_pass(self) -> None:
        config = self._config()
        report = verify_stats_are_physical(self._stats(-2.1, 2.1), config)
        self.assertEqual(report["bound_rad"], PHYSICAL_HAND_SAMPLE_BOUND_RAD)
        self.assertLessEqual(report["state_abs_max_rad"], PHYSICAL_HAND_SAMPLE_BOUND_RAD)

    def test_corrupted_stats_are_refused_and_name_the_channel(self) -> None:
        config = self._config()
        with self.assertRaises(ValueError) as ctx:
            verify_stats_are_physical(self._stats(REAL_SPIKE, 2.1), config)
        message = str(ctx.exception)
        self.assertIn("right_hand_middle_1_joint", message)
        self.assertIn("collapse", message)

    def test_synthetic_legs_waist_are_not_flagged(self) -> None:
        """Channels 0:15 are a constant standing pose, never a spike source."""
        config = self._config()
        stats = self._stats(-2.0, 2.0)
        stats["observation.state"]["min"][0] = -3.0
        stats["observation.state"]["max"][0] = -3.0
        # A constant channel is zero-range, not out of bound.
        report = verify_stats_are_physical(stats, config)
        self.assertLessEqual(report["state_abs_max_rad"], PHYSICAL_HAND_SAMPLE_BOUND_RAD)

    def test_wrong_width_is_refused(self) -> None:
        config = self._config()
        stats = self._stats(-1.0, 1.0)
        stats["observation.state"]["min"] = stats["observation.state"]["min"][:-1]
        with self.assertRaises(ValueError):
            verify_stats_are_physical(stats, config)

    def test_missing_state_block_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            verify_stats_are_physical({"action": {}}, self._config())


class RawEpisodeProvenanceTest(unittest.TestCase):
    def test_repair_defaults_to_no_repair_and_carries_the_source_slice(self) -> None:
        episode = RawEpisode(
            collection="C",
            episode_index=1,
            state=np.zeros((3, SOURCE_STATE_DIM), dtype=np.float32),
            timestamp=np.arange(3, dtype=np.float64) / 30.0,
            state_names=source_names(),
            data_file="data/chunk-000/file-000.parquet",
            video_file=Path("v.mp4"),
            video_start_s=0.0,
            video_stop_s=1.0,
        )
        self.assertFalse(episode.repair.repaired)
        self.assertIsNone(episode.data_path)
        self.assertEqual(episode.repair.as_dict()["invalid_source_samples"], 0)


class ShippedPackRegressionTest(unittest.TestCase):
    """The guard must flag the shipped pack and pass a repaired one.

    Skipped when the pack or the unpinned source root is absent: the repo may be
    inspected outside its data container.
    """

    PACK = Path("data/datasets/psi0-unitree-dex3-sonic-v1")

    def test_shipped_stats_are_flagged_when_present(self) -> None:
        stats_path = self.PACK / "train/meta/stats_psi0.json"
        if not stats_path.is_file():
            self.skipTest("pack not mounted")
        from humanoid_lab.datasets.psi0_dex3.contract import load_config

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        # The pack on disk predates the guard and is known to be corrupted; if it
        # ever passes, it was rebuilt and this regression test should be updated.
        with self.assertRaises(ValueError) as ctx:
            verify_stats_are_physical(stats, load_config())
        self.assertIn("bound", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
