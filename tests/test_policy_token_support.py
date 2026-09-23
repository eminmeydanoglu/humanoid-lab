"""The token-support measurement: quantization identity, distances, calibration, verdicts.

The measurement only means something if the FSQ snap used here is the client's own snap,
if the nearest-neighbour metrics really return the nearest neighbour under the metric they
name, if the leave-one-episode-out calibration never lets a frame see its own episode, and
if the A/B/C/D rule reads the numbers it says it reads.  These tests pin those four things
on synthetic input, with no dataset, no telemetry file and no rollout.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "policy-token-support.py"

sys.path.insert(0, str(ROOT / "src"))
from humanoid_lab.psi0_bridge.actions import fsq_quantize as client_fsq  # noqa: E402


def load_module():
    specification = importlib.util.spec_from_file_location("policy_token_support", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


SUPPORT = load_module()


def synthetic_cloud() -> tuple[np.ndarray, np.ndarray]:
    """Two episodes of eight frames each, on the FSQ grid, with one channel constant.

    The two episodes occupy disjoint parts of the range on purpose: any zero distance in a
    leave-one-episode-out result then means the own episode leaked into the reference.
    """
    base = np.zeros((16, 64), dtype=np.float32)
    base[:8, 0] = np.linspace(-0.5, -0.25, 8)
    base[8:, 0] = np.linspace(0.25, 0.5, 8)
    base[:, 1] = 0.0625
    labels = np.array([0] * 8 + [1] * 8, dtype=int)
    return base, labels


class QuantizationTest(unittest.TestCase):
    def test_snap_is_the_clients_own_snap(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(0.0, 0.4, size=(97, 64)).astype(np.float32)
        values[0] = np.linspace(-1.5, 1.5, 64)  # deliberately outside the FSQ range
        np.testing.assert_array_equal(SUPPORT.fsq_quantize(values), client_fsq(values))

    def test_snap_lands_on_the_grid_and_clips(self) -> None:
        values = np.array([[-0.7, 0.7, 0.0312, -0.0312]], dtype=np.float32)
        snapped = SUPPORT.fsq_quantize(values)
        np.testing.assert_array_equal(snapped, np.array([[-0.625, 0.625, 0.0, 0.0]], dtype=np.float32))

    def test_grid_helpers_agree_with_the_values(self) -> None:
        values = np.array([[0.0, 0.0625, -0.0625, 0.625, -0.625]], dtype=np.float32)
        levels = SUPPORT.grid_level(values)
        np.testing.assert_array_equal(levels, np.array([[10, 11, 9, 20, 0]], dtype=np.int16))
        np.testing.assert_allclose(SUPPORT.grid_residual(values), 0.0, atol=1e-9)

    def test_residual_never_exceeds_half_a_step(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.uniform(-1.0, 1.0, size=(64, 64))
        self.assertLessEqual(float(SUPPORT.grid_residual(values).max()), SUPPORT.FSQ_STEP / 2 + 1e-12)

    def test_quantization_report_matches_a_hand_computation(self) -> None:
        raw = np.zeros((4, 64), dtype=np.float32)
        raw[:, 0] = np.array([0.01, 0.02, -0.02, 0.03], dtype=np.float32)
        report = SUPPORT.quantization_report(raw)
        # Every channel snaps by its distance to 0.0 except the ones that move a whole step.
        self.assertEqual(report["raw_tokens"], 4)
        self.assertAlmostEqual(report["nearest_grid_distance_levels"]["mean"],
                               float(np.abs(raw).mean() * 1.0) / SUPPORT.FSQ_STEP, places=9)
        self.assertTrue(0.0 <= report["raw_outside_fsq_range_frac"] <= 1.0)


class DistanceTest(unittest.TestCase):
    def test_value_l1_is_the_manhattan_nearest_neighbour(self) -> None:
        cloud = np.zeros((3, 64), dtype=np.float32)
        cloud[1, 0] = 0.5
        cloud[2, 0] = -0.5
        query = np.zeros((2, 64), dtype=np.float32)
        query[0, 0] = 0.4
        query[1, 0] = -0.45
        np.testing.assert_allclose(SUPPORT._token_distance(query, cloud, "value_l1"), [0.1, 0.05], atol=1e-6)

    def test_grid_and_hamming_are_measured_in_levels(self) -> None:
        cloud = np.zeros((1, 64), dtype=np.float32)
        query = np.zeros((1, 64), dtype=np.float32)
        query[0, 0] = 0.1875  # three levels away
        query[0, 1] = 0.0625  # one level away
        self.assertAlmostEqual(float(SUPPORT._token_distance(query, cloud, "grid_l1")[0]), 4.0)
        self.assertAlmostEqual(float(SUPPORT._token_distance(query, cloud, "hamming")[0]), 2.0)

    def test_leave_one_episode_out_never_uses_the_own_episode(self) -> None:
        cloud, labels = synthetic_cloud()
        duplicate = cloud.copy()
        duplicate[1] = cloud[0]  # a near-duplicate inside episode 0
        distance = SUPPORT.loeo_cloud_distance(duplicate, labels, "value_l1")
        for episode in np.unique(labels):
            self.assertTrue(np.all(distance[labels == episode] > 0.0))

    def test_envelope_leave_one_episode_out_is_bounded(self) -> None:
        cloud, labels = synthetic_cloud()
        rates = SUPPORT.envelope_loeo(cloud, labels)
        self.assertEqual(rates["frames"], cloud.shape[0])
        self.assertTrue(0.0 <= rates["frame_all_channels_inside_p1_p99_frac"] <= 1.0)


class CalibrationTest(unittest.TestCase):
    def test_balanced_cloud_is_episode_balanced_and_seeded(self) -> None:
        values = np.arange(64 * 20, dtype=np.float32).reshape(20, 64)
        owners = np.array([0] * 12 + [1] * 8)
        first, first_labels = SUPPORT.balanced_cloud(values, owners, [0, 1], 5, 7)
        second, _ = SUPPORT.balanced_cloud(values, owners, [0, 1], 5, 7)
        self.assertEqual(first.shape, (10, 64))
        self.assertEqual(sorted(np.unique(first_labels).tolist()), [0, 1])
        np.testing.assert_array_equal(first, second)

    def test_envelope_bounds_are_the_weighted_quantiles(self) -> None:
        pool = np.zeros((100, 64), dtype=np.float32)
        pool[0, 0] = 1.0
        weights = np.full(100, 1.0 / 100)
        quantiles = SUPPORT.weighted_quantiles(pool, weights, [0.01, 0.99])
        self.assertLessEqual(quantiles[1][0], 1.0)

    def test_step_metrics_count_repeated_tokens(self) -> None:
        tokens = np.zeros((4, 64), dtype=np.float32)
        tokens[1, 0] = 0.0625
        tokens[2, 0] = 0.0625
        tokens[3, 0] = 0.125
        steps = SUPPORT.step_metrics(tokens)
        self.assertEqual(steps["steps"], 3)
        self.assertAlmostEqual(steps["hamming"]["mean"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(steps["repeated_token_frac"], 1.0 / 3.0, places=6)


class OccupancyTest(unittest.TestCase):
    def test_undemonstrated_levels_are_flagged(self) -> None:
        levels_used = [np.zeros(SUPPORT.GRID_LEVELS, dtype=bool) for _ in range(64)]
        for channel in range(64):
            levels_used[channel][10] = True  # only the zero level is demonstrated
        tokens = np.full((2, 64), 0.0, dtype=np.float32)
        tokens[1, 3] = 0.0625  # a level the demonstration never uses
        report = SUPPORT.occupancy(tokens, {"levels_used": levels_used})
        self.assertEqual(report["frames"], 2)
        self.assertAlmostEqual(report["frames_with_any_undemonstrated_level"], 0.5, places=6)
        self.assertAlmostEqual(report["channel_samples_outside_demonstrated_levels"],
                               1.0 / (2 * 64), places=6)


class JoinTest(unittest.TestCase):
    def test_join_matches_published_targets_to_applied_tokens(self) -> None:
        raw = np.zeros((3, 64), dtype=np.float32)
        raw[:, 0] = [0.01, 0.02, 0.03]
        telemetry = {
            "raw": raw,
            "raw_index": np.array([0, 1, 2]),
            "raw_published": np.array([True, False, True]),
            "applied": SUPPORT.fsq_quantize(raw[[0, 2]]),
            "applied_index": np.array([0, 2]),
            "applied_time": np.array([0, 1], dtype=np.int64),
        }
        join = SUPPORT.join_frames(telemetry)
        self.assertEqual(join["published_targets"], 2)
        self.assertEqual(join["published_without_applied"], 0)
        self.assertEqual(join["raw_applied_index_match_frac"], 1.0)
        self.assertEqual(join["applied_equals_fsq_raw_frac"], 1.0)
        self.assertEqual(join["max_abs_applied_minus_fsq_raw"], 0.0)

    def test_a_published_target_without_an_applied_record_is_reported(self) -> None:
        raw = np.zeros((2, 64), dtype=np.float32)
        telemetry = {
            "raw": raw, "raw_index": np.array([0, 1]), "raw_published": np.array([True, True]),
            "applied": SUPPORT.fsq_quantize(raw[[]]).reshape(0, 64),
            "applied_index": np.zeros(0, dtype=np.int64), "applied_time": np.zeros(0, dtype=np.int64),
        }
        join = SUPPORT.join_frames(telemetry)
        self.assertEqual(join["published_without_applied"], 2)
        self.assertIsNone(join["applied_equals_fsq_raw_frac"])

    def test_join_to_rows_takes_the_nearest_tracking_row(self) -> None:
        rows = np.array([0, 100, 200], dtype=np.int64)
        applied = np.array([10, 160, 400], dtype=np.int64)
        position, offset = SUPPORT.join_to_rows(applied, rows)
        np.testing.assert_array_equal(position, [0, 2, 2])
        np.testing.assert_allclose(offset, [10e-9, -40e-9, 200e-9], atol=1e-12)


class DecisionTest(unittest.TestCase):
    """The A/B/C/D rule, exercised on the held-out reference it is defined against."""

    def calibration(self) -> dict:
        return {
            "thresholds": {"q99": {"value_l1": 4.0, "grid_l1": 20.0}},
            "envelope_loeo": {"frame_all_channels_inside_p1_p99_frac": 0.60},
            "heldout": {
                "value_l1": {"median": 1.2, "p95": 1.9, "p99": 2.4},
                "grid_l1": {"median": 19.0, "p95": 30.0, "p99": 38.0},
                "envelope_frame_outside_frac": 0.05,
                "envelope_channel_outside_frac": 0.002,
                "hull_frame_outside_frac": 0.01,
                "steps": {"value_l1": {"median": 0.75, "p25": 0.25, "p75": 2.0},
                          "grid_l1": {"median": 12.0, "p25": 4.0, "p75": 32.0},
                          "repeated_token_frac": 0.04},
            },
        }

    def decide(self, *, value_l1: float, grid_l1: float, envelope: float, step: float,
               permuted_ratio: float = 1.5, wrong_task_ratio: float = 1.2) -> dict:
        corpus = {"tokens": 100,
                  "windows": {"value_l1_median": value_l1, "grid_l1_median": grid_l1,
                              "envelope_frame_outside_frac": envelope,
                              "envelope_channel_outside_frac": 0.01,
                              "hull_frame_outside_frac": 0.1,
                              "step_value_l1_median": step, "step_grid_l1_median": step / 0.0625},
                  "repeated_token_frac": 0.0,
                  "controls": {"permuted_median_ratio": permuted_ratio,
                               "wrong_task_median_ratio": wrong_task_ratio}}
        return SUPPORT.decide("psi0", corpus, self.calibration())

    def test_inside_every_family_is_A(self) -> None:
        verdict = self.decide(value_l1=1.5, grid_l1=25.0, envelope=0.08, step=0.75)
        self.assertEqual(verdict["decision"], "A")
        self.assertTrue(verdict["controls_discriminative"])

    def test_outside_every_family_is_B(self) -> None:
        verdict = self.decide(value_l1=9.0, grid_l1=60.0, envelope=0.9, step=9.0)
        self.assertEqual(verdict["decision"], "B")
        self.assertEqual(verdict["families"]["envelope"]["verdict"], "out")

    def test_mixed_evidence_is_C(self) -> None:
        # Distance family inside, envelope and dynamics outside.
        verdict = self.decide(value_l1=1.5, grid_l1=25.0, envelope=0.9, step=9.0)
        self.assertEqual(verdict["decision"], "C")

    def test_borderline_distance_is_not_A(self) -> None:
        verdict = self.decide(value_l1=2.2, grid_l1=35.0, envelope=0.08, step=0.75)
        self.assertEqual(verdict["families"]["distance"]["value_l1"]["verdict"], "borderline")
        self.assertEqual(verdict["decision"], "C")

    def test_in_support_without_separating_controls_is_not_A(self) -> None:
        verdict = self.decide(value_l1=1.5, grid_l1=25.0, envelope=0.08, step=0.75,
                              permuted_ratio=1.0, wrong_task_ratio=1.0)
        self.assertEqual(verdict["decision"], "C")
        self.assertFalse(verdict["controls_discriminative"])

    def test_missing_telemetry_is_D(self) -> None:
        self.assertEqual(SUPPORT.decide("groot", {"tokens": 0}, self.calibration())["decision"], "D")

    def test_hull_ratio_is_undefined_when_the_held_out_rate_is_zero(self) -> None:
        calibration = self.calibration()
        calibration["heldout"]["hull_frame_outside_frac"] = 0.0
        corpus = {"tokens": 100,
                  "windows": {"value_l1_median": 1.5, "grid_l1_median": 25.0,
                              "envelope_frame_outside_frac": 0.08,
                              "envelope_channel_outside_frac": 0.01,
                              "hull_frame_outside_frac": 0.02,
                              "step_value_l1_median": 0.75, "step_grid_l1_median": 12.0},
                  "repeated_token_frac": 0.0,
                  "controls": {"permuted_median_ratio": 3.0, "wrong_task_median_ratio": 1.3}}
        envelope = SUPPORT.decide("psi0", corpus, calibration)["families"]["envelope"]
        self.assertIsNone(envelope["hull_ratio"])
        self.assertEqual(envelope["rollout_hull_outside_frac"], 0.02)


class FigureInputTest(unittest.TestCase):
    def test_meta_blob_round_trips(self) -> None:
        meta = {"tag": "psi0-rollout-1", "model": "psi0", "rollout": 1, "onset_seconds_after_start": 0.055}
        blob = np.array(json.dumps(meta))
        self.assertEqual(json.loads(str(blob.tolist()))["rollout"], 1)


if __name__ == "__main__":
    unittest.main()
