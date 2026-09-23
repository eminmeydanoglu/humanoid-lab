"""The rollout analysis: onset detection, the video frame map, and failure signals.

The onset rule is the part a reviewer has to be able to trust, because the clip
that gets watched is cut at it: these tests pin the sustained-window behaviour,
the baseline scaling, the frame-map conversion and the measured fall/stack
signals on synthetic input, with no session and no GPU.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze-blockstacking-rollout.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_blockstacking_rollout", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZER = load_analyzer()


def tracking_row(*, sim_s: float, wall_ns: int, arm: float, hand: float, root_z: float = 0.79) -> dict:
    """One 50 Hz tracking row with two arm joints, two hand joints and a root."""
    return {
        "sim_s": sim_s,
        "wall_time_ns": wall_ns,
        "body_joints": None,
        "body_measured_velocity": [0.0, arm, 0.0, 0.0, arm],
        "left_hand_measured_velocity": [hand, hand],
        "right_hand_measured_velocity": [hand, hand],
        "left_hand_measured": [0.0, 0.0],
        "right_hand_measured": [0.0, 0.0],
        "root_position": [0.0, 0.0, root_z],
    }


COLUMNS = {
    "body_joints": ["waist_yaw_joint", "left_shoulder_pitch_joint", "left_knee_joint",
                    "right_elbow_joint", "right_wrist_roll_joint"],
    "left_hand_joints": ["left_hand_index_0_joint", "left_hand_thumb_0_joint"],
    "right_hand_joints": ["right_hand_index_0_joint", "right_hand_thumb_0_joint"],
}


class VelocitySeriesTest(unittest.TestCase):
    def test_only_arm_and_hand_joints_enter_the_activity_norm(self) -> None:
        # The knee carries the same velocity as the arm here; a series that took
        # every body joint would report the legs as arm motion.
        row = {"sim_s": 1.0, "body_measured_velocity": [0.0, 3.0, 3.0, 4.0, 0.0],
               "left_hand_measured_velocity": [0.0, 0.0],
               "right_hand_measured_velocity": [0.0, 0.0]}
        series = ANALYZER.velocity_series([row], COLUMNS)
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0][1], 5.0)

    def test_a_missing_velocity_block_is_skipped(self) -> None:
        series = ANALYZER.velocity_series([{"sim_s": 0.0}], COLUMNS)
        self.assertEqual(series, [])


class SustainedOnsetTest(unittest.TestCase):
    def test_a_sustained_step_is_an_onset(self) -> None:
        series = [(index * 0.02, 0.0) for index in range(10)]
        series += [(0.2 + index * 0.02, 1.0) for index in range(30)]
        onset = ANALYZER.sustained_onset(series, threshold=0.5, min_sustain_s=0.3)
        self.assertAlmostEqual(onset, 0.2)

    def test_a_short_spike_is_not_an_onset(self) -> None:
        series = [(index * 0.02, 0.0) for index in range(10)]
        series += [(0.2, 5.0), (0.22, 5.0), (0.24, 0.0)]
        self.assertIsNone(ANALYZER.sustained_onset(series, threshold=0.5, min_sustain_s=0.3))

    def test_a_series_that_never_exceeds_the_threshold_has_no_onset(self) -> None:
        series = [(index * 0.02, 0.1) for index in range(50)]
        self.assertIsNone(ANALYZER.sustained_onset(series, threshold=0.5, min_sustain_s=0.3))

    def test_an_empty_series_has_no_onset(self) -> None:
        self.assertIsNone(ANALYZER.sustained_onset([], threshold=0.5, min_sustain_s=0.3))


class QuietBaselineWindowTest(unittest.TestCase):
    def score(self, segment):
        return [row["body_measured_velocity"][1] for row in segment]

    def test_the_quietest_window_is_chosen_not_the_first(self) -> None:
        # 10 windows of 1 s: the first five are noisy (the settle moving), the
        # last five are a still hold.  The quiet one must win.
        rows = []
        for index in range(50):
            noisy = index < 25
            rows.append({"wall_time_ns": index * int(1e9 / 5), "sim_s": index * 0.2,
                         "body_measured_velocity": [0.0, 5.0 if noisy else 0.0]})
        low, high = ANALYZER.quiet_baseline_window(
            rows, start_ns=50 * int(1e9 / 5), reset_ns=0, length_s=1.0, score=self.score
        )
        # The property that matters is that the chosen window is a still one,
        # not that it starts at a particular sample: re-scoring it must be quiet.
        values = [row["body_measured_velocity"][1] for row in rows
                  if low <= row["wall_time_ns"] < high]
        self.assertTrue(values)
        self.assertEqual(sorted(values)[len(values) // 2], 0.0)
        self.assertGreaterEqual(low, 20 * int(1e9 / 5))

    def test_a_window_shorter_than_the_requested_length_is_used_as_is(self) -> None:
        rows = [{"wall_time_ns": 0, "sim_s": 0.0, "body_measured_velocity": [0.0, 0.0]}]
        self.assertEqual(
            ANALYZER.quiet_baseline_window(rows, start_ns=int(0.4e9), reset_ns=0, length_s=2.0, score=self.score),
            (0, int(0.4e9)),
        )


class VideoTimeForWallTest(unittest.TestCase):
    # 25 fps map that ran slower than real time: frame n is *not* at n/25 wall
    # seconds, which is exactly the case a subtraction of wall clocks gets wrong.
    FRAMES = (
        {"frame": 0, "wall_time_ns": 1_000_000_000},
        {"frame": 1, "wall_time_ns": 1_080_000_000},
        {"frame": 2, "wall_time_ns": 1_160_000_000},
        {"frame": 3, "wall_time_ns": 1_240_000_000},
    )

    def test_the_first_frame_at_or_after_a_wall_time_is_selected(self) -> None:
        self.assertAlmostEqual(
            ANALYZER.video_time_for_wall(self.FRAMES, 1.05, 25.0), 1 / 25.0
        )
        self.assertAlmostEqual(
            ANALYZER.video_time_for_wall(self.FRAMES, 1.20, 25.0), 3 / 25.0
        )

    def test_the_last_frame_at_or_before_a_wall_time_is_selected_for_the_end(self) -> None:
        self.assertAlmostEqual(
            ANALYZER.video_time_for_wall(self.FRAMES, 1.13, 25.0, last=True), 1 / 25.0
        )

    def test_a_time_outside_the_map_has_no_frame(self) -> None:
        self.assertIsNone(ANALYZER.video_time_for_wall(self.FRAMES, 2.0, 25.0))
        self.assertIsNone(ANALYZER.video_time_for_wall(self.FRAMES, None, 25.0))
        self.assertIsNone(ANALYZER.video_time_for_wall(self.FRAMES, 1.05, 0.0))


class VideoTimeIsTheFrameIndexOverFpsTest(unittest.TestCase):
    def test_a_slow_render_loop_does_not_compress_the_video_time(self) -> None:
        # One frame every 0.08 s of wall clock at 25 fps: the 9th frame is 0.64 s
        # into the file even though it was recorded 0.72 s after the first.
        frames = [{"frame": index, "wall_time_ns": 1_000_000_000 + index * 80_000_000}
                  for index in range(10)]
        self.assertAlmostEqual(
            ANALYZER.video_time_for_wall(frames, 1.64, 25.0), 8 / 25.0
        )


class FailureSignalTest(unittest.TestCase):
    def test_the_support_band_releasing_is_not_reported_as_a_fall(self) -> None:
        # The start-up band holds the robot at ~0.96 m and lets go when the
        # policy takes over; the resulting drop to standing height is not a fall.
        rows = [tracking_row(sim_s=index * 0.02, wall_ns=index, arm=0.0, hand=0.0, root_z=0.96)
                for index in range(10)]
        rows += [tracking_row(sim_s=(10 + index) * 0.02, wall_ns=10 + index, arm=0.0, hand=0.0,
                              root_z=0.77) for index in range(90)]
        fall = ANALYZER.detect_fall(rows, rows[:10])
        self.assertFalse(fall["detected"])
        self.assertAlmostEqual(fall["band_hold_root_z_m"], 0.96, places=3)
        self.assertAlmostEqual(fall["band_release_drop_m"], 0.19, places=3)

    def test_a_sustained_low_pelvis_is_a_fall(self) -> None:
        rows = [tracking_row(sim_s=index * 0.02, wall_ns=index, arm=0.0, hand=0.0, root_z=0.77)
                for index in range(10)]
        rows += [tracking_row(sim_s=(10 + index) * 0.02, wall_ns=10 + index, arm=0.0, hand=0.0,
                              root_z=0.30) for index in range(60)]
        fall = ANALYZER.detect_fall(rows, rows[:5])
        self.assertTrue(fall["detected"])
        self.assertIsNotNone(fall["sim_s"])

    def test_a_brief_dip_below_the_threshold_is_not_a_fall(self) -> None:
        rows = [tracking_row(sim_s=index * 0.02, wall_ns=index, arm=0.0, hand=0.0, root_z=0.77)
                for index in range(10)]
        rows += [tracking_row(sim_s=(10 + index) * 0.02, wall_ns=10 + index, arm=0.0, hand=0.0,
                              root_z=0.30 if index < 5 else 0.77) for index in range(30)]
        fall = ANALYZER.detect_fall(rows, rows[:5])
        self.assertFalse(fall["detected"])

    def test_too_few_samples_report_unknown_rather_than_ok(self) -> None:
        fall = ANALYZER.detect_fall([], [])
        self.assertIsNone(fall["detected"])

    def test_a_cube_held_above_the_table_is_measured_as_lifted(self) -> None:
        baseline = [{"wall_time_ns": 0, "scene": {"live_worktop_height_m": 0.8358,
                    "cubes": {"red": {"live_center_z_m": 0.8608}}}}]
        run = [{"wall_time_ns": 10,
                "scene": {"live_worktop_height_m": 0.8358,
                          "cubes": {"red": {"live_center_z_m": 0.9108,
                                            "live_center_xyz_m": [0.5, 0.1, 0.9108],
                                            "displacement_xy_m": 0.05}}}}]
        stack = ANALYZER.stack_state(run, baseline)
        self.assertTrue(stack["any_cube_lifted"])
        self.assertEqual(stack["highest_cube"], "red")
        self.assertAlmostEqual(stack["cube_lift_above_surface_m"]["red"], 0.05, places=4)

    def test_a_cube_left_on_the_table_is_not_lifted(self) -> None:
        baseline = [{"wall_time_ns": 0, "scene": {"live_worktop_height_m": 0.8358,
                    "cubes": {"red": {"live_center_z_m": 0.8608}}}}]
        run = [{"wall_time_ns": 10,
                "scene": {"live_worktop_height_m": 0.8358,
                          "cubes": {"red": {"live_center_z_m": 0.8608}}}}]
        stack = ANALYZER.stack_state(run, baseline)
        self.assertFalse(stack["any_cube_lifted"])


if __name__ == "__main__":
    unittest.main()
