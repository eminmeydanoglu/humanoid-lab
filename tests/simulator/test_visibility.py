"""Tests for the target visibility metric, its report and the selection table.

These are pure-Python checks: no numpy, torch, Isaac or dataset access, so they
run anywhere:

    PYTHONPATH=src python3 -m unittest tests.simulator.test_visibility
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.visibility import (  # noqa: E402
    DEFAULT_AFTER_SECONDS,
    DEFAULT_BEFORE_SECONDS,
    DEFAULT_THRESHOLD,
    FrameVisibility,
    LeafVisibility,
    VisibilityConfig,
    VisibilityError,
    aggregate_leaf_visibility,
    below_threshold_intervals,
    build_report,
    candidate_csv,
    candidate_row,
    candidate_rows,
    fraction,
    nearest_frame,
    summarize_frames,
    window_bounds,
)
from humanoid_lab.simulators.isaac.visibility_cli import (  # noqa: E402
    load_reports,
    select_rows,
    write_rows,
)

GRASP_TIME = 3.84
RENDER_FPS = 50.0
RENDER_FRAMES = 500
GRASP_FRAME = 192


def frame(render_frame: int, value: float, *, valid: bool = True, exact: bool = True, state: str = "visible") -> FrameVisibility:
    time_seconds = render_frame / RENDER_FPS
    return FrameVisibility(
        render_frame=render_frame,
        source_time_seconds=time_seconds,
        relative_to_grasp_seconds=time_seconds - GRASP_TIME,
        visible_fraction=value if valid else None,
        visible_pixels=int(round(value * 1000)) if valid else 0,
        unoccluded_projected_pixels=1000.0 if valid else 0.0,
        valid=valid,
        exact=exact,
        state=state,
        reason="" if exact else "fully_occluded_leaf",
    )


def series(values: dict[int, float], **kwargs) -> list[FrameVisibility]:
    return [frame(index, value, **kwargs) for index, value in sorted(values.items())]


def report_for(frames: list[FrameVisibility], config: VisibilityConfig | None = None, **kwargs):
    return build_report(
        sequence_key="pickup_table__unit_0__000",
        grasp_source_frame=96,
        grasp_time_seconds=GRASP_TIME,
        source_fps=25.0,
        source_frames=250,
        render_fps=RENDER_FPS,
        render_frames=RENDER_FRAMES,
        config=config or VisibilityConfig(),
        head_camera={"name": "head_camera", "resolution": [640, 480]},
        frames=frames,
        analysis_resolution=[1280, 960],
        **kwargs,
    )


class ConfigTests(unittest.TestCase):
    def test_defaults_are_sane(self) -> None:
        config = VisibilityConfig()
        self.assertEqual(config.threshold, DEFAULT_THRESHOLD)
        self.assertEqual(config.before_seconds, DEFAULT_BEFORE_SECONDS)
        self.assertEqual(config.after_seconds, DEFAULT_AFTER_SECONDS)

    def test_rejects_out_of_range_and_non_finite_values(self) -> None:
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(VisibilityError):
                VisibilityConfig(threshold=threshold)
        for name in ("before_seconds", "after_seconds"):
            for value in (-0.001, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(VisibilityError):
                    VisibilityConfig(**{name: value})

    def test_accepts_boundary_values(self) -> None:
        VisibilityConfig(threshold=0.0)
        VisibilityConfig(threshold=1.0)
        VisibilityConfig(before_seconds=0.0, after_seconds=0.0)


class FractionTests(unittest.TestCase):
    def test_visible_pixels_over_unoccluded_projection(self) -> None:
        self.assertAlmostEqual(fraction(500, 1000.0), 0.5)
        self.assertEqual(fraction(0, 1000.0), 0.0)
        self.assertEqual(fraction(0, 0.0), 0.0)

    def test_one_pixel_rounding_slack_is_clamped_to_one(self) -> None:
        self.assertEqual(fraction(1000, 999.5), 1.0)

    def test_inconsistent_and_negative_counts_are_rejected(self) -> None:
        for visible, projected in ((1000, 900.0), (-1, 100.0), (10, -1.0)):
            with self.subTest(visible=visible, projected=projected), self.assertRaises(VisibilityError):
                fraction(visible, projected)


class WindowTests(unittest.TestCase):
    def test_exact_boundaries_are_closed(self) -> None:
        bounds = window_bounds(GRASP_TIME, VisibilityConfig(before_seconds=1.0, after_seconds=1.0), RENDER_FPS, RENDER_FRAMES)
        # 3.84 s * 50 Hz = 192; +-1 s = 50 frames.
        self.assertEqual((bounds.first, bounds.last), (142, 242))
        self.assertFalse(bounds.clamped)
        self.assertEqual(bounds.frames, 101)

    def test_rounding_is_ceil_start_floor_end_with_float_tolerance(self) -> None:
        # A frame timestamp is included only when it lies inside [start, end].
        bounds = window_bounds(1.0, VisibilityConfig(before_seconds=0.0, after_seconds=0.01), RENDER_FPS, RENDER_FRAMES)
        self.assertEqual((bounds.first, bounds.last), (50, 50))
        bounds = window_bounds(1.0, VisibilityConfig(before_seconds=0.01, after_seconds=0.01), RENDER_FPS, RENDER_FRAMES)
        self.assertEqual((bounds.first, bounds.last), (50, 50))

    def test_window_is_clamped_to_the_clip_and_reports_it(self) -> None:
        bounds = window_bounds(9.9, VisibilityConfig(before_seconds=1.0, after_seconds=1.0), RENDER_FPS, RENDER_FRAMES)
        self.assertEqual(bounds.last, RENDER_FRAMES - 1)
        self.assertTrue(bounds.clamped)

    def test_window_without_a_frame_is_an_error(self) -> None:
        # A 10-frame clip is 0.2 s long; a window at 5.0 s clamps away entirely.
        with self.assertRaises(VisibilityError):
            window_bounds(5.0, VisibilityConfig(before_seconds=0.0, after_seconds=0.0), RENDER_FPS, 10)

    def test_nearest_frame_rounds_to_the_closest_and_clamps(self) -> None:
        self.assertEqual(nearest_frame(GRASP_TIME, RENDER_FPS, RENDER_FRAMES), GRASP_FRAME)
        self.assertEqual(nearest_frame(1.0 / 50.0, RENDER_FPS, RENDER_FRAMES), 1)
        self.assertEqual(nearest_frame(-5.0, RENDER_FPS, RENDER_FRAMES), 0)
        self.assertEqual(nearest_frame(999.0, RENDER_FPS, RENDER_FRAMES), RENDER_FRAMES - 1)


class IntervalTests(unittest.TestCase):
    def test_contiguous_runs_inside_the_window(self) -> None:
        bounds = window_bounds(GRASP_TIME, VisibilityConfig(), RENDER_FPS, RENDER_FRAMES)
        frames = series({140: 0.9, 142: 0.1, 143: 0.2, 144: 0.7, 200: 0.0, 201: 0.0, 203: 0.0, 260: 0.0})
        intervals = below_threshold_intervals(frames, 0.5, bounds.first, bounds.last)
        self.assertEqual([(row["start_render_frame"], row["end_render_frame"]) for row in intervals], [(142, 143), (200, 201), (203, 203)])
        self.assertEqual(intervals[0]["frames"], 2)
        self.assertAlmostEqual(intervals[0]["min_fraction"], 0.1)
        self.assertAlmostEqual(intervals[1]["start_seconds"], 200 / RENDER_FPS - GRASP_TIME, places=5)

    def test_invalid_frames_break_a_run_and_are_not_counted_invisible(self) -> None:
        bounds = window_bounds(GRASP_TIME, VisibilityConfig(), RENDER_FPS, RENDER_FRAMES)
        frames = [frame(150, 0.0), frame(151, 0.0, valid=False), frame(152, 0.0)]
        intervals = below_threshold_intervals(frames, 0.5, bounds.first, bounds.last)
        self.assertEqual([(row["start_render_frame"], row["end_render_frame"]) for row in intervals], [(150, 150), (152, 152)])

    def test_threshold_is_strict(self) -> None:
        bounds = window_bounds(GRASP_TIME, VisibilityConfig(), RENDER_FPS, RENDER_FRAMES)
        frames = series({150: 0.5, 151: 0.499999})
        intervals = below_threshold_intervals(frames, 0.5, bounds.first, bounds.last)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["start_render_frame"], 151)


class SummaryTests(unittest.TestCase):
    def test_min_mean_and_grasp_fraction(self) -> None:
        frames = series({0: 0.0, GRASP_FRAME: 0.4, 200: 0.8, 300: 1.0})
        summary = summarize_frames(
            frames, threshold=0.5, first_frame=142, last_frame=242, grasp_frame=GRASP_FRAME
        )
        self.assertAlmostEqual(summary["clip_min_fraction"], 0.0)
        self.assertAlmostEqual(summary["grasp_fraction"], 0.4)
        # Only frames 192 and 200 were recorded inside the expected 101-frame window.
        self.assertEqual(summary["window_frames"], 101)
        self.assertEqual(summary["window_recorded_frames"], 2)
        self.assertEqual(summary["window_valid_frames"], 2)
        self.assertEqual(summary["window_invalid_frames"], 99)
        self.assertAlmostEqual(summary["window_mean_fraction"], 0.6)
        self.assertFalse(summary["object_below_threshold_for_whole_window"])

    def test_whole_window_booleans_require_every_frame_measured(self) -> None:
        frames = [frame(index, 0.1) for index in range(142, 243)]
        summary = summarize_frames(
            frames, threshold=0.5, first_frame=142, last_frame=242, grasp_frame=GRASP_FRAME
        )
        self.assertTrue(summary["object_below_threshold_for_whole_window"])
        self.assertFalse(summary["object_out_of_frame_for_whole_window"])
        self.assertEqual(summary["window_below_threshold_fraction"], 1.0)
        self.assertEqual(summary["below_threshold_intervals"][0]["frames"], 101)

        with_invalid = list(frames)
        with_invalid[10] = frame(with_invalid[10].render_frame, 0.0, valid=False, exact=False, state="unmeasured")
        summary = summarize_frames(
            with_invalid, threshold=0.5, first_frame=142, last_frame=242, grasp_frame=GRASP_FRAME
        )
        self.assertFalse(summary["object_below_threshold_for_whole_window"])
        self.assertTrue(summary["object_below_threshold_at_any_window_frame"])
        self.assertEqual(summary["window_invalid_frames"], 1)

    def test_out_of_frame_window(self) -> None:
        frames = [frame(index, 0.0, state="outside_head_image") for index in range(142, 243)]
        summary = summarize_frames(
            frames, threshold=0.5, first_frame=142, last_frame=242, grasp_frame=GRASP_FRAME
        )
        self.assertTrue(summary["object_out_of_frame_for_whole_window"])
        self.assertTrue(summary["object_below_threshold_for_whole_window"])
        self.assertEqual(summary["window_min_fraction"], 0.0)

    def test_fully_occluded_is_not_reported_as_out_of_frame(self) -> None:
        frames = [frame(index, 0.0, state="fully_occluded") for index in range(142, 243)]
        summary = summarize_frames(
            frames, threshold=0.5, first_frame=142, last_frame=242, grasp_frame=GRASP_FRAME
        )
        self.assertTrue(summary["object_below_threshold_for_whole_window"])
        self.assertFalse(summary["object_out_of_frame_for_whole_window"])

    def test_missing_rows_make_the_whole_window_incomplete(self) -> None:
        frames = [frame(index, 0.0, state="outside_head_image") for index in range(142, 242)]
        summary = summarize_frames(
            frames, threshold=0.5, first_frame=142, last_frame=242, grasp_frame=GRASP_FRAME
        )
        self.assertEqual(summary["window_recorded_frames"], 100)
        self.assertEqual(summary["window_invalid_frames"], 1)
        self.assertFalse(summary["object_below_threshold_for_whole_window"])
        self.assertFalse(summary["object_out_of_frame_for_whole_window"])

    def test_no_valid_frames_yields_no_booleans_and_no_mean(self) -> None:
        frames = [frame(index, 0.0, valid=False, exact=False, state="unmeasured") for index in range(142, 243)]
        summary = summarize_frames(
            frames, threshold=0.5, first_frame=142, last_frame=242, grasp_frame=GRASP_FRAME
        )
        self.assertFalse(summary["object_below_threshold_for_whole_window"])
        self.assertFalse(summary["object_out_of_frame_for_whole_window"])
        self.assertIsNone(summary["window_mean_fraction"])
        self.assertEqual(summary["window_below_threshold_fraction"], 0.0)

    def test_invalid_frames_report_a_null_fraction(self) -> None:
        from humanoid_lab.simulators.isaac.visibility import unmeasured_frame

        row = unmeasured_frame(10, render_fps=RENDER_FPS, grasp_time_seconds=GRASP_TIME, reason="nothing measured").to_row()
        self.assertIsNone(row["visible_fraction"])
        self.assertIs(False, row["valid"])
        self.assertEqual(row["render_frame"], 10)
        self.assertAlmostEqual(row["source_time_seconds"], 10 / RENDER_FPS)
        # null survives a JSON round trip and comes back as an invalid frame.
        restored = FrameVisibility.from_row(json.loads(json.dumps(row)))
        self.assertIsNone(restored.visible_fraction)
        self.assertFalse(restored.valid)


class ReportTests(unittest.TestCase):
    def _report(self):
        frames = [frame(index, 0.0 if index < 150 else 0.9) for index in range(0, 300)]
        return report_for(frames)

    def test_report_schema_and_json_round_trip(self) -> None:
        report = self._report()
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["sequence_key"], "pickup_table__unit_0__000")
        self.assertEqual(report["result"], "COMPLETED")
        self.assertEqual(report["grasp"], {"source_frame": 96, "source_time_seconds": 3.84, "render_frame": 192})
        self.assertEqual(report["window"]["start_render_frame"], 142)
        self.assertEqual(report["window"]["end_render_frame"], 242)
        self.assertEqual(report["window"]["clamped"], False)
        self.assertTrue(report["metric"]["occlusion_counted_as_not_visible"])
        self.assertTrue(report["metric"]["image_boundary_clipping_counted_as_not_visible"])
        self.assertEqual(report["frames_recorded"], len(report["frames"]))
        row = report["frames"][0]
        for key in ("render_frame", "source_time_seconds", "relative_to_grasp_seconds", "visible_fraction", "valid", "state", "reason"):
            self.assertIn(key, row)
        json.dumps(report)  # must be JSON-serializable as a whole

    def test_grasp_frame_is_inside_the_reported_window(self) -> None:
        report = self._report()
        self.assertLessEqual(report["window"]["start_render_frame"], report["grasp"]["render_frame"])
        self.assertLess(report["grasp"]["render_frame"], report["window"]["end_render_frame"])
        relative = next(
            row["relative_to_grasp_seconds"]
            for row in report["frames"]
            if row["render_frame"] == report["grasp"]["render_frame"]
        )
        self.assertEqual(relative, 0.0)

    def test_error_is_recorded_when_requested(self) -> None:
        report = report_for([], result="FAILED", error="no target")
        self.assertEqual(report["result"], "FAILED")
        self.assertEqual(report["error"], "no target")

    def test_window_is_clamped_for_a_late_grasp(self) -> None:
        report = build_report(
            sequence_key="pickup_table__late_0__000",
            grasp_source_frame=222,
            grasp_time_seconds=8.88,
            source_fps=25.0,
            source_frames=250,
            render_fps=RENDER_FPS,
            render_frames=RENDER_FRAMES,
            config=VisibilityConfig(before_seconds=1.0, after_seconds=1.5),
            head_camera={"name": "head_camera", "resolution": [640, 480]},
            frames=[],
            analysis_resolution=[1280, 960],
        )
        self.assertEqual(report["window"]["end_render_frame"], RENDER_FRAMES - 1)
        self.assertTrue(report["window"]["clamped"])


class CandidateTests(unittest.TestCase):
    def _report(self, fractions: list[float], *, zero_state: str = "visible") -> dict:
        frames = [
            frame(index, value, state=zero_state if value == 0.0 else "visible")
            for index, value in enumerate(fractions)
        ]
        return report_for(frames)

    def test_candidate_row_carries_the_selection_fields(self) -> None:
        report = self._report([0.0] * 300, zero_state="outside_head_image")
        row = candidate_row(report)
        self.assertEqual(row["sequence_key"], "pickup_table__unit_0__000")
        self.assertEqual(row["grasp_source_frame"], 96)
        self.assertTrue(row["object_below_threshold_for_whole_window"])
        self.assertTrue(row["object_out_of_frame_for_whole_window"])
        self.assertEqual(row["invalid_frames"], 0)
        self.assertIn("window_below_threshold_fraction", row)

    def test_unsupported_schema_is_rejected(self) -> None:
        report = self._report([1.0] * 300)
        report["schema_version"] = 99
        with self.assertRaises(VisibilityError):
            candidate_row(report)

    def test_rows_are_sorted_and_csv_has_a_stable_header(self) -> None:
        first = self._report([1.0] * 300)
        second = self._report([0.0] * 300)
        second["sequence_key"] = "pickup_table__aaa_0__000"
        rows = candidate_rows([first, second])
        self.assertEqual([row["sequence_key"] for row in rows], ["pickup_table__aaa_0__000", "pickup_table__unit_0__000"])
        text = candidate_csv(rows)
        self.assertTrue(text.startswith("sequence_key,result,grasp_source_frame,"))
        self.assertEqual(len(text.strip().splitlines()), 3)
        self.assertIn("true", text.splitlines()[1])

    def test_empty_csv_is_empty(self) -> None:
        self.assertEqual(candidate_csv([]), "")


class LeafAggregationTests(unittest.TestCase):
    """The denominator arithmetic that turns per-leaf annotator values into pixels."""

    IMAGE = (640, 480)

    def _leaf(self, visible: int, occlusion: float | None, bbox=None, name: str = "leaf_0"):
        return LeafVisibility(name=name, path=f"/Object/{name}", visible_pixels=visible, occlusion_ratio=occlusion, bbox=bbox)

    def test_fully_visible_leaf_projects_its_pixels(self) -> None:
        aggregate = aggregate_leaf_visibility([self._leaf(1000, 0.0)], self.IMAGE)
        self.assertAlmostEqual(aggregate.projected_pixels, 1000.0)
        self.assertTrue(aggregate.exact)
        self.assertEqual(aggregate.classification, "visible")
        self.assertEqual(aggregate.reason, "")

    def test_half_occluded_leaf_doubles_its_visible_pixels(self) -> None:
        aggregate = aggregate_leaf_visibility([self._leaf(500, 0.5)], self.IMAGE)
        self.assertAlmostEqual(aggregate.projected_pixels, 1000.0)
        self.assertTrue(aggregate.exact)

    def test_multi_mesh_leaves_are_summed(self) -> None:
        leaves = [
            self._leaf(400, 0.0, name="leaf_0"),
            self._leaf(300, 0.5, name="leaf_1"),
            self._leaf(0, None, name="leaf_2"),  # out of view, no row
        ]
        aggregate = aggregate_leaf_visibility(leaves, self.IMAGE)
        self.assertAlmostEqual(aggregate.projected_pixels, 400.0 + 600.0)
        self.assertTrue(aggregate.exact)
        self.assertEqual(aggregate.classification, "visible")

    def test_fully_occluded_leaf_is_excluded_and_marked_not_exact(self) -> None:
        leaves = [self._leaf(200, 0.0, name="leaf_0"), self._leaf(0, 1.0, name="leaf_1")]
        aggregate = aggregate_leaf_visibility(leaves, self.IMAGE)
        self.assertAlmostEqual(aggregate.projected_pixels, 200.0)
        self.assertFalse(aggregate.exact)
        self.assertIn("fully_occluded_leaf", aggregate.reason)

    def test_all_leaves_fully_occluded_are_classified_and_project_nothing(self) -> None:
        leaves = [self._leaf(0, 1.0, name="leaf_0"), self._leaf(0, 1.0, name="leaf_1")]
        aggregate = aggregate_leaf_visibility(leaves, self.IMAGE)
        self.assertEqual(aggregate.projected_pixels, 0.0)
        self.assertEqual(aggregate.classification, "fully_occluded")
        self.assertFalse(aggregate.exact)

    def test_nothing_projects_without_rows(self) -> None:
        aggregate = aggregate_leaf_visibility([self._leaf(0, None)], self.IMAGE)
        self.assertEqual(aggregate.projected_pixels, 0.0)
        self.assertEqual(aggregate.classification, "outside")
        self.assertTrue(aggregate.exact)

    def test_visible_leaf_without_a_usable_ratio_is_unmeasured(self) -> None:
        for ratio in (None, -1.0, 1.0, 1.5):
            with self.subTest(ratio=ratio):
                aggregate = aggregate_leaf_visibility([self._leaf(100, ratio)], self.IMAGE)
                self.assertEqual(aggregate.classification, "unmeasured")
                self.assertIn("occlusion ratio", aggregate.reason)

    def test_leaf_row_at_the_analysis_border_marks_the_frame_not_exact(self) -> None:
        for bbox in ((0, 100, 200, 300), (100, 0, 200, 300), (100, 100, self.IMAGE[0] - 1, 300), (100, 100, 200, self.IMAGE[1] - 1)):
            with self.subTest(bbox=bbox):
                aggregate = aggregate_leaf_visibility([self._leaf(100, 0.0, bbox=bbox)], self.IMAGE)
                self.assertFalse(aggregate.exact)
                self.assertEqual(aggregate.reason, "analysis_view_clipped")

    def test_an_interior_bbox_is_exact(self) -> None:
        aggregate = aggregate_leaf_visibility([self._leaf(100, 0.0, bbox=(100, 100, 200, 300))], self.IMAGE)
        self.assertTrue(aggregate.exact)
        self.assertEqual(aggregate.reason, "")

    def test_reasons_are_unique_and_ordered(self) -> None:
        leaves = [
            self._leaf(100, 0.0, bbox=(0, 0, 200, 300), name="leaf_0"),
            self._leaf(0, 1.0, name="leaf_1"),
        ]
        aggregate = aggregate_leaf_visibility(leaves, self.IMAGE)
        self.assertEqual(aggregate.reason, "analysis_view_clipped,fully_occluded_leaf")


class CliVisibilityFlagTests(unittest.TestCase):
    """The visibility flags have to be accepted only in the right combination."""

    def _validate(self, **overrides) -> None:
        import argparse
        import contextlib
        import io

        from humanoid_lab.simulators.isaac.cli import _validate_visibility_args

        values = dict(
            visibility=False,
            replay="pickup_table__apple_0__000",
            visibility_threshold=None,
            visibility_before=None,
            visibility_after=None,
        )
        values.update(overrides)
        parser = argparse.ArgumentParser()
        with contextlib.redirect_stderr(io.StringIO()):
            _validate_visibility_args(parser, argparse.Namespace(**values))

    def test_plain_replay_and_full_visibility_pass(self) -> None:
        self._validate()
        self._validate(visibility=True)
        self._validate(visibility=True, visibility_threshold=0.3, visibility_before=0.0, visibility_after=2.5)

    def test_window_flags_without_visibility_are_rejected(self) -> None:
        for overrides in (
            {"visibility_threshold": 0.5},
            {"visibility_before": 1.0},
            {"visibility_after": 1.0},
        ):
            with self.subTest(**overrides), self.assertRaises(SystemExit):
                self._validate(**overrides)

    def test_visibility_without_replay_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self._validate(visibility=True, replay=None)

    def test_out_of_range_values_are_rejected(self) -> None:
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(SystemExit):
                self._validate(visibility=True, visibility_threshold=threshold)
        for name in ("visibility_before", "visibility_after"):
            for value in (-0.5, float("nan"), float("inf")):
                with self.subTest(**{name: value}), self.assertRaises(SystemExit):
                    self._validate(visibility=True, **{name: value})


class CollectorTests(unittest.TestCase):
    def _write_report(self, root: Path, key: str, fractions: list[float], result: str = "COMPLETED") -> None:
        directory = root / key
        directory.mkdir(parents=True)
        report = self._report(fractions, key, result)
        (directory / "visibility.json").write_text(json.dumps(report))

    def _report(self, fractions: list[float], key: str, result: str = "COMPLETED") -> dict:
        frames = [
            frame(index, value, state="outside_head_image" if value == 0.0 else "visible")
            for index, value in enumerate(fractions)
        ]
        report = report_for(frames)
        report["sequence_key"] = key
        report["result"] = result
        return report

    def test_load_select_and_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "grail-replay"
            self._write_report(root, "pickup_table__bbb_0__000", [0.0] * 300)
            self._write_report(root, "pickup_table__aaa_0__000", [1.0] * 300)
            self._write_report(root, "pickup_table__ccc_0__000", [0.0] * 300, result="FAILED")
            reports = load_reports(root)
            self.assertEqual(len(reports), 3)
            rows = select_rows(reports)
            self.assertEqual([row["sequence_key"] for row in rows], ["pickup_table__aaa_0__000", "pickup_table__bbb_0__000"])
            self.assertEqual(len(select_rows(reports, include_incomplete=True)), 3)
            below = select_rows(reports, below_threshold_only=True)
            self.assertEqual([row["sequence_key"] for row in below], ["pickup_table__bbb_0__000"])
            out_of_frame = select_rows(reports, out_of_frame_only=True)
            self.assertEqual([row["sequence_key"] for row in out_of_frame], ["pickup_table__bbb_0__000"])

    def test_missing_root_and_empty_root_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(VisibilityError):
                load_reports(Path(tmp) / "absent")
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(VisibilityError):
                load_reports(empty)

    def test_malformed_report_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "grail-replay"
            directory = root / "pickup_table__aaa_0__000"
            directory.mkdir(parents=True)
            (directory / "visibility.json").write_text("{ not json")
            with self.assertRaises(VisibilityError):
                load_reports(root)

    def test_write_rows_is_atomic_and_jsonl_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rows.jsonl"
            rows = candidate_rows([self._report([1.0] * 300, "pickup_table__aaa_0__000")])
            write_rows(rows, output, "jsonl")
            parsed = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(parsed[0]["sequence_key"], "pickup_table__aaa_0__000")
            csv_path = Path(tmp) / "rows.csv"
            write_rows(rows, csv_path, "csv")
            self.assertEqual(len(csv_path.read_text().strip().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
