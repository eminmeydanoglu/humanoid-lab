"""Target-object visibility metric, grasp-window summaries and report schema.

Primary metric
--------------
``visible_fraction`` of one render frame is the fraction of the target
object's unoccluded projected pixels that are visible inside the actual
640x480 head-camera image:

    visible_fraction = target pixels visible in the head-camera image
                       -----------------------------------------------
                       target pixels the unoccluded object projects onto the
                       head-camera image plane, including pixels outside the image

Occlusion by other geometry (robot, hands, table) and clipping at the image
boundary both count as *not visible*: the numerator only counts pixels inside
the image, while the denominator is the object's full projection, measured
over a wider view with head-camera optics and the same pixel scale. A fully
visible, unclipped object therefore reads 1.0; an object half outside the
image reads about 0.5; a fully occluded or out-of-view object reads 0.0.

The metric is measured per mesh leaf of the target and aggregated with each
leaf's occlusion ratio. Frames whose denominator cannot be measured exactly
(for example a leaf fully covered by the hand while another leaf is visible)
are reported with ``exact=false`` and a reason, never silently.

This module is pure Python: it has no Isaac, numpy or torch dependency, so it
is unit-testable anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

VISIBILITY_SCHEMA_VERSION = 1
VISIBILITY_REPORT_NAME = "visibility.json"
# Everything below the threshold is "not usable" for the grasp-window check.
DEFAULT_THRESHOLD = 0.5
DEFAULT_BEFORE_SECONDS = 1.0
DEFAULT_AFTER_SECONDS = 1.0
# The analysis view renders head-camera optics at this multiple of the head
# camera's angles. Resolution scales with it, so the pixel scale is unchanged
# and the head image is exactly the central crop of the analysis image.
ANALYSIS_VIEW_SCALE = 2
# Absorb float noise when a window edge lands exactly on a frame timestamp.
_WINDOW_EPSILON_FRAMES = 1e-9
# Rounding slack between independently rasterized pixel counts.
_PIXEL_TOLERANCE = 1.0

# Frame states explain a visible_fraction of zero without pretending to
# separate occlusion from clipping any further than the annotators can.
STATE_VISIBLE = "visible"
STATE_FULLY_OCCLUDED = "fully_occluded"
STATE_OUTSIDE_IMAGE = "outside_head_image"
STATE_UNMEASURED = "unmeasured"

_METRIC_DEFINITION = (
    "visible target pixels inside the 640x480 head-camera image divided by the "
    "target's unoccluded projection onto the head-camera image plane, including "
    "pixels outside the image; occlusion by other geometry and clipping at the image "
    "boundary both count as not visible"
)


class VisibilityError(ValueError):
    """A visibility input, config or report violated its contract."""


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise VisibilityError(f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class VisibilityConfig:
    """Opt-in visibility analysis parameters."""

    threshold: float = DEFAULT_THRESHOLD
    before_seconds: float = DEFAULT_BEFORE_SECONDS
    after_seconds: float = DEFAULT_AFTER_SECONDS

    def __post_init__(self) -> None:
        threshold = _finite("visibility threshold", self.threshold)
        if not 0.0 <= threshold <= 1.0:
            raise VisibilityError(f"visibility threshold must be in [0, 1], got {threshold}")
        before = _finite("visibility before_seconds", self.before_seconds)
        after = _finite("visibility after_seconds", self.after_seconds)
        if before < 0.0 or after < 0.0:
            raise VisibilityError(
                f"visibility window offsets must be non-negative, got before={before}, after={after}"
            )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "before_seconds", before)
        object.__setattr__(self, "after_seconds", after)


@dataclass(frozen=True)
class FrameVisibility:
    """One render frame's target visibility.

    ``source_time_seconds`` is the motion time the frame reproduces. The render
    inserts one midpoint per source interval, so every frame lies on the motion
    grid and its time is exactly ``render_frame / render_fps``.

    ``visible_fraction`` is the primary metric, or ``None`` when the frame is
    not valid: a JSON null cannot be mistaken for a measured zero. ``exact`` is
    false when the denominator had to exclude a leaf the annotator could not
    measure; such frames still report the fraction, which is then an upper
    bound of the true fraction, so a below-threshold decision stays sound.
    """

    render_frame: int
    source_time_seconds: float
    relative_to_grasp_seconds: float
    visible_fraction: float | None
    visible_pixels: int
    unoccluded_projected_pixels: float
    valid: bool = True
    exact: bool = True
    state: str = STATE_VISIBLE
    reason: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "render_frame": int(self.render_frame),
            "source_time_seconds": round(float(self.source_time_seconds), 6),
            "relative_to_grasp_seconds": round(float(self.relative_to_grasp_seconds), 6),
            "visible_fraction": (
                None if self.visible_fraction is None else round(float(self.visible_fraction), 6)
            ),
            "visible_pixels": int(self.visible_pixels),
            "unoccluded_projected_pixels": round(float(self.unoccluded_projected_pixels), 3),
            "valid": bool(self.valid),
            "exact": bool(self.exact),
            "state": self.state,
            "reason": self.reason,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FrameVisibility":
        return cls(
            render_frame=int(row["render_frame"]),
            source_time_seconds=float(row["source_time_seconds"]),
            relative_to_grasp_seconds=float(row["relative_to_grasp_seconds"]),
            visible_fraction=(
                None if row["visible_fraction"] is None else float(row["visible_fraction"])
            ),
            visible_pixels=int(row["visible_pixels"]),
            unoccluded_projected_pixels=float(row["unoccluded_projected_pixels"]),
            valid=bool(row["valid"]),
            exact=bool(row["exact"]),
            state=str(row["state"]),
            reason=str(row["reason"]),
        )


def fraction(visible_pixels: int, unoccluded_projected_pixels: float) -> float:
    """Return the primary metric from its two pixel counts.

    Zero projected pixels means the object does not reach the head camera's
    field of view at all, which is a visible fraction of zero, not missing
    data. A numerator larger than the denominator by more than one pixel is an
    inconsistent measurement and is rejected.
    """
    if visible_pixels < 0:
        raise VisibilityError(f"visible pixels cannot be negative, got {visible_pixels}")
    projected = _finite("unoccluded projected pixels", unoccluded_projected_pixels)
    if projected < 0.0:
        raise VisibilityError(f"unoccluded projected pixels cannot be negative, got {projected}")
    if visible_pixels == 0:
        return 0.0
    if projected <= 0.0 or visible_pixels > projected + _PIXEL_TOLERANCE:
        raise VisibilityError(
            f"inconsistent pixel counts: {visible_pixels} visible of {projected:.3f} unoccluded projected"
        )
    return min(1.0, visible_pixels / projected)


def unmeasured_frame(
    render_frame: int, *, render_fps: float, grasp_time_seconds: float, reason: str
) -> FrameVisibility:
    """One row whose metric could not be measured at all."""
    time_seconds = render_frame / float(render_fps)
    return FrameVisibility(
        render_frame=int(render_frame),
        source_time_seconds=time_seconds,
        relative_to_grasp_seconds=time_seconds - float(grasp_time_seconds),
        visible_fraction=None,
        visible_pixels=0,
        unoccluded_projected_pixels=0.0,
        valid=False,
        exact=False,
        state=STATE_UNMEASURED,
        reason=reason,
    )


def nearest_frame(time_seconds: float, render_fps: float, render_frames: int) -> int:
    """Frame whose timestamp is closest to ``time_seconds``, clamped to the clip."""
    if render_frames <= 0:
        raise VisibilityError("render_frames must be positive")
    frame = math.floor(float(time_seconds) * float(render_fps) + 0.5)
    return min(max(frame, 0), render_frames - 1)


@dataclass(frozen=True)
class WindowBounds:
    """The closed render-frame window of the grasp-window summary."""

    first: int
    last: int
    requested_first: int
    requested_last: int

    @property
    def frames(self) -> int:
        return self.last - self.first + 1

    @property
    def clamped(self) -> bool:
        """True when the requested window reached beyond the clip."""
        return self.first != self.requested_first or self.last != self.requested_last


def window_bounds(
    grasp_time_seconds: float,
    config: VisibilityConfig,
    render_fps: float,
    render_frames: int,
) -> WindowBounds:
    """Return the closed render-frame window ``[grasp - before, grasp + after]``.

    A frame belongs to the window when its timestamp lies inside the closed
    interval, so the first frame is the ceiling of the start time and the last
    is the floor of the end time, both with a 1e-9 frame tolerance for float
    error at exact boundaries. The window is then clamped to the clip; an empty
    window is an error rather than a silently empty summary.
    """
    if render_frames <= 0:
        raise VisibilityError("render_frames must be positive")
    start_seconds = float(grasp_time_seconds) - config.before_seconds
    end_seconds = float(grasp_time_seconds) + config.after_seconds
    requested_first = math.ceil(start_seconds * render_fps - _WINDOW_EPSILON_FRAMES)
    requested_last = math.floor(end_seconds * render_fps + _WINDOW_EPSILON_FRAMES)
    first = max(0, requested_first)
    last = min(render_frames - 1, requested_last)
    if first > last:
        raise VisibilityError(
            f"visibility window [{start_seconds:g}, {end_seconds:g}] s contains no render frame of "
            f"{render_frames} frames at {render_fps:g} Hz"
        )
    return WindowBounds(first, last, requested_first, requested_last)


def below_threshold_intervals(
    frames: Sequence[FrameVisibility], threshold: float, first_frame: int, last_frame: int
) -> list[dict[str, Any]]:
    """Contiguous runs of frames inside the window whose fraction is below the threshold.

    Only valid frames can be below the threshold, so an unmeasured frame breaks
    a run instead of being counted as invisible.
    """
    selected = sorted(
        (frame for frame in frames if frame.valid and first_frame <= frame.render_frame <= last_frame),
        key=lambda frame: frame.render_frame,
    )
    intervals: list[dict[str, Any]] = []
    current: list[FrameVisibility] = []
    for frame in selected:
        below = frame.visible_fraction < threshold
        if below and current and frame.render_frame != current[-1].render_frame + 1:
            intervals.append(_interval(current))
            current = []
        if below:
            current.append(frame)
        elif current:
            intervals.append(_interval(current))
            current = []
    if current:
        intervals.append(_interval(current))
    return intervals


def _interval(frames: Sequence[FrameVisibility]) -> dict[str, Any]:
    first, last = frames[0], frames[-1]
    return {
        "start_render_frame": int(first.render_frame),
        "end_render_frame": int(last.render_frame),
        "start_seconds": round(float(first.relative_to_grasp_seconds), 6),
        "end_seconds": round(float(last.relative_to_grasp_seconds), 6),
        "duration_seconds": round(
            float(last.relative_to_grasp_seconds - first.relative_to_grasp_seconds), 6
        ),
        "frames": len(frames),
        "min_fraction": round(min(float(frame.visible_fraction) for frame in frames), 6),
    }


def _mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else None


def summarize_frames(
    frames: Sequence[FrameVisibility],
    *,
    threshold: float,
    first_frame: int,
    last_frame: int,
    grasp_frame: int,
) -> dict[str, Any]:
    """Build the grasp-window and clip summary of one visibility time series."""
    window = [frame for frame in frames if first_frame <= frame.render_frame <= last_frame]
    expected_window_frames = last_frame - first_frame + 1
    valid_window = [frame for frame in window if frame.valid]
    valid_all = [frame for frame in frames if frame.valid]
    below = [frame for frame in valid_window if frame.visible_fraction < threshold]
    grasp = next((frame for frame in frames if frame.render_frame == grasp_frame), None)
    intervals = below_threshold_intervals(frames, threshold, first_frame, last_frame)
    return {
        "frames": len(frames),
        "valid_frames": len(valid_all),
        "exact_frames": sum(1 for frame in frames if frame.valid and frame.exact),
        "invalid_frames": len(frames) - len(valid_all),
        "clip_min_fraction": round(min((f.visible_fraction for f in valid_all), default=0.0), 6),
        "clip_mean_fraction": _rounded(_mean(f.visible_fraction for f in valid_all)),
        "window_frames": expected_window_frames,
        "window_recorded_frames": len(window),
        "window_valid_frames": len(valid_window),
        "window_invalid_frames": expected_window_frames - len(valid_window),
        "window_min_fraction": round(min((f.visible_fraction for f in valid_window), default=0.0), 6),
        "window_mean_fraction": _rounded(_mean(f.visible_fraction for f in valid_window)),
        "window_below_threshold_frames": len(below),
        "window_below_threshold_fraction": (
            round(len(below) / len(valid_window), 6) if valid_window else 0.0
        ),
        "grasp_render_frame": int(grasp_frame),
        "grasp_fraction": (
            round(float(grasp.visible_fraction), 6)
            if grasp is not None and grasp.visible_fraction is not None
            else None
        ),
        "grasp_valid": bool(grasp.valid) if grasp else False,
        "below_threshold_intervals": intervals,
        # Selection booleans: the strict ones require every window frame to be
        # measured and below the threshold (or exactly zero), so a sequence
        # selected by them really is invisible across the whole grasp window.
        "object_below_threshold_for_whole_window": len(window) == expected_window_frames
        and len(valid_window) == expected_window_frames
        and len(below) == expected_window_frames,
        "object_out_of_frame_for_whole_window": len(window) == expected_window_frames
        and len(valid_window) == expected_window_frames
        and all(
            frame.visible_fraction == 0.0 and frame.state == STATE_OUTSIDE_IMAGE
            for frame in window
        ),
        "object_below_threshold_at_any_window_frame": bool(below),
    }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def build_report(
    *,
    sequence_key: str,
    grasp_source_frame: int,
    grasp_time_seconds: float,
    source_fps: float,
    source_frames: int,
    render_fps: float,
    render_frames: int,
    config: VisibilityConfig,
    head_camera: Mapping[str, Any],
    frames: Sequence[FrameVisibility],
    analysis_resolution: Sequence[int] | None = None,
    analysis_view_scale: int = ANALYSIS_VIEW_SCALE,
    result: str = "COMPLETED",
    error: str | None = None,
) -> dict[str, Any]:
    """Assemble the machine-readable visibility report."""
    bounds = window_bounds(grasp_time_seconds, config, render_fps, render_frames)
    grasp_frame = nearest_frame(grasp_time_seconds, render_fps, render_frames)
    ordered = sorted(frames, key=lambda frame: frame.render_frame)
    report = {
        "schema_version": VISIBILITY_SCHEMA_VERSION,
        "sequence_key": sequence_key,
        "result": result,
        "metric": {
            "name": "visible_fraction",
            "definition": _METRIC_DEFINITION,
            "numerator": "target pixels visible in the head-camera image",
            "denominator": (
                "target pixels the unoccluded object projects onto the head-camera image plane, "
                "including pixels outside the image, at head-camera pixel scale"
            ),
            "occlusion_counted_as_not_visible": True,
            "image_boundary_clipping_counted_as_not_visible": True,
            "analysis_view": {
                "view_scale": int(analysis_view_scale),
                "resolution": list(analysis_resolution) if analysis_resolution else None,
                "pixel_scale_equal_to_head": True,
            },
        },
        "head_camera": dict(head_camera),
        "source": {"fps": float(source_fps), "frames": int(source_frames)},
        "render": {"fps": float(render_fps), "frames": int(render_frames)},
        # A GUI run can be stopped early, so the number of rows actually
        # measured is reported next to the clip length it was asked to cover.
        "frames_recorded": len(ordered),
        "grasp": {
            "source_frame": int(grasp_source_frame),
            "source_time_seconds": round(float(grasp_time_seconds), 6),
            "render_frame": int(grasp_frame),
        },
        "window": {
            "threshold": config.threshold,
            "before_seconds": config.before_seconds,
            "after_seconds": config.after_seconds,
            "start_seconds": round(float(grasp_time_seconds) - config.before_seconds, 6),
            "end_seconds": round(float(grasp_time_seconds) + config.after_seconds, 6),
            "start_render_frame": int(bounds.first),
            "end_render_frame": int(bounds.last),
            "frames": int(bounds.frames),
            "clamped": bool(bounds.clamped),
        },
        "frames": [frame.to_row() for frame in ordered],
        "summary": summarize_frames(
            ordered,
            threshold=config.threshold,
            first_frame=bounds.first,
            last_frame=bounds.last,
            grasp_frame=grasp_frame,
        ),
    }
    if error is not None:
        report["error"] = error
    return report


def candidate_row(report: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one report into a selection row for batch filtering."""
    if int(report.get("schema_version", 0)) != VISIBILITY_SCHEMA_VERSION:
        raise VisibilityError(f"unsupported visibility report schema_version in {report.get('sequence_key')!r}")
    summary = report["summary"]
    return {
        "sequence_key": report["sequence_key"],
        "result": report.get("result", ""),
        "grasp_source_frame": int(report["grasp"]["source_frame"]),
        "grasp_time_seconds": float(report["grasp"]["source_time_seconds"]),
        "threshold": float(report["window"]["threshold"]),
        "before_seconds": float(report["window"]["before_seconds"]),
        "after_seconds": float(report["window"]["after_seconds"]),
        "window_frames": int(summary["window_frames"]),
        "window_recorded_frames": int(summary.get("window_recorded_frames", summary["window_frames"])),
        "window_valid_frames": int(summary["window_valid_frames"]),
        "window_min_fraction": float(summary["window_min_fraction"]),
        "window_mean_fraction": (
            None if summary["window_mean_fraction"] is None else float(summary["window_mean_fraction"])
        ),
        "grasp_fraction": None if summary["grasp_fraction"] is None else float(summary["grasp_fraction"]),
        "window_below_threshold_fraction": float(summary["window_below_threshold_fraction"]),
        "object_below_threshold_for_whole_window": bool(summary["object_below_threshold_for_whole_window"]),
        "object_out_of_frame_for_whole_window": bool(summary["object_out_of_frame_for_whole_window"]),
        "invalid_frames": int(summary["invalid_frames"]),
    }


def candidate_rows(reports: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Selection rows of many reports, sorted by sequence key."""
    return sorted((candidate_row(report) for report in reports), key=lambda row: row["sequence_key"])


def candidate_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render selection rows as CSV with a fixed column order."""
    if not rows:
        return ""
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if value is None:
                values.append("")
            elif isinstance(value, bool):
                values.append("true" if value else "false")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def report_path_for(directory: Any) -> Path:
    """Report path inside one sequence output directory."""
    return Path(directory) / VISIBILITY_REPORT_NAME


@dataclass(frozen=True)
class LeafVisibility:
    """One labelled mesh leaf of the target in one frame.

    ``visible_pixels`` counts the leaf's pixels in the analysis render;
    ``occlusion_ratio`` is the annotator's 0..1 occlusion (0 fully visible,
    1 fully occluded, negative when the annotator cannot say); ``bbox`` is the
    loose bounding box in analysis-image pixels.
    """

    name: str
    path: str
    visible_pixels: int
    occlusion_ratio: float | None
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class LeafAggregate:
    """Unoccluded projection and measurement quality of all leaves of one frame."""

    projected_pixels: float
    exact: bool
    classification: str
    reason: str


def aggregate_leaf_visibility(
    leaves: Sequence[LeafVisibility], image_size: tuple[int, int]
) -> LeafAggregate:
    """Aggregate per-leaf visible pixels and occlusion ratios.

    Each measurable leaf contributes ``visible / (1 - occlusion)`` unoccluded
    projected pixels. A leaf the annotator cannot measure (no usable ratio, or
    fully occluded while others are visible) is excluded and the frame is
    marked not exact; its fraction is then an upper bound of the true fraction,
    which keeps a below-threshold decision sound. A leaf whose pixels are
    visible but has no usable ratio at all makes the frame unmeasurable.

    ``classification`` is ``visible`` when the object still projects somewhere,
    ``fully_occluded`` when every reported leaf is fully covered, and
    ``outside`` when nothing projects.
    """
    width, height = image_size
    projected = 0.0
    exact = True
    reasons: list[str] = []
    reported_leaves = 0
    occluded_leaves = 0
    for leaf in leaves:
        if leaf.visible_pixels > 0:
            if leaf.occlusion_ratio is None or not 0.0 <= leaf.occlusion_ratio < 1.0:
                return LeafAggregate(0.0, False, "unmeasured", f"no usable occlusion ratio for {leaf.path}")
            projected += leaf.visible_pixels / (1.0 - leaf.occlusion_ratio)
            if _touches_border(leaf.bbox, width, height):
                exact = False
                reasons.append("analysis_view_clipped")
            continue
        if leaf.occlusion_ratio is None:
            continue
        reported_leaves += 1
        if leaf.occlusion_ratio >= 1.0:
            occluded_leaves += 1
            exact = False
            reasons.append("fully_occluded_leaf")
        elif leaf.occlusion_ratio >= 0.0:
            exact = False
            reasons.append("unmeasured_leaf")
        if _touches_border(leaf.bbox, width, height):
            exact = False
            reasons.append("analysis_view_clipped")
    if projected > 0.0:
        classification = "visible"
    elif reported_leaves and occluded_leaves == reported_leaves:
        classification = "fully_occluded"
    else:
        classification = "outside"
    return LeafAggregate(projected, exact, classification, ",".join(dict.fromkeys(reasons)))


def _touches_border(bbox: tuple[int, int, int, int] | None, width: int, height: int) -> bool:
    """True when a loose box reaches the image edge, so its projection may be clipped."""
    if bbox is None:
        return False
    return bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= width - 1 or bbox[3] >= height - 1
