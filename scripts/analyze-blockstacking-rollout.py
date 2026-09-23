#!/usr/bin/env python3
"""Turn one rollout session's raw record into analysed, watchable artifacts.

Input is the directory ``scripts/blockstacking-rollout.py`` wrote: ``session.json``
plus ``raw/`` (bridge telemetry, Isaac samples, 50 Hz tracking, the raw video and
its frame map).  Output is one directory per rollout with a metadata file, the
time series of that rollout only, and a video cut at the robot's motion onset,
plus a session ``manifest.json``.

**Why the crop is not a fixed offset.**  A VLA backend spends seconds to minutes
between "Start" and the first closed-loop action (policy server build, first
inference, SONIC re-arm), and that latency differs per model and per switch.  A
fixed cut would hide or pad the useful part, so the onset is measured from the
recorded signals:

* *velocity*: the L2 norm of the measured arm and hand joint velocities
  (50 Hz, from the tracking rows).  The robot is held still by the SONIC support
  band until the policy takes over, so the pre-Start hold gives a baseline; the
  onset is the first time the norm stays above ``max(k * baseline_p99, floor)``
  for at least ``min_sustain_s``.
* *action*: the same rule on the applied `pose` messages (the change in each
  commanded hand/Dex3 target between consecutive messages), which shows when
  the backend actually started commanding.
* *height*: the first departure of either palm's world z from its pre-Start
  mean by more than ``height_delta_m``.

The reported onset is the earliest *sustained* velocity onset, corroborated by
the other two signals (all three are written down, with the thresholds and the
baseline statistics, so the choice can be audited or recomputed).  Everything is
a pure function of the recorded files plus the constants below.

    scripts/analyze-blockstacking-rollout.py data/outputs/blockstacking-debug/<tag>
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = 1

#: The scene every session runs unless it names a variant of its own.
SHIPPED_PROFILE = "configs/profiles/isaac-g1-sonic-blockstacking-dex3.json"

#: Joints whose measured velocity defines "the arms started moving": the bones
#: between the torso and the hands on both sides, matched by name so the
#: analysis cannot drift from the asset's ordering.
ARM_JOINT_MARKERS = ("shoulder", "elbow", "wrist")
#: Sustained-window length; below this, a single noisy sample is not an onset.
MIN_SUSTAIN_S = 0.3
#: Multiple of the baseline 99th percentile that counts as motion.
VELOCITY_BASELINE_K = 6.0
#: Absolute floor for the same test (rad/s), for a baseline that is suspiciously quiet.
VELOCITY_FLOOR_RAD_S = 0.05
#: Length of the quiet pre-Start window the baseline is taken from.  The whole
#: settle window cannot be the baseline: the start-up support band holds the
#: robot above its standing height and lets go when the policy takes over, so
#: that release transient would raise the threshold above the motion it is
#: supposed to detect.
BASELINE_WINDOW_S = 2.0
#: Palm world-height departure that counts as the hands leaving the hold pose (m).
HEIGHT_DELTA_M = 0.03
#: Commanded-target change per applied message that counts as the backend commanding.
ACTION_DELTA_FLOOR = 1e-3
#: Pelvis height below which the robot is unambiguously down rather than crouched
#: (standing is ~0.78 m, the support band holds it at ~0.96 m): a fall is not
#: the band letting go.
ROBOT_DOWN_ROOT_Z_M = 0.45
#: How long the pelvis must stay below that height to count as a fall.
ROBOT_DOWN_SUSTAIN_S = 0.5
#: Seconds of context kept before the onset in the cropped video.
DEFAULT_PRE_ROLL_S = 1.0
#: Cube centre this far above the table surface counts as lifted onto something (m).
STACK_LIFT_M = 0.03


class AnalysisError(RuntimeError):
    """The session cannot be analysed; the message is meant for the operator."""


# -- loading ---------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{path}:{number} is not JSON: {exc}") from None
    return rows


def ensure_tracking_rows(raw_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Load the 50 Hz tracking series, converting the Parquet when needed.

    The host has no ``pyarrow``; the image does, so the conversion runs in the
    container through the checkout that is mounted there.  The JSONL is cached
    next to the Parquet so a rerun does not convert twice.
    """
    jsonl_path = raw_dir / "isaac.tracking.jsonl"
    if jsonl_path.is_file():
        return load_jsonl(jsonl_path), None
    parquet_path = raw_dir / "isaac.tracking.parquet"
    if not parquet_path.is_file():
        return [], None
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        rows = pq.read_table(parquet_path).to_pylist()
        write_jsonl(jsonl_path, rows)
        return rows, "pyarrow (host)"
    container_in = f"/workspace/humanoid-lab/{parquet_path.relative_to(repo_root()).as_posix()}"
    container_out = f"/workspace/humanoid-lab/{jsonl_path.relative_to(repo_root()).as_posix()}"
    script = (
        "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && exec python -c "
        "\"import json,sys,pyarrow.parquet as pq; rows=pq.read_table(sys.argv[1]).to_pylist();"
        " open(sys.argv[2],'w',encoding='utf-8').writelines(json.dumps(r)+chr(10) for r in rows)\" "
        f"{container_in} {container_out}"
    )
    completed = subprocess.run(
        ["docker", "exec", "humanoid-lab-dev", "bash", "-lc", script],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise AnalysisError(
            "cannot convert the tracking Parquet (no host pyarrow and the container "
            f"conversion failed): {completed.stderr.strip()[:400]}"
        )
    rows = load_jsonl(jsonl_path)
    return rows, "container (isaac-sonic pyarrow)"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


# -- signals ---------------------------------------------------------------


def velocity_series(rows: Sequence[dict[str, Any]], columns: dict[str, Any]) -> list[tuple[float, float]]:
    """(sim_s, L2 norm of the measured arm+hand joint velocities) per tracking row."""
    body_names = list(columns.get("body_joints") or [])
    indices = [
        index for index, name in enumerate(body_names)
        if any(marker in name for marker in ARM_JOINT_MARKERS)
    ]
    series: list[tuple[float, float]] = []
    for row in rows:
        velocities: list[float] = []
        body = row.get("body_measured_velocity") or []
        for index in indices:
            if index < len(body):
                value = finite(body[index])
                if value is not None:
                    velocities.append(value)
        for key in ("left_hand_measured_velocity", "right_hand_measured_velocity"):
            for value in row.get(key) or []:
                number = finite(value)
                if number is not None:
                    velocities.append(number)
        if not velocities:
            continue
        sim_s = finite(row.get("sim_s"))
        if sim_s is None:
            continue
        series.append((sim_s, math.sqrt(sum(value * value for value in velocities))))
    return series


def action_series(events: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    """(wall_s, change in the commanded hand targets since the previous message)."""
    series: list[tuple[float, float]] = []
    previous: list[float] | None = None
    for event in events:
        fields = event.get("fields") or {}
        current: list[float] = []
        for key in ("left_hand_joints", "right_hand_joints"):
            value = fields.get(key)
            if isinstance(value, list):
                current.extend(float(item) for item in flatten(value))
        wall = finite(event.get("wall_time"))
        if wall is None or not current:
            continue
        if previous is not None and len(previous) == len(current):
            delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(current, previous)))
            series.append((wall, delta))
        previous = current
    return series


def flatten(value: Any) -> Iterable[float]:
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def palm_series(samples: Sequence[dict[str, Any]], link: str) -> list[tuple[float, float]]:
    """(wall_s, palm world z) per Isaac sample for one palm link."""
    series: list[tuple[float, float]] = []
    for sample in samples:
        pose = ((sample.get("palm_pose_w") or {}).get(link))
        wall = finite(sample.get("wall_time_ns"))
        if not isinstance(pose, list) or len(pose) < 3 or wall is None:
            continue
        height = finite(pose[2])
        if height is not None:
            series.append((wall / 1e9, height))
    return series


def sustained_onset(
    series: Sequence[tuple[float, float]],
    *,
    threshold: float,
    min_sustain_s: float,
) -> float | None:
    """First time the series stays above ``threshold`` for ``min_sustain_s``."""
    if not series:
        return None
    run_start: float | None = None
    for timestamp, value in series:
        if value > threshold:
            if run_start is None:
                run_start = timestamp
            elif timestamp - run_start >= min_sustain_s:
                return run_start
        else:
            run_start = None
    return None


# -- analysis --------------------------------------------------------------


def window(rows: Sequence[dict[str, Any]], key: str, low_ns: int, high_ns: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        value = finite(row.get(key))
        if value is None:
            continue
        if low_ns <= int(value) <= high_ns:
            selected.append(row)
    return selected


def rollout_series(rows: Sequence[dict[str, Any]], key_ns: str, low_ns: int, high_ns: int) -> list[dict[str, Any]]:
    return window(rows, key_ns, low_ns, high_ns)


def analyse_rollout(
    *,
    index: int,
    rollout: dict[str, Any],
    tracking: Sequence[dict[str, Any]],
    tracking_columns: dict[str, Any],
    bridge_events: Sequence[dict[str, Any]],
    samples: Sequence[dict[str, Any]],
    frame_map: Sequence[dict[str, Any]],
    video_fps: float,
    pre_roll_s: float,
) -> dict[str, Any]:
    start_ns = int(rollout["start_wall_ns"])
    stop_ns = int(rollout["stop_wall_ns"])
    reset_ns = int(rollout["reset_wall_ns"])
    # The quietest pre-Start window is the baseline: SONIC holds the robot in the
    # standing pose there, while the whole settle may still contain the
    # support-band release.
    baseline_low_ns, baseline_high_ns = quiet_baseline_window(
        tracking, start_ns=start_ns, reset_ns=reset_ns, length_s=BASELINE_WINDOW_S,
        score=lambda segment: [value for _, value in velocity_series(segment, tracking_columns)],
    )
    tracking_baseline = window(tracking, "wall_time_ns", baseline_low_ns, baseline_high_ns)
    tracking_run = window(tracking, "wall_time_ns", start_ns, stop_ns)
    bridge_run = window(bridge_events, "wall_time_ns", start_ns, stop_ns)
    bridge_baseline = window(bridge_events, "wall_time_ns", baseline_low_ns, baseline_high_ns)
    samples_run = window(samples, "wall_time_ns", start_ns, stop_ns)
    samples_baseline = window(samples, "wall_time_ns", reset_ns, start_ns)

    velocities_baseline = velocity_series(tracking_baseline, tracking_columns)
    velocities_run = velocity_series(tracking_run, tracking_columns)
    baseline_p99 = percentile([value for _, value in velocities_baseline], 0.99)
    threshold = max(VELOCITY_BASELINE_K * (baseline_p99 or 0.0), VELOCITY_FLOOR_RAD_S)

    # sim_s -> wall time, so every onset is reported on both clocks.
    sim_origin = None
    for row in tracking_run:
        sim_s, wall_ns = finite(row.get("sim_s")), finite(row.get("wall_time_ns"))
        if sim_s is not None and wall_ns is not None:
            sim_origin = (sim_s, wall_ns / 1e9)
            break

    velocity_onset_sim = sustained_onset(velocities_run, threshold=threshold, min_sustain_s=MIN_SUSTAIN_S)

    velocity_onset_wall = (
        sim_origin[1] + (velocity_onset_sim - sim_origin[0]) if velocity_onset_sim is not None and sim_origin else None
    )

    actions_baseline = [
        event for event in bridge_baseline
        if event.get("kind") == "applied_action" and not event.get("decode_error")
    ]
    actions_run = [
        event for event in bridge_run
        if event.get("kind") == "applied_action" and not event.get("decode_error")
    ]
    action_deltas_baseline = action_series(actions_baseline)
    action_deltas_run = action_series(actions_run)
    action_threshold = max(
        percentile([value for _, value in action_deltas_baseline], 0.99) or 0.0,
        ACTION_DELTA_FLOOR,
    )
    action_onset_wall = sustained_onset(
        action_deltas_run, threshold=action_threshold, min_sustain_s=MIN_SUSTAIN_S
    )

    heights: dict[str, Any] = {}
    height_onset_wall: float | None = None
    for link in ("left_hand_palm_link", "right_hand_palm_link"):
        baseline_values = [value for _, value in palm_series(samples_baseline, link)]
        run_values = palm_series(samples_run, link)
        if not baseline_values or not run_values:
            heights[link] = {"baseline_mean_m": None, "onset_wall": None}
            continue
        mean = sum(baseline_values) / len(baseline_values)
        onsets = [
            (timestamp, value) for timestamp, value in run_values if abs(value - mean) > HEIGHT_DELTA_M
        ]
        onset = sustained_onset(
            run_values, threshold=mean + HEIGHT_DELTA_M, min_sustain_s=MIN_SUSTAIN_S
        )
        downward = sustained_onset(
            [(t, -v) for t, v in run_values], threshold=-(mean - HEIGHT_DELTA_M), min_sustain_s=MIN_SUSTAIN_S
        )
        heights[link] = {
            "baseline_mean_m": round(mean, 5),
            "baseline_first_m": round(run_values[0][1], 5),
            "min_m": round(min(v for _, v in run_values), 5),
            "max_m": round(max(v for _, v in run_values), 5),
            "onset_above_wall": onsets[0][0] if onsets else None,
            "onset_up_wall": onset,
            "onset_down_wall": downward,
        }
        for candidate in (onsets[0][0] if onsets else None,):
            if candidate is not None and (height_onset_wall is None or candidate < height_onset_wall):
                height_onset_wall = candidate

    onset_wall = velocity_onset_wall
    onset_source = "velocity"
    if onset_wall is None and action_onset_wall is not None:
        onset_wall = action_onset_wall
        onset_source = "action"

    # Video: the frame map is the only exact frame<->time relation.
    frames = [
        frame for frame in frame_map
        if start_ns <= int(finite(frame.get("wall_time_ns")) or 0) <= stop_ns
    ]
    onset_frame = None
    if onset_wall is not None:
        for frame in frames:
            if (finite(frame.get("wall_time_ns")) or 0) / 1e9 >= onset_wall - pre_roll_s:
                onset_frame = int(frame["frame"])
                break
    fall = detect_fall(tracking_run, tracking_baseline)
    stack = stack_state(samples_run, samples_baseline)
    errors = [
        {"wall_time": event.get("wall_time"), "state": event.get("state"), "error": event.get("error")}
        for event in bridge_run
        if event.get("kind") == "session" and event.get("state") == "ERROR"
    ]
    return {
        "index": index,
        "start_wall_ns": start_ns,
        "stop_wall_ns": stop_ns,
        "duration_wall_s": round((stop_ns - start_ns) / 1e9, 3),
        "sim_seconds_recorded": (
            round(finite(tracking_run[-1].get("sim_s")) - finite(tracking_run[0].get("sim_s")), 3)
            if tracking_run and finite(tracking_run[0].get("sim_s")) is not None else None
        ),
        "tracking_rows": len(tracking_run),
        "observations": sum(1 for event in bridge_run if event.get("kind") == "observation"),
        "target_actions": sum(1 for event in bridge_run if event.get("kind") == "target_action"),
        "applied_actions": len(actions_run),
        "frames_in_video": len(frames),
        "onset": {
            "method": "sustained velocity onset (arms+hands) with action/height corroboration",
            "chosen_source": onset_source if onset_wall is not None else None,
            "wall_time": onset_wall,
            "sim_s": velocity_onset_sim,
            "seconds_after_start": (
                None if onset_wall is None else round(onset_wall - start_ns / 1e9, 3)
            ),
            "velocity": {
                "baseline_window_wall_ns": [baseline_low_ns, baseline_high_ns],
                "baseline_window_s": BASELINE_WINDOW_S,
                "baseline_p99_rad_s": baseline_p99,
                "threshold_rad_s": threshold,
                "onset_wall": velocity_onset_wall,
                "onset_sim_s": velocity_onset_sim,
                "k": VELOCITY_BASELINE_K,
                "floor_rad_s": VELOCITY_FLOOR_RAD_S,
                "min_sustain_s": MIN_SUSTAIN_S,
                # A pre-Start window that is itself moving (the robot left limp
                # between a Stop and the Reset) raises the threshold above the
                # motion it is meant to catch; say so instead of reporting a
                # silent "no onset".
                "unavailable_reason": (
                    None if velocity_onset_sim is not None else
                    "the pre-Start hold was not still: baseline p99 "
                    f"{baseline_p99:.3f} rad/s gives a {threshold:.3f} rad/s threshold while the "
                    f"rollout's own arm+hand velocity peaks at "
                    f"{max((value for _, value in velocities_run), default=0.0):.3f} rad/s"
                ),
            },
            "action": {
                "baseline_p99": action_threshold,
                "onset_wall": action_onset_wall,
                "min_sustain_s": MIN_SUSTAIN_S,
            },
            "height": {"delta_m": HEIGHT_DELTA_M, "onset_wall": height_onset_wall, "palms": heights},
        },
        "fall": fall,
        "session_errors": errors,
        "stack": stack,
        "behaviour": behaviour_summary(
            tracking_run=tracking_run,
            tracking_columns=tracking_columns,
            samples_run=samples_run,
            samples_baseline=samples_baseline,
            actions_run=actions_run,
        ),
        "video": {
            "fps": video_fps,
            "first_frame": int(frames[0]["frame"]) if frames else None,
            "last_frame": int(frames[-1]["frame"]) if frames else None,
            "onset_frame": onset_frame,
        },
    }


def quiet_baseline_window(
    rows: Sequence[dict[str, Any]],
    *,
    start_ns: int,
    reset_ns: int,
    length_s: float,
    score: Callable[[Sequence[dict[str, Any]]], Sequence[float]],
) -> tuple[int, int]:
    """The calmest ``length_s`` window between the reset and the Start.

    Deterministic: every candidate window of that length is scored by the median
    of ``score`` (the measured arm+hand velocity), and the lowest score wins
    (ties go to the earliest window).  This is what keeps the band-release
    transient -- which happens *after* Start -- out of the threshold it would
    otherwise inflate.
    """
    span_ns = int(length_s * 1e9)
    if start_ns - reset_ns <= span_ns:
        return reset_ns, start_ns
    best: tuple[float, int, int] | None = None
    window_start = reset_ns
    while window_start + span_ns <= start_ns:
        values = sorted(score(window(rows, "wall_time_ns", window_start, window_start + span_ns)))
        if values:
            candidate = values[len(values) // 2]
            if best is None or candidate < best[0]:
                best = (candidate, window_start, window_start + span_ns)
        window_start += max(span_ns // 2, 1)
    if best is None:
        return start_ns - span_ns, start_ns
    return best[1], best[2]


def behaviour_summary(
    *,
    tracking_run: Sequence[dict[str, Any]],
    tracking_columns: dict[str, Any],
    samples_run: Sequence[dict[str, Any]],
    samples_baseline: Sequence[dict[str, Any]],
    actions_run: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """What the robot actually did: hand heights, cube motion, arm joint travel.

    These are the numbers the observation "the hands are held high and do not
    come down" is checked against -- palm heights above the worktop, how long
    they stayed there, whether any cube moved at all, and how far each arm joint
    travelled -- so they are recorded per rollout instead of being recomputed by
    hand later.
    """
    surface = None
    for sample in reversed(samples_run):
        scene = sample.get("scene") or {}
        if scene.get("live_worktop_height_m") is not None:
            surface = float(scene["live_worktop_height_m"])
            break

    palms: dict[str, Any] = {}
    for link in ("left_hand_palm_link", "right_hand_palm_link"):
        heights = [
            (sample["palm_pose_w"][link][2] - surface)
            for sample in samples_run
            if isinstance(sample.get("palm_pose_w", {}).get(link), list)
            and len(sample["palm_pose_w"][link]) >= 3 and surface is not None
        ]
        baseline_heights = [
            (sample["palm_pose_w"][link][2] - surface)
            for sample in samples_baseline
            if isinstance(sample.get("palm_pose_w", {}).get(link), list)
            and len(sample["palm_pose_w"][link]) >= 3 and surface is not None
        ]
        if not heights:
            palms[link] = None
            continue
        ordered = sorted(heights)
        palms[link] = {
            "baseline_median_above_table_m": (
                round(sorted(baseline_heights)[len(baseline_heights) // 2], 4) if baseline_heights else None
            ),
            "min_above_table_m": round(ordered[0], 4),
            "median_above_table_m": round(ordered[len(ordered) // 2], 4),
            "max_above_table_m": round(ordered[-1], 4),
            "end_above_table_m": round(heights[-1], 4),
            "fraction_above_0_10_m": round(sum(1 for value in heights if value > 0.10) / len(heights), 3),
            "samples": len(heights),
        }

    cube_displacement: dict[str, float] = {}
    cube_lift: dict[str, float] = {}
    for sample in samples_run:
        for color, entry in (sample.get("scene", {}).get("cubes") or {}).items():
            displacement = finite(entry.get("displacement_xy_m"))
            if displacement is not None:
                cube_displacement[color] = max(cube_displacement.get(color, 0.0), displacement)
            height = finite(entry.get("live_center_z_m"))
            if height is not None and surface is not None:
                cube_lift[color] = max(cube_lift.get(color, 0.0), height - (surface + 0.025))

    body_names = list(tracking_columns.get("body_joints") or [])
    arm_indices = [
        index for index, name in enumerate(body_names)
        if any(marker in name for marker in ARM_JOINT_MARKERS)
    ]
    arm_travel: dict[str, Any] = {}
    for index in arm_indices:
        values = [
            finite(row["body_measured"][index])
            for row in tracking_run
            if isinstance(row.get("body_measured"), list) and index < len(row["body_measured"])
        ]
        values = [value for value in values if value is not None]
        if values:
            arm_travel[body_names[index]] = {
                "min_rad": round(min(values), 4),
                "max_rad": round(max(values), 4),
                "range_rad": round(max(values) - min(values), 4),
            }

    hand_targets: dict[str, Any] = {}
    for side in ("left", "right"):
        values = [
            float(value)
            for event in actions_run
            for value in flatten((event.get("fields") or {}).get(f"{side}_hand_joints") or [])
        ]
        if values:
            hand_targets[side] = {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "range": round(max(values) - min(values), 4),
                "samples": len(values),
            }

    return {
        "table_surface_m": surface,
        "palm_height_above_table_m": palms,
        "cube_max_displacement_xy_m": {key: round(value, 5) for key, value in cube_displacement.items()},
        "cube_max_lift_above_resting_m": {key: round(value, 5) for key, value in cube_lift.items()},
        "arm_joint_travel_rad": arm_travel,
        "applied_hand_target_range": hand_targets or None,
    }


def detect_fall(tracking_run: Sequence[dict[str, Any]], baseline_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Whether the robot went down, distinguished from the support band releasing.

    The start-up band lifts the robot above its standing height and lets go when
    the policy takes over, so a drop at the beginning of a rollout is the band
    doing its job.  A fall is the pelvis staying below :data:`ROBOT_DOWN_ROOT_Z_M`,
    which no standing or crouching pose reaches.
    """
    heights = [
        (finite(row.get("sim_s")), finite((row.get("root_position") or [None, None, None])[2]))
        for row in tracking_run
    ]
    usable = [(sim, z) for sim, z in heights if sim is not None and z is not None]
    if len(usable) < 10:
        return {"detected": None, "reason": "not enough root samples"}
    band_values = sorted(
        value for value in
        (finite((row.get("root_position") or [None, None, None])[2]) for row in baseline_rows)
        if value is not None
    )
    tail = [z for _, z in usable[-100:]]
    below = [sim for sim, z in usable if z < ROBOT_DOWN_ROOT_Z_M]
    sustained = sustained_onset(
        [(sim, -z) for sim, z in usable], threshold=-ROBOT_DOWN_ROOT_Z_M, min_sustain_s=ROBOT_DOWN_SUSTAIN_S
    )
    return {
        "detected": sustained is not None,
        "criterion": f"pelvis below {ROBOT_DOWN_ROOT_Z_M} m for >= {ROBOT_DOWN_SUSTAIN_S} s",
        "band_hold_root_z_m": round(band_values[len(band_values) // 2], 4) if band_values else None,
        "settled_root_z_m": round(sorted(tail)[len(tail) // 2], 4),
        "minimum_root_z_m": round(min(z for _, z in usable), 4),
        "band_release_drop_m": (
            None if not band_values else round(band_values[len(band_values) // 2] - min(z for _, z in usable), 4)
        ),
        "sim_s_below_threshold": below[0] if below else None,
        "sim_s": sustained,
    }


def stack_state(samples_run: Sequence[dict[str, Any]], samples_baseline: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Where the cubes are and on what they rest, measured, not judged.

    A cube whose centre sits at least ``STACK_LIFT_M`` above the table surface
    plus half its size is resting on something; which cube is higher says what
    is on top.  The declared order (red, yellow, blue) is *not* applied here --
    this is a measurement the report can quote.
    """
    initial: dict[str, float] = {}
    final: dict[str, dict[str, Any]] = {}
    for sample in samples_baseline[:5]:
        for color, entry in (sample.get("scene", {}).get("cubes") or {}).items():
            height = finite(entry.get("live_center_z_m"))
            if height is not None:
                initial.setdefault(color, height)
    for sample in samples_run:
        for color, entry in (sample.get("scene", {}).get("cubes") or {}).items():
            final[color] = {
                "center_xyz_m": entry.get("live_center_xyz_m"),
                "displacement_xy_m": entry.get("displacement_xy_m"),
            }
    last = samples_run[-1].get("scene", {}) if samples_run else {}
    surface = finite(last.get("live_worktop_height_m"))
    heights = {
        color: finite((entry or {}).get("live_center_z_m"))
        for color, entry in (last.get("cubes") or {}).items()
    }
    lifted = {
        color: (None if height is None or surface is None else round(height - (surface + 0.025), 4))
        for color, height in heights.items()
    }
    return {
        "table_surface_m": surface,
        "cube_center_z_m": heights,
        "cube_lift_above_surface_m": lifted,
        "cube_pose_w": final,
        "initial_cube_center_z_m": initial,
        "any_cube_lifted": (
            None if not any(value is not None for value in lifted.values())
            else any(value is not None and value > STACK_LIFT_M for value in lifted.values())
        ),
        "highest_cube": (
            None if not any(value is not None for value in lifted.values())
            else max((color for color, value in lifted.items() if value is not None), key=lambda c: lifted[c])
        ),
    }


# -- video -----------------------------------------------------------------


def ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip()[:300]}
    return json.loads(completed.stdout)


def crop_video(source: Path, target: Path, *, start_s: float, duration_s: float) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{max(0.0, start_s):.3f}", "-i", str(source), "-t", f"{duration_s:.3f}",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise AnalysisError(f"ffmpeg failed on {source}: {completed.stderr.strip()[:400]}")
    probe = ffprobe(target)
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "probe": {
            "duration_s": float(probe.get("format", {}).get("duration", 0.0) or 0.0),
            "frames": next(
                (int(s["nb_frames"]) for s in probe.get("streams", []) if s.get("nb_frames")), None
            ),
            "width": next((s.get("width") for s in probe.get("streams", []) if s.get("width")), None),
            "height": next((s.get("height") for s in probe.get("streams", []) if s.get("height")), None),
            "codec": next((s.get("codec_name") for s in probe.get("streams", []) if s.get("codec_name")), None),
        },
    }


# -- main ------------------------------------------------------------------


def verify_session(session_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check that everything the manifest promises is there and readable.

    A video that does not decode and a telemetry file that does not parse are
    the two ways a campaign can look complete while being unusable, so both are
    opened here: every clip is probed for a video stream and a duration, and
    every JSONL is read line by line.
    """
    checks: list[dict[str, Any]] = []

    def note(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    raw = manifest.get("raw", {})
    raw_video = Path(raw.get("video", {}).get("path", ""))
    if raw_video.is_file():
        probe = ffprobe(raw_video)
        streams = probe.get("streams", [])
        duration = float(probe.get("format", {}).get("duration", 0.0) or 0.0)
        note("raw video decodes", bool(streams) and duration > 0.0,
             f"{len(streams)} stream(s), {duration:.2f}s, {raw_video.stat().st_size} bytes")
    else:
        note("raw video decodes", False, f"missing: {raw_video}")

    for key in ("video_timestamps", "bridge_telemetry", "isaac_samples", "tracking_jsonl"):
        path = Path(raw.get(key, ""))
        if not path.is_file():
            note(f"{key} present", False, f"missing: {path}")
            continue
        try:
            rows = sum(1 for line in path.open(encoding="utf-8", errors="strict") if line.strip())
        except (OSError, UnicodeDecodeError) as exc:
            note(f"{key} parses", False, f"{type(exc).__name__}: {exc}")
            continue
        note(f"{key} parses", rows > 0, f"{rows} JSON lines, {path.stat().st_size} bytes")

    parquet = Path(raw.get("isaac_tracking", ""))
    note("tracking parquet present", parquet.is_file(),
         f"{parquet.stat().st_size} bytes" if parquet.is_file() else f"missing: {parquet}")

    for rollout in manifest.get("rollouts", []):
        index = rollout.get("index")
        crop = (rollout.get("video") or {}).get("cropped")
        if crop is None:
            note(f"rollout {index} crop", False, str((rollout.get("video") or {}).get("crop_skipped", "no clip")))
            continue
        path = Path(crop["path"])
        probe = ffprobe(path) if path.is_file() else {}
        streams = probe.get("streams", [])
        duration = float(probe.get("format", {}).get("duration", 0.0) or 0.0)
        note(f"rollout {index} crop decodes", bool(streams) and duration > 0.0,
             f"{duration:.2f}s, {path.stat().st_size} bytes, start {crop.get('start_s')}s")
        for name in ("bridge_telemetry", "isaac_samples", "tracking", "video_frame_map"):
            target = (rollout.get("files") or {}).get(name)
            if not target:
                continue
            target_path = Path(target)
            try:
                rows = sum(1 for line in target_path.open(encoding="utf-8", errors="strict") if line.strip())
                note(f"rollout {index} {name} parses", rows > 0, f"{rows} JSON lines")
            except (OSError, UnicodeDecodeError) as exc:
                note(f"rollout {index} {name} parses", False, f"{type(exc).__name__}: {exc}")

    passed = all(check["ok"] for check in checks)
    return {"passed": passed, "checks": checks}


def contact_sheet(source: Path, target: Path, *, columns: int = 3, rows: int = 2) -> dict[str, Any] | None:
    """A 3x2 overview of a clip, so a long recording can be judged at a glance."""
    frames = columns * rows
    completed = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", f"select='not(mod(n\\,{max(1, 140)}))',scale=320:-1,tile={columns}x{rows}",
            "-frames:v", "1", "-q:v", "3", str(target),
        ],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0 or not target.is_file():
        return None
    return {"path": str(target), "bytes": target.stat().st_size, "frames": frames}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse one BlockStacking rollout session")
    parser.add_argument("session_dir", type=Path, nargs="?",
                        help="a session directory written by scripts/blockstacking-rollout.py")
    parser.add_argument("--index", type=Path, default=None,
                        help="instead of analysing, collect every <dir>/manifest.json underneath "
                             "this campaign directory into <dir>/campaign.json")
    parser.add_argument("--pre-roll", type=float, default=DEFAULT_PRE_ROLL_S,
                        help="seconds of context kept before the onset in the cropped video")
    parser.add_argument("--no-video", action="store_true", help="skip cropping (analysis only)")
    return parser.parse_args(argv)


def build_campaign_index(campaign_dir: Path) -> dict[str, Any]:
    """One machine-readable entry point over every analysed session underneath.

    A second reader (or a later comparison against the training dataset) should
    not have to know how the per-session directories are named; this lists them
    with the checkpoint identity, the onset and the verification verdict.
    """
    sessions: list[dict[str, Any]] = []
    for manifest_path in sorted(campaign_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        session_dir = manifest_path.parent
        sessions.append({
            "tag": manifest.get("tag"),
            "session_dir": str(session_dir.relative_to(campaign_dir)),
            "manifest": str(manifest_path),
            "model": manifest.get("model"),
            "checkpoints": manifest.get("checkpoints"),
            "checkpoint_step": manifest.get("checkpoint_step"),
            "prompt": manifest.get("prompt"),
            "video_fps": (manifest.get("config") or {}).get("video_fps"),
            "raw_video": (manifest.get("raw") or {}).get("video"),
            "rollouts": [
                {
                    "index": rollout.get("index"),
                    "onset_seconds_after_start": (rollout.get("onset") or {}).get("seconds_after_start"),
                    "onset_source": (rollout.get("onset") or {}).get("chosen_source"),
                    "duration_wall_s": rollout.get("duration_wall_s"),
                    "sim_seconds_recorded": rollout.get("sim_seconds_recorded"),
                    "applied_actions": rollout.get("applied_actions"),
                    "observations": rollout.get("observations"),
                    "cubes_lifted": (rollout.get("stack") or {}).get("any_cube_lifted"),
                    "robot_down": (rollout.get("fall") or {}).get("detected"),
                    "cropped_video": ((rollout.get("video") or {}).get("cropped") or {}).get("path"),
                    "rollout_dir": f"{session_dir.relative_to(campaign_dir)}/rollouts/rollout-{int(rollout.get('index', 0)):02d}",
                }
                for rollout in manifest.get("rollouts", [])
            ],
            "verification_passed": (manifest.get("verification") or {}).get("passed"),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_dir": str(campaign_dir),
        "sessions": sessions,
        "task": "BlockStacking",
        "prompt": next((session["prompt"] for session in sessions if session.get("prompt")), None),
        "notes": [
            "Per-session manifests carry the exact commands, paths and method constants.",
            "Onsets are measured, not assumed: see manifest.method.onset and each rollout's onset block.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.index is not None:
        campaign_dir = args.index.resolve()
        if not campaign_dir.is_dir():
            print(f"error: {campaign_dir} is not a directory", file=sys.stderr)
            return 2
        index = build_campaign_index(campaign_dir)
        target = campaign_dir / "campaign.json"
        target.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[analyze] campaign index: {target} ({len(index['sessions'])} session(s))")
        return 0
    if args.session_dir is None:
        print("error: give a session directory or --index CAMPAIGN_DIR", file=sys.stderr)
        return 2
    session_dir = args.session_dir.resolve()
    session_path = session_dir / "session.json"
    if not session_path.is_file():
        print(f"error: {session_path} is missing", file=sys.stderr)
        return 2
    session = json.loads(session_path.read_text(encoding="utf-8"))
    raw_dir = session_dir / "raw"

    bridge_events = load_jsonl(raw_dir / "telemetry" / "bridge-telemetry.jsonl")
    samples = load_jsonl(raw_dir / "isaac.samples.jsonl")
    frame_map = load_jsonl(raw_dir / "video.timestamps.jsonl")
    metrics_path = raw_dir / "isaac.metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    tracking_columns = (metrics.get("tracking_columns") or {}) if isinstance(metrics, dict) else {}
    if not tracking_columns:
        print("warning: isaac.metrics.json has no tracking_columns; velocity onsets are unavailable",
              file=sys.stderr)
    tracking, tracking_source = ensure_tracking_rows(raw_dir)
    video_path = raw_dir / "video.raw.mp4"
    video_fps = float((metrics.get("video") or {}).get("fps") or 25.0)
    raw_probe = ffprobe(video_path) if video_path.is_file() else {}

    rollouts_dir = session_dir / "rollouts"
    analysed: list[dict[str, Any]] = []
    for rollout in session.get("rollouts", []):
        index = int(rollout["index"])
        record = analyse_rollout(
            index=index, rollout=rollout, tracking=tracking, tracking_columns=tracking_columns,
            bridge_events=bridge_events, samples=samples, frame_map=frame_map,
            video_fps=video_fps, pre_roll_s=args.pre_roll,
        )
        start_ns, stop_ns = record["start_wall_ns"], record["stop_wall_ns"]
        rollout_dir = rollouts_dir / f"rollout-{index:02d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(rollout_dir / "bridge-telemetry.jsonl", window(bridge_events, "wall_time_ns", start_ns, stop_ns))
        write_jsonl(rollout_dir / "isaac.samples.jsonl", window(samples, "wall_time_ns", start_ns, stop_ns))
        write_jsonl(rollout_dir / "video.timestamps.jsonl", window(frame_map, "wall_time_ns", start_ns, stop_ns))
        if tracking:
            write_jsonl(rollout_dir / "tracking.jsonl", window(tracking, "wall_time_ns", start_ns, stop_ns))
        record["files"] = {
            "bridge_telemetry": str(rollout_dir / "bridge-telemetry.jsonl"),
            "isaac_samples": str(rollout_dir / "isaac.samples.jsonl"),
            "tracking": str(rollout_dir / "tracking.jsonl") if tracking else None,
            "video_frame_map": str(rollout_dir / "video.timestamps.jsonl"),
        }
        if not args.no_video and video_path.is_file():
            onset = record["onset"]
            start_wall = None if onset["wall_time"] is None else onset["wall_time"] - pre_roll_seconds(args)
            start_s = video_time_for_wall(frame_map, start_wall, video_fps)
            end_s = video_time_for_wall(frame_map, record["stop_wall_ns"] / 1e9, video_fps, last=True)
            if onset["wall_time"] is None:
                record["video"]["crop_skipped"] = "no onset detected; the raw video is kept uncropped"
            elif start_s is None or end_s is None:
                record["video"]["crop_skipped"] = "the frame map does not cover this rollout"
            else:
                crop = crop_video(
                    video_path, rollout_dir / "video.mp4",
                    start_s=start_s, duration_s=max(0.5, end_s - start_s),
                )
                crop["start_s"] = round(start_s, 3)
                crop["requested_duration_s"] = round(max(0.5, end_s - start_s), 3)
                crop["onset_frame"] = onset["chosen_source"] and record["video"]["onset_frame"]
                sheet = contact_sheet(Path(crop["path"]), rollout_dir / "contact-sheet.jpg")
                if sheet is not None:
                    crop["contact_sheet"] = sheet
                record["video"]["cropped"] = crop
        (rollout_dir / "rollout.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        analysed.append(record)
        after_start = record["onset"]["seconds_after_start"]
        onset_text = "none" if after_start is None else f"{after_start:.2f}s after Start"
        print(
            f"[analyze] rollout {index}: onset {onset_text}, "
            f"observations {record['observations']}, applied {record['applied_actions']}"
        )

    scene_profile = session.get("scene_profile") or {}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tag": session.get("tag"),
        "task": session.get("task"),
        "prompt": session.get("prompt"),
        "model": session.get("requested_model_id"),
        "requested_model": session.get("requested_model"),
        "checkpoints": session.get("checkpoints"),
        "checkpoint_step": session.get("checkpoint_step"),
        "command": session.get("command"),
        "config": {
            # The scene the session actually ran: the shipped profile unless the
            # session named a variant, whose digest is carried next to it.
            "profile": scene_profile.get("file") or SHIPPED_PROFILE,
            "profile_sha256": scene_profile.get("sha256"),
            "scene_profile": scene_profile or None,
            "video_fps": video_fps,
            "control_hz": next(
                (event.get("control_hz") for event in bridge_events if event.get("kind") == "policy"), None
            ),
            "physics_dt": 0.005,
            "render_interval": metrics.get("render_interval"),
            "prompt": next(
                (event.get("prompt") for event in bridge_events if event.get("kind") == "policy"), None
            ),
        },
        "raw": {
            "video": {"path": str(video_path), "bytes": video_path.stat().st_size if video_path.is_file() else 0,
                      "frames": (metrics.get("video") or {}).get("frames"),
                      "duration_s": float(raw_probe.get("format", {}).get("duration", 0.0) or 0.0) if raw_probe else None},
            "video_timestamps": str(raw_dir / "video.timestamps.jsonl"),
            "bridge_telemetry": str(raw_dir / "telemetry" / "bridge-telemetry.jsonl"),
            "isaac_samples": str(raw_dir / "isaac.samples.jsonl"),
            "isaac_tracking": str(raw_dir / "isaac.tracking.parquet"),
            "tracking_jsonl": str(raw_dir / "isaac.tracking.jsonl"),
            "tracking_source": tracking_source,
            "isaac_metrics": str(metrics_path),
            "logs": session.get("copied_logs"),
        },
        "counts": {
            "bridge_events": len(bridge_events),
            "isaac_samples": len(samples),
            "video_frames": len(frame_map),
            "tracking_rows": len(tracking),
        },
        "rollouts": analysed,
        "method": {
            "onset": (
                "sustained velocity onset: measured arm+hand joint velocity norm above "
                f"max({VELOCITY_BASELINE_K} x pre-Start baseline p99, {VELOCITY_FLOOR_RAD_S} rad/s) for "
                f"{MIN_SUSTAIN_S}s; corroborated by applied-action deltas and palm height departure"
            ),
            "pre_roll_s": pre_roll_seconds(args),
            "robot_down_root_z_m": ROBOT_DOWN_ROOT_Z_M,
            "robot_down_sustain_s": ROBOT_DOWN_SUSTAIN_S,
            "baseline_window_s": BASELINE_WINDOW_S,
            "stack_lift_threshold_m": STACK_LIFT_M,
        },
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verification = verify_session(session_dir, manifest)
    manifest["verification"] = verification
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (session_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[analyze] manifest: {session_dir / 'manifest.json'}")
    print(f"[analyze] verification: {'PASS' if verification['passed'] else 'FAIL'} "
          f"({sum(1 for c in verification['checks'] if c['ok'])}/{len(verification['checks'])} checks)")
    for check in verification["checks"]:
        if not check["ok"]:
            print("[analyze]   failed: {name}: {detail}".format(**check))
    return 0 if verification["passed"] else 1


def pre_roll_seconds(args: argparse.Namespace) -> float:
    return max(0.0, float(args.pre_roll))


def video_time_for_wall(
    frame_map: Sequence[dict[str, Any]],
    wall_s: float | None,
    fps: float,
    *,
    last: bool = False,
) -> float | None:
    """Video time of the first (or last) recorded frame at or after ``wall_s``.

    The render loop writes one frame per render and ffmpeg plays them at the
    declared fps, so a run slower than real time has fewer frames than wall
    seconds.  Video time is therefore frame index / fps, and the frame map --
    not a subtraction of wall clocks -- is what converts a wall-clock onset into
    a cut point.
    """
    if wall_s is None or fps <= 0.0:
        return None
    selected: int | None = None
    for frame in frame_map:
        timestamp = finite(frame.get("wall_time_ns"))
        index = finite(frame.get("frame"))
        if timestamp is None or index is None:
            continue
        if last:
            if timestamp / 1e9 <= wall_s:
                selected = int(index)
        elif timestamp / 1e9 >= wall_s:
            selected = int(index)
            break
    if selected is None:
        return None
    return selected / fps


if __name__ == "__main__":
    raise SystemExit(main())
