#!/usr/bin/env python3
"""Measured-motion evidence from the simulator's own tracking Parquet.

The simulator writes one row per controlled physics tick: the command it
applied, the measured joint positions, and the wall clock.  This reads that
file and reports what the *robot* did -- per-joint amplitude, tracking error
against the commanded targets, and the cadence at which the targets changed --
so a run is judged by measured motion, not by the fact that a command was sent.

The negative control is the same reader with ``--expect-no-commands``: a session
without a task must have produced no controlled rows at all.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

#: The Flux/Dex3 arm slice inside the 29 body joints (hardware order), and the
#: dex3 hand order.  Both match the node's own training manifest.
ARM_INDICES = list(range(15, 29))
ARM_NAMES = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
HAND_NAMES = ("thumb0", "thumb1", "thumb2", "middle0", "middle1", "index0", "index1")


def load_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise SystemExit("error: tracking file missing: %s" % path)
    return pq.read_table(path).to_pylist()


def amplitudes(measured: np.ndarray) -> list[float]:
    if measured.size == 0:
        return []
    return (measured.max(axis=0) - measured.min(axis=0)).tolist()


def errors(target: np.ndarray, measured: np.ndarray) -> dict:
    if target.size == 0:
        return {"mean": None, "p95": None, "max": None}
    delta = np.abs(target - measured)
    return {"mean": round(float(delta.mean()), 4),
            "p95": round(float(np.percentile(delta, 95)), 4),
            "max": round(float(delta.max()), 4)}


def command_cadence(rows: list[dict]) -> dict:
    """Cadence of distinct body target vectors, from the simulator's wall clock."""
    times = []
    previous = None
    for row in rows:
        target = tuple(round(value, 6) for value in row["body_target"])
        if target != previous:
            times.append(row["wall_time_ns"] / 1e9)
            previous = target
    if len(times) < 3:
        return {"changes": len(times), "hz": None}
    intervals = [b - a for a, b in zip(times, times[1:])]
    return {"changes": len(times),
            "hz": round((len(times) - 1) / (times[-1] - times[0]), 2),
            "median_interval_ms": round(statistics.median(intervals) * 1e3, 2),
            "max_interval_ms": round(max(intervals) * 1e3, 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None, help="the run's summary JSON")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--min-arm-amplitude", type=float, default=0.02)
    parser.add_argument("--min-moving-arm-joints", type=int, default=1)
    parser.add_argument("--min-moving-hand-joints", type=int, default=1)
    parser.add_argument("--min-control-span-s", type=float, default=5.0)
    parser.add_argument("--max-tracking-error", type=float, default=0.25)
    parser.add_argument("--max-hand-tracking-error", type=float, default=0.4)
    #: Target changes as the *simulator's* control loop sampled them.  The node
    #: itself publishes at 30 Hz (measured at the DDS topic); under this host's
    #: load the simulator advances at RTF ~0.65 and samples the newest target at
    #: roughly 20-30 Hz wall, so this gate allows for that while still catching a
    #: stalled command stream.  Chunk timing is time-indexed and unaffected.
    parser.add_argument("--min-command-hz", type=float, default=20.0)
    parser.add_argument("--expect-no-commands", action="store_true",
                        help="negative control: no controlled rows may exist")
    parser.add_argument("--first-row-after", type=float, default=None,
                        help="unix epoch of StartTask: no controlled row may precede it")
    parser.add_argument("--last-row-before", type=float, default=None,
                        help="unix epoch of StopTask: no controlled row may follow it")
    args = parser.parse_args(argv)

    problems: list[str] = []
    report: dict = {"tracking": str(args.tracking)}
    rows = load_rows(args.tracking)
    report["controlled_rows"] = len(rows)

    if rows:
        first_wall = rows[0]["wall_time_ns"] / 1e9
        last_wall = rows[-1]["wall_time_ns"] / 1e9
        report["first_controlled_wall_unix"] = round(first_wall, 3)
        report["last_controlled_wall_unix"] = round(last_wall, 3)
        if args.first_row_after is not None:
            report["start_wall_unix"] = round(args.first_row_after, 3)
            if first_wall < args.first_row_after - 0.5:
                problems.append(
                    "controlled rows exist %.3fs before StartTask" % (args.first_row_after - first_wall))
        if args.last_row_before is not None:
            report["stop_wall_unix"] = round(args.last_row_before, 3)
            if last_wall > args.last_row_before + 0.5:
                problems.append(
                    "controlled rows exist %.3fs after StopTask" % (last_wall - args.last_row_before))
        report["control_span_s"] = round(last_wall - first_wall, 3)
        if not args.expect_no_commands and last_wall - first_wall < args.min_control_span_s:
            problems.append("control lasted %.2fs, need %.2fs" %
                            (last_wall - first_wall, args.min_control_span_s))

    if args.expect_no_commands:
        if rows:
            problems.append("negative control: %d controlled rows were applied" % len(rows))
        report["result"] = "PASS" if not problems else "FAIL"
    elif len(rows) < args.min_rows:
        problems.append("only %d controlled rows (need %d)" % (len(rows), args.min_rows))
        report["result"] = "FAIL"
    else:
        body_measured = np.asarray([row["body_measured"] for row in rows], dtype=np.float64)
        body_target = np.asarray([row["body_target"] for row in rows], dtype=np.float64)
        arm_measured = body_measured[:, ARM_INDICES]
        arm_target = body_target[:, ARM_INDICES]
        report["arm"] = {
            "amplitude_rad": {name: round(value, 4)
                              for name, value in zip(ARM_NAMES, amplitudes(arm_measured))},
            "tracking_error_rad": errors(arm_target, arm_measured),
        }
        moving = [name for name, value in zip(ARM_NAMES, amplitudes(arm_measured))
                  if value >= args.min_arm_amplitude]
        report["arm"]["moving_joints"] = moving
        if len(moving) < args.min_moving_arm_joints:
            problems.append("only %d arm joints moved at least %.3f rad" %
                            (len(moving), args.min_arm_amplitude))
        if report["arm"]["tracking_error_rad"]["p95"] is None or \
                report["arm"]["tracking_error_rad"]["p95"] > args.max_tracking_error:
            problems.append("arm p95 tracking error %.4f exceeds %.3f" %
                            (report["arm"]["tracking_error_rad"]["p95"] or -1.0, args.max_tracking_error))

        report["hands"] = {}
        for side in ("left", "right"):
            measured = np.asarray([row["%s_hand_measured" % side] for row in rows], dtype=np.float64)
            targets = [row["%s_hand_target" % side] for row in rows]
            target = np.asarray([value if value is not None else [np.nan] * 7 for value in targets],
                                dtype=np.float64)
            amplitude = amplitudes(measured)
            side_report = {
                "amplitude_rad": {name: round(value, 4) for name, value in zip(HAND_NAMES, amplitude)},
                "tracking_error_rad": errors(target, measured),
                "moving_joints": [name for name, value in zip(HAND_NAMES, amplitude)
                                  if value >= args.min_arm_amplitude],
            }
            report["hands"][side] = side_report
            if len(side_report["moving_joints"]) < args.min_moving_hand_joints:
                problems.append("%s hand: only %d joints moved at least %.3f rad" %
                                (side, len(side_report["moving_joints"]), args.min_arm_amplitude))
            hand_error = side_report["tracking_error_rad"]["p95"]
            if hand_error is None or not np.isfinite(hand_error) or hand_error > args.max_hand_tracking_error:
                problems.append("%s hand p95 tracking error %s exceeds %.3f" %
                                (side, hand_error, args.max_hand_tracking_error))

        report["body_reference"] = {
            "indices": ARM_INDICES,
            "names": ARM_NAMES,
        }
        report["command_cadence"] = command_cadence(rows)
        cadence = report["command_cadence"]
        if cadence["hz"] is not None and cadence["hz"] < args.min_command_hz:
            problems.append("command cadence %.2f Hz below %.2f Hz" % (cadence["hz"], args.min_command_hz))

        report["support_released"] = any(not row["support_active"] for row in rows)
        report["wall_span_s"] = round((rows[-1]["wall_time_ns"] - rows[0]["wall_time_ns"]) / 1e9, 3)
        report["sim_span_s"] = round(rows[-1]["sim_s"] - rows[0]["sim_s"], 3)
        report["result"] = "PASS" if not problems else "FAIL"

    if args.metrics and args.metrics.is_file():
        summary = json.loads(args.metrics.read_text(encoding="utf-8"))
        report["run_summary"] = {
            key: summary.get(key)
            for key in ("result", "rejected_commands", "applied_commands", "duration_s")
            if key in summary
        }
        if summary.get("result") != "COMPLETED":
            problems.append("simulator did not complete: %s" % summary.get("result"))
        controller = summary.get("controller")
        if isinstance(controller, dict):
            report["run_summary"]["controller"] = {
                key: controller.get(key)
                for key in ("commands_received", "commands_applied", "stale_polls", "state_published")
                if key in controller
            }
            if not args.expect_no_commands and controller.get("kind") != "flux_dds":
                problems.append("simulator controller is not flux_dds")
            positions = [
                (row["root_x"], row["root_y"], row["root_z"])
                for row in controller.get("command_trace", [])
                if all(key in row for key in ("root_x", "root_y", "root_z"))
            ]
            if positions:
                motion = max(max(point[axis] for point in positions) -
                             min(point[axis] for point in positions) for axis in range(3))
                report["run_summary"]["root_motion_m"] = round(motion, 4)
                if motion > 0.01:
                    problems.append("fixed root moved %.4fm" % motion)
        trace = summary.get("root_z_trace")
        if isinstance(trace, list) and trace:
            report["run_summary"]["root_z_range_m"] = [round(min(trace), 4), round(max(trace), 4)]

    report["problems"] = problems
    report["result"] = "PASS" if not problems else "FAIL"
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
