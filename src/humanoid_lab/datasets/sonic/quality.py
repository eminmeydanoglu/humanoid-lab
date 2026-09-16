"""Offline numerical QC shared by pilot and bulk conversion.

Every check returns plain numbers plus a threshold decision, so a pilot report can
show per-joint error statistics next to the exact rule that produced its
PASS/FAIL verdict instead of an unexplained verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .joints import hand_order
from .schema import HAND_DIM

#: Guard rails for the 50 Hz canonical reference.  They are deliberately loose:
#: they exist to catch broken indexing (a joint column fed by the wrong source),
#: not to judge an individual demonstration.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "finite.non_finite_count": 0.0,
    "range.violation_count": 0.0,
    "velocity.abs_p99": 25.0,
    "acceleration.abs_p99": 800.0,
    "jerk.abs_p99": 60000.0,
    "source_echo.max_abs_error_rad": 1e-5,
    "synthetic.max_abs_error_rad": 1e-6,
    "future_clamp.mean_fraction": 0.6,
}


def load_joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    limits = {
        name: (float(spec["lower"]), float(spec["upper"]))
        for name, spec in document["joints"].items()
    }
    if not limits:
        raise ValueError(f"{path} declares no joint limits")
    return limits


def finite_report(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    per_channel = (~finite).sum(axis=0) if data.ndim > 1 else np.array([(~finite).sum()])
    return {
        "non_finite_count": int((~finite).sum()),
        "worst_channel_index": int(np.argmax(per_channel)) if per_channel.size else 0,
        "samples": int(finite.size),
    }


def range_report(values: np.ndarray, joint_names: tuple[str, ...], limits: dict[str, tuple[float, float]]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(joint_names):
        raise ValueError("range check needs [frames, joints] and one name per joint")
    violations: dict[str, dict[str, float]] = {}
    worst = 0.0
    for index, name in enumerate(joint_names):
        if name not in limits:
            raise ValueError(f"joint limits are missing {name!r}")
        lower, upper = limits[name]
        excess = np.maximum(np.maximum(lower - data[:, index], data[:, index] - upper), 0.0)
        if excess.max() > 0.0:
            violations[name] = {
                "max_excess_rad": float(excess.max()),
                "frames": int(np.count_nonzero(excess > 0.0)),
                "lower": lower,
                "upper": upper,
            }
            worst = max(worst, float(excess.max()))
    return {
        "violation_count": int(sum(item["frames"] for item in violations.values())),
        "violating_joints": violations,
        "max_excess_rad": worst,
    }


def derivative_metrics(values: np.ndarray, fps: float = 50.0) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("derivative metrics need a non-empty [frames, channels] array")
    if not np.isfinite(data).all():
        raise ValueError("derivative metrics need finite values")
    velocity = np.diff(data, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps

    def stats(array: np.ndarray, label: str) -> dict[str, float]:
        if array.size == 0:
            return {f"{label}_abs_p95": 0.0, f"{label}_abs_p99": 0.0, f"{label}_abs_max": 0.0}
        magnitude = np.abs(array)
        return {
            f"{label}_abs_p95": float(np.percentile(magnitude, 95)),
            f"{label}_abs_p99": float(np.percentile(magnitude, 99)),
            f"{label}_abs_max": float(magnitude.max()),
        }

    result: dict[str, Any] = {}
    result.update(stats(velocity, "velocity"))
    result.update(stats(acceleration, "acceleration"))
    result.update(stats(jerk, "jerk"))
    return result


def trajectory_metrics(values: np.ndarray, fps: float = 50.0) -> dict[str, float | int]:
    """Legacy summary kept for callers that only need counts and p99s."""
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    safe = np.where(finite, data, 0.0)
    velocity = np.diff(safe, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    percentile = lambda array: float(np.percentile(np.abs(array), 99)) if array.size else 0.0
    return {
        "non_finite_count": int((~finite).sum()),
        "velocity_abs_p99": percentile(velocity),
        "acceleration_abs_p99": percentile(acceleration),
        "jerk_abs_p99": percentile(jerk),
    }


def joint_error_metrics(target: np.ndarray, measured: np.ndarray, joint_names: tuple[str, ...]) -> dict[str, Any]:
    """Per-joint MAE/p95/max plus the aggregate, for target-vs-measured traces."""
    reference = np.asarray(target, dtype=np.float64)
    actual = np.asarray(measured, dtype=np.float64)
    if reference.shape != actual.shape or reference.ndim != 2:
        raise ValueError("per-joint metrics need two equally shaped [frames, joints] arrays")
    if reference.shape[1] != len(joint_names):
        raise ValueError("per-joint metrics need one name per column")
    error = np.abs(reference - actual)
    per_joint = {
        name: {
            "mae": float(error[:, index].mean()),
            "p95": float(np.percentile(error[:, index], 95)),
            "max": float(error[:, index].max()),
        }
        for index, name in enumerate(joint_names)
    }
    return {
        "per_joint": per_joint,
        "overall": {
            "mae": float(error.mean()),
            "p95": float(np.percentile(error, 95)),
            "max": float(error.max()),
        },
        "frames": int(error.shape[0]),
    }


def resampling_report(
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
    target_timestamps: np.ndarray,
    target_values: np.ndarray,
    joint_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compare the stored canonical channels against a direct interpolation.

    Channels that only ever get resampled must match the direct interpolation to
    floating-point noise; a larger error means the stored trajectory was composed
    from something else.
    """
    source_t = np.asarray(source_timestamps, dtype=np.float64)
    target_t = np.asarray(target_timestamps, dtype=np.float64)
    source = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target channel counts differ")
    direct = np.stack(
        [np.interp(target_t, source_t, source[:, index]) for index in range(source.shape[1])], axis=1
    )
    error = np.abs(direct - target)
    per_joint = {
        name: float(error[:, index].max()) for index, name in enumerate(joint_names)
    }
    return {
        "max_abs_error_rad": float(error.max()),
        "mean_abs_error_rad": float(error.mean()),
        "per_joint_max_abs_error_rad": per_joint,
        "worst_joint": joint_names[int(np.argmax(error.max(axis=0)))],
        "source_frames": int(source.shape[0]),
        "target_frames": int(target.shape[0]),
    }


def clamp_report(clamp_fraction: np.ndarray) -> dict[str, float]:
    clamp = np.asarray(clamp_fraction, dtype=np.float64)
    if clamp.ndim != 1 or len(clamp) == 0:
        raise ValueError("clamp fractions must be a non-empty vector")
    return {
        "mean_fraction": float(clamp.mean()),
        "max_fraction": float(clamp.max()),
        "rows_with_clamp": int(np.count_nonzero(clamp > 0.0)),
    }


def name_round_trip_report(source_names: tuple[str, ...], canonical_names: tuple[str, ...]) -> dict[str, Any]:
    """Check that a name-based permutation is a bijection between vocabularies."""
    if sorted(source_names) != sorted(canonical_names):
        missing = sorted(set(canonical_names) - set(source_names))
        extra = sorted(set(source_names) - set(canonical_names))
        return {"ok": False, "missing": missing, "extra": extra}
    return {"ok": True, "missing": [], "extra": [], "joints": len(canonical_names)}


def permutation_round_trip(source_names: tuple[str, ...], target_names: tuple[str, ...]) -> dict[str, Any]:
    """Verify the name permutation and its inverse over a marker vector.

    Reordering values from ``source_names`` into ``target_names`` and back must
    reproduce the marker vector exactly, and each target name must resolve to a
    distinct source index.  This is the check that catches a silently wrong
    joint order, which shape validation cannot see.
    """
    if len(set(source_names)) != len(source_names):
        return {"ok": False, "reason": "duplicate source names"}
    missing = [name for name in target_names if name not in source_names]
    if missing:
        return {"ok": False, "reason": "missing target names", "missing": missing}
    permutation = [source_names.index(name) for name in target_names]
    if len(set(permutation)) != len(permutation):
        return {"ok": False, "reason": "permutation is not injective"}
    marker = np.arange(len(source_names), dtype=np.float64)[None, :]
    forward = marker[..., permutation]
    inverse = np.argsort(np.array(permutation, dtype=np.int64))
    return {
        "ok": bool(np.array_equal(forward[..., inverse], marker)),
        "joints": len(target_names),
        "identity": permutation == list(range(len(permutation))),
    }


def hand_range_report(
    left: np.ndarray,
    right: np.ndarray,
    limits: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Range check the Dex3 hand channels, per side and per joint.

    The body range check does not cover the hands, so before this existed a hand
    trajectory could sit arbitrarily far outside the modelled Dex3 range and the
    episode still reported ``range.violation_count = 0``.  The report is
    informational (a recorded source may legitimately use a different Dex3
    hardware revision whose stroke is wider than the pinned model's); callers
    decide with :func:`hand_range_decision` whether that is tolerable.
    """
    report: dict[str, Any] = {"sides": {}, "max_excess_rad": 0.0, "violating_channels": 0}
    worst = 0.0
    violating = 0
    for side, values in (("left", left), ("right", right)):
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != HAND_DIM:
            raise ValueError("hand range check needs [frames, 7] per side")
        channels: dict[str, dict[str, float]] = {}
        side_worst = 0.0
        for index, name in enumerate(hand_order(side)):
            if name not in limits:
                raise ValueError(f"joint limits are missing {name!r}")
            lower, upper = limits[name]
            excess = np.maximum(np.maximum(lower - data[:, index], data[:, index] - upper), 0.0)
            if excess.max() > 0.0:
                channels[name] = {
                    "max_excess_rad": float(excess.max()),
                    "frames": int(np.count_nonzero(excess > 0.0)),
                    "observed": [float(data[:, index].min()), float(data[:, index].max())],
                    "limits": [float(lower), float(upper)],
                }
        report["sides"][side] = {
            "channels": channels,
            "violating_channels": len(channels),
            "max_excess_rad": side_worst if channels else 0.0,
        }
        if channels:
            report["sides"][side]["max_excess_rad"] = max(item["max_excess_rad"] for item in channels.values())
            side_worst = report["sides"][side]["max_excess_rad"]
        violating += len(channels)
        worst = max(worst, side_worst)
    report["max_excess_rad"] = worst
    report["violating_channels"] = violating
    return report


def hand_range_decision(
    report: dict[str, Any],
    *,
    allowed_channels: int,
    excess_tolerance_rad: float = 0.0,
) -> dict[str, Any]:
    """Decide whether a hand range report is tolerable for a given source.

    ``allowed_channels`` and ``excess_tolerance_rad`` are a source-specific
    policy: a source known to be recorded on a wider-stroke Dex3 revision
    declares how many channels may sit outside the pinned model, and by how much,
    before the trajectory is treated as a mapping error instead of a known
    hardware difference.  The tolerance is what separates "the same systematic
    revision delta" from "a channel that is simply wrong": a source allowed 8
    channels with a 0.35 rad tolerance still fails if any channel deviates by
    more than the revision delta.
    """
    excess = float(report.get("max_excess_rad", 0.0))
    observed = int(report.get("violating_channels", 0))
    within = observed <= allowed_channels and excess <= excess_tolerance_rad + 1e-9
    return {
        "metric": "hand_range.violating_channels",
        "value": float(observed),
        "limit": float(allowed_channels),
        "max_excess_rad": excess,
        "excess_tolerance_rad": float(excess_tolerance_rad),
        "result": "PASS" if within else "FAIL",
    }


def evaluate_thresholds(values: dict[str, float], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    """Decide PASS/FAIL per threshold; a missing value is UNVERIFIED, never PASS."""
    decisions: list[dict[str, Any]] = []
    for name, limit in thresholds.items():
        value = values.get(name)
        if value is None:
            decisions.append({"metric": name, "value": None, "limit": limit, "result": "UNVERIFIED"})
            continue
        decisions.append(
            {
                "metric": name,
                "value": float(value),
                "limit": float(limit),
                "result": "PASS" if float(value) <= float(limit) else "FAIL",
            }
        )
    return decisions


def overall_result(decisions: list[dict[str, Any]]) -> str:
    if any(item["result"] == "FAIL" for item in decisions):
        return "FAIL"
    if any(item["result"] == "UNVERIFIED" for item in decisions):
        return "UNVERIFIED"
    return "PASS"
