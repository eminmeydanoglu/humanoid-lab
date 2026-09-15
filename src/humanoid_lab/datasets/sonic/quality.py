"""Offline numerical QC shared by pilot and bulk conversion."""

from __future__ import annotations

import numpy as np


def trajectory_metrics(values: np.ndarray, fps: float = 50.0) -> dict[str, float | int]:
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
