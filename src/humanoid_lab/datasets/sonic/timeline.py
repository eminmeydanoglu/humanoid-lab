"""Timestamp-based resampling and trajectory derivatives."""

from __future__ import annotations

import numpy as np

from .schema import PROCESSED_FPS


def uniform_timeline(timestamps: np.ndarray, fps: float = PROCESSED_FPS) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.ndim != 1 or len(ts) == 0 or not np.isfinite(ts).all() or (len(ts) > 1 and not np.all(np.diff(ts) > 0)):
        raise ValueError("source timestamps must be finite and strictly increasing")
    count = int(np.floor((ts[-1] - ts[0]) * fps + 1e-9)) + 1
    return ts[0] + np.arange(count, dtype=np.float64) / fps


def linear_resample(timestamps: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64)
    data = np.asarray(values)
    dst = np.asarray(target, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != len(ts):
        raise ValueError("values must be [frames, channels]")
    return np.stack([np.interp(dst, ts, data[:, index]) for index in range(data.shape[1])], axis=1).astype(data.dtype)


def finite_difference(positions: np.ndarray, fps: float = PROCESSED_FPS) -> np.ndarray:
    q = np.asarray(positions, dtype=np.float64)
    if q.ndim != 2 or len(q) == 0:
        raise ValueError("positions must be non-empty [frames, joints]")
    velocity = np.zeros_like(q)
    if len(q) > 1:
        velocity[:-1] = np.diff(q, axis=0) * fps
        # Match SONIC's 30→50 Hz planner: the terminal reference keeps the
        # previous finite-difference velocity instead of introducing a false
        # zero-velocity step into every clamped future window at the clip tail.
        velocity[-1] = velocity[-2]
    return velocity.astype(np.asarray(positions).dtype)
