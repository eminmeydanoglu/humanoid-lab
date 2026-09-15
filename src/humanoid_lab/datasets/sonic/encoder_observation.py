"""Exact 1751D G1 observation layout for the pinned SONIC v1.1 encoder."""

from __future__ import annotations

import numpy as np

from .schema import CanonicalEpisode

ENCODER_INPUT_DIM = 1751
FUTURE_OFFSETS = np.arange(0, 46, 5, dtype=np.int64)


def quaternion_to_rotation_6d(wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    # First two rotation-matrix columns, flattened row-wise as upstream does.
    return np.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z),
        2 * (x*z - y*w), 2 * (y*z + x*w),
    ), axis=-1).astype(np.float32)


def build_g1_encoder_observation(episode: CanonicalEpisode) -> tuple[np.ndarray, np.ndarray]:
    episode.validate()
    n = len(episode.timestamps)
    base = np.arange(n, dtype=np.int64)[:, None]
    unclamped = base + FUTURE_OFFSETS[None, :]
    indices = np.minimum(unclamped, n - 1)
    output = np.zeros((n, ENCODER_INPUT_DIM), dtype=np.float32)
    # [0:4] stays [0,0,0,0] for G1 mode id 0.
    output[:, 4:294] = episode.joint_pos[indices].reshape(n, 290)
    output[:, 294:584] = episode.joint_vel[indices].reshape(n, 290)
    output[:, 584:644] = quaternion_to_rotation_6d(episode.body_quat_wxyz)[indices].reshape(n, 60)
    clamp_fraction = np.mean(unclamped >= n, axis=1).astype(np.float32)
    return output, clamp_fraction
