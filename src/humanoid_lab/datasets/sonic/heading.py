"""Heading-relative anchor orientation, mirroring the pinned C++ deployment.

The deployment's ``motion_anchor_orientation_heading*`` observations compare the
reference root orientation against the *robot's* current base orientation
(``math_utils.hpp``: ``calc_heading_quat_d(base_quat)`` conjugated into the
reference root rotation, then the first two rotation-matrix columns packed
row-wise).  Offline conversion has no recorded robot base, so the caller must
supply — and record — which base quaternion stands in for the live measurement.
This module only implements the deterministic math; policy lives in
``encoder_observation.OrientationPolicy``.

Every helper below is a transcription of the pinned C++
(``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/math_utils.hpp``):
``quat_mul``, ``quat_conjugate``, ``calc_heading_quat`` and
``quat_to_rotation_matrix``.  Array shapes follow numpy broadcasting, so the
same functions serve one frame and a batch of frames.
"""

from __future__ import annotations

import numpy as np

Quaternion = np.ndarray  # (..., 4) in wxyz order


def quat_unit(wxyz: Quaternion) -> Quaternion:
    """Normalize; returns a zero quaternion unchanged instead of dividing by zero."""
    q = np.asarray(wxyz, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.divide(q, norm, out=np.zeros_like(q), where=norm > 1e-12)


def quat_mul(left: Quaternion, right: Quaternion) -> Quaternion:
    """Hamilton product in wxyz order with the pinned C++ sign convention."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_conjugate(wxyz: Quaternion) -> Quaternion:
    q = np.asarray(wxyz, dtype=np.float64).copy()
    q[..., 1:] *= -1.0
    return q


def quat_rotate(wxyz: Quaternion, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion (C++ ``quat_rotate_d``)."""
    q = np.asarray(wxyz, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    w = q[..., 0:1]
    q_vec = q[..., 1:]
    scale_a = 2.0 * w * w - 1.0
    a = v * scale_a
    b = 2.0 * w * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.sum(q_vec * v, axis=-1, keepdims=True)
    return a + b + c


def quat_from_angle_axis(angle: float | np.ndarray, axis: tuple[float, float, float]) -> Quaternion:
    vector = np.asarray(axis, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    theta = np.asarray(angle, dtype=np.float64) / 2.0
    sin_theta = np.sin(theta)[..., None]
    cos_theta = np.cos(theta)[..., None]
    return quat_unit(np.concatenate((cos_theta, sin_theta * vector), axis=-1))


def calc_heading(wxyz: Quaternion) -> np.ndarray:
    """Yaw of the quaternion's rotated x-axis (C++ ``calc_heading_d``)."""
    direction = quat_rotate(wxyz, np.array([1.0, 0.0, 0.0]))
    return np.arctan2(direction[..., 1], direction[..., 0])


def calc_heading_quat(wxyz: Quaternion) -> Quaternion:
    """Yaw-only quaternion about the world z-axis (C++ ``calc_heading_quat_d``)."""
    return quat_from_angle_axis(calc_heading(wxyz), (0.0, 0.0, 1.0))


def quat_to_rotation_matrix(wxyz: Quaternion) -> np.ndarray:
    """Rotation matrix with the same elements and normalization as the pinned C++."""
    q = quat_unit(wxyz)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), axis=-1),
            np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), axis=-1),
            np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )


def rotation_matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    """First two rotation-matrix columns, flattened row-wise exactly as upstream does."""
    return np.asarray(matrix, dtype=np.float64)[..., :, :2].reshape(*np.shape(matrix)[:-2], 6)


def world_quaternion_to_6d(wxyz: Quaternion) -> np.ndarray:
    """Raw world orientation 6D. Diagnostic only; never the encoder contract."""
    return rotation_matrix_to_6d(quat_to_rotation_matrix(wxyz)).astype(np.float32)


def heading_relative_anchor_6d(base_quat: Quaternion, reference_root_quat: Quaternion) -> np.ndarray:
    """One ``motion_anchor_orientation_heading`` sample.

    ``base_quat`` is the quaternion that stands in for the robot's live base;
    ``reference_root_quat`` is the reference root orientation of the future
    frame.  Both are world-frame wxyz.  ``apply_delta_heading`` (the operator
    yaw reset) is identity offline, so it is not a parameter here.
    """
    left = calc_heading_quat(base_quat)
    relative = quat_mul(quat_conjugate(left), np.asarray(reference_root_quat, dtype=np.float64))
    return rotation_matrix_to_6d(quat_to_rotation_matrix(relative)).astype(np.float32)
