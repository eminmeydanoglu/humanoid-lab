"""Temporal smoothing for streamed SMPL joints and root quaternion.

Mirrors SONIC's own live-webcam smoothing (see gear_sonic
examples/live_camera_teleop/soma_to_smpl.py): an exponential moving average
where `weight` is the coefficient on the *history* term. Quaternions are
sign-aligned before interpolation and re-normalized.
"""

from __future__ import annotations

import numpy as np


class EmaSmoother:
    def __init__(self, weight: float = 0.0):
        self.weight = float(weight)
        self._joints = None
        self._quat = None

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0

    def __call__(self, joints: np.ndarray, body_quat: np.ndarray):
        """joints: (24,3) float32, body_quat: (4,) float32 (wxyz or xyzw,
        sign-aligned consistently is what matters)."""
        joints = np.asarray(joints, dtype=np.float32)
        body_quat = np.asarray(body_quat, dtype=np.float32).reshape(4)

        if not self.enabled:
            return joints, body_quat

        if self._joints is None:
            self._joints = joints.copy()
            self._quat = body_quat.copy()
            return joints, body_quat

        w = self.weight
        self._joints = w * self._joints + (1.0 - w) * joints

        q_prev, q_new = self._quat, body_quat.copy()
        if float(np.dot(q_prev, q_new)) < 0.0:  # sign-align before lerp
            q_new = -q_new
        q = w * q_prev + (1.0 - w) * q_new
        self._quat = q / (np.linalg.norm(q) + 1e-8)

        return self._joints.astype(np.float32), self._quat.astype(np.float32)
