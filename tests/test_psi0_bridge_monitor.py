"""Monitor freshness and reset invalidation.

The session treats a snapshot older than ``FRAME_MAX_AGE_S``/``STATE_MAX_AGE_S``
as not ready; the stream is 30 Hz, the camera 25 Hz and ``g1_debug`` 200 Hz, so
half a second is more than ten policy periods of slack while still failing closed
long before a frozen frame could drive the robot.

``Monitor.__init__`` opens no sockets (the poll thread does, in ``_serve``), so
the cached snapshots can be injected directly here.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from humanoid_lab.psi0_bridge.monitor import (
    DEFAULT_POLL_HZ,
    CAMERA_TIMEOUT_MS,
    FrameSnapshot,
    Monitor,
    StateSnapshot,
)
from humanoid_lab.psi0_bridge.session import FRAME_MAX_AGE_S, STATE_MAX_AGE_S

RAW_STATE = np.zeros(43, dtype=np.float32)
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class FreshnessContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = Monitor()

    def _inject(self, age_s: float) -> None:
        stamp = time.monotonic() - age_s
        self.monitor._state = StateSnapshot(payload={}, raw_state=RAW_STATE, timestamp_s=stamp)
        self.monitor._frame = FrameSnapshot(frame=FRAME, timestamp_s=stamp)

    def test_the_threshold_is_tight_enough_for_the_closed_loop(self) -> None:
        # Fine for a 30 Hz stream (>= 10 periods of slack)...
        self.assertLessEqual(FRAME_MAX_AGE_S, 0.5)
        self.assertLessEqual(STATE_MAX_AGE_S, 0.5)
        # ... and loose enough that one slow camera round trip is not a failure.
        self.assertGreaterEqual(FRAME_MAX_AGE_S, CAMERA_TIMEOUT_MS / 1000.0)
        self.assertEqual(DEFAULT_POLL_HZ, 30.0)

    def test_snapshots_older_than_half_a_second_are_not_ready(self) -> None:
        self._inject(0.6)
        self.assertIsNone(self.monitor.state(max_age_s=STATE_MAX_AGE_S))
        self.assertIsNone(self.monitor.frame(max_age_s=FRAME_MAX_AGE_S))

    def test_a_just_polled_snapshot_is_ready(self) -> None:
        self._inject(0.1)
        self.assertIsNotNone(self.monitor.state(max_age_s=STATE_MAX_AGE_S))
        self.assertIsNotNone(self.monitor.frame(max_age_s=FRAME_MAX_AGE_S))

    def test_invalidate_drops_both_caches(self) -> None:
        self._inject(0.0)
        self.monitor.invalidate()
        self.assertIsNone(self.monitor.state())
        self.assertIsNone(self.monitor.frame())
        self.assertIsNone(self.monitor.state(max_age_s=STATE_MAX_AGE_S))
        self.assertIsNone(self.monitor.frame(max_age_s=FRAME_MAX_AGE_S))

    def test_status_reports_camera_reconnects(self) -> None:
        status = self.monitor.status()
        self.assertEqual(status["camera"]["reconnects"], 0)
        self.assertFalse(status["camera"]["alive"])


if __name__ == "__main__":
    unittest.main()
