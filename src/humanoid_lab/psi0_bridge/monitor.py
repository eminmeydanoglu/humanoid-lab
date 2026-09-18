"""One ZMQ thread that owns the bridge's read sockets.

ZeroMQ sockets are not thread-safe, so every read socket lives here and only
this thread touches them: a SUB on the SONIC ``g1_debug`` topic (``:5557``) and
a REQ client for the camera's ``get_frame`` (``:5558``).  The session loop and
the web UI read cached snapshots instead of opening sockets of their own, so
each endpoint has exactly one consumer and the UI preview is the same frame the
policy consumes.

The camera service owns the ``240x320`` transform; this module never resizes or
crops, and it keeps the native 640x480 frame verbatim.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .camera import CAMERA_ENDPOINT
from .state import StateContractError, build_raw_state
from .state_source import STATE_ENDPOINT, STATE_TOPIC

DEFAULT_POLL_HZ = 30.0
IDLE_POLL_HZ = 5.0
#: One camera round trip; short enough that a stalled camera service cannot hold
#: up the state poll for long, generous enough for a loaded JPEG encode.
CAMERA_TIMEOUT_MS = 250

#: A snapshot older than this is reported as not alive (the SONIC deploy and the
#: Isaac loop both run well above 1 Hz).
ALIVE_MAX_AGE_S = 1.0


@dataclass(frozen=True)
class StateSnapshot:
    """One ``g1_debug`` payload, already validated into the raw 43D vector."""

    payload: Mapping[str, Any]
    raw_state: np.ndarray
    timestamp_s: float


@dataclass(frozen=True)
class FrameSnapshot:
    """One camera frame exactly as the camera service returned it."""

    frame: np.ndarray
    timestamp_s: float


class Monitor:
    """Polls the state and camera endpoints and caches the newest snapshots."""

    def __init__(
        self,
        *,
        state_endpoint: str = STATE_ENDPOINT,
        state_topic: str = STATE_TOPIC,
        camera_endpoint: str = CAMERA_ENDPOINT,
        poll_hz: float = DEFAULT_POLL_HZ,
        idle_poll_hz: float = IDLE_POLL_HZ,
        camera_timeout_ms: int = CAMERA_TIMEOUT_MS,
    ) -> None:
        self.state_endpoint = state_endpoint
        self.state_topic = state_topic
        self.camera_endpoint = camera_endpoint
        self.poll_hz = float(poll_hz)
        self.idle_poll_hz = float(idle_poll_hz)
        self.camera_timeout_ms = int(camera_timeout_ms)

        self._lock = threading.Lock()
        self._state: Optional[StateSnapshot] = None
        self._frame: Optional[FrameSnapshot] = None
        self._state_error: Optional[str] = None
        self._camera_error: Optional[str] = None
        self._state_count = 0
        self._frame_count = 0
        self._camera_reconnects = 0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._active = False
        self._fatal_error: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="psi0-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None

    def set_active(self, active: bool) -> None:
        """Poll at :attr:`poll_hz` while a session runs, :attr:`idle_poll_hz` otherwise."""
        self._active = bool(active)

    def invalidate(self) -> None:
        """Drop the cached state and frame.

        Called on Reset: the robot and scene are about to move, so a ``Start``
        must wait for the next observation instead of reusing one captured
        before the reset.
        """
        with self._lock:
            self._state = None
            self._frame = None

    # -- snapshots ---------------------------------------------------------

    def state(self, *, max_age_s: Optional[float] = None) -> Optional[StateSnapshot]:
        with self._lock:
            snapshot = self._state
        if snapshot is None:
            return None
        if max_age_s is not None and time.monotonic() - snapshot.timestamp_s > max_age_s:
            return None
        return snapshot

    def frame(self, *, max_age_s: Optional[float] = None) -> Optional[FrameSnapshot]:
        with self._lock:
            snapshot = self._frame
        if snapshot is None:
            return None
        if max_age_s is not None and time.monotonic() - snapshot.timestamp_s > max_age_s:
            return None
        return snapshot

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            state = self._state
            frame = self._frame
            state_error = self._state_error
            camera_error = self._camera_error
            state_count = self._state_count
            frame_count = self._frame_count

        state_age = None if state is None else now - state.timestamp_s
        frame_age = None if frame is None else now - frame.timestamp_s
        shape = None if frame is None else list(np.asarray(frame.frame).shape)
        with self._lock:
            reconnects = self._camera_reconnects
        return {
            "error": self._fatal_error,
            "state": {
                "endpoint": self.state_endpoint,
                "topic": self.state_topic,
                "alive": state_age is not None and state_age <= ALIVE_MAX_AGE_S,
                "age_s": state_age,
                "count": state_count,
                "last_error": state_error,
            },
            "camera": {
                "endpoint": self.camera_endpoint,
                "alive": frame_age is not None and frame_age <= ALIVE_MAX_AGE_S,
                "age_s": frame_age,
                "shape": shape,
                "count": frame_count,
                "last_error": camera_error,
                "reconnects": reconnects,
            },
        }

    # -- thread body -------------------------------------------------------

    def _current_hz(self) -> float:
        hz = self.poll_hz if self._active else min(self.idle_poll_hz, self.poll_hz)
        return hz if hz > 0 else 1.0

    def _serve(self) -> None:
        from .camera import CameraClient
        from .state_source import StateSubscriber

        try:
            camera = CameraClient(self.camera_endpoint, timeout_ms=self.camera_timeout_ms)
            state = StateSubscriber(self.state_endpoint, topic=self.state_topic)
        except Exception as exc:  # noqa: BLE001 - report instead of dying silently
            with self._lock:
                self._fatal_error = f"{type(exc).__name__}: {exc}"
            return
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                self._poll_state(state)
                self._poll_camera(camera)
                # Read the state again after the camera: even a camera that is
                # timing out at its full budget cannot stretch the state gap past
                # one camera timeout plus one loop period.  (``count`` in the
                # status therefore counts reads, not distinct published samples.)
                self._poll_state(state)
                delay = (1.0 / self._current_hz()) - (time.monotonic() - started)
                if delay > 0:
                    self._stop.wait(delay)
        finally:
            state.close()
            camera.close()

    def _poll_state(self, state: Any) -> None:
        payload = state.latest()
        if payload is None:
            return
        try:
            raw_state = build_raw_state(payload)
        except StateContractError as exc:
            with self._lock:
                self._state_error = str(exc)
            return
        with self._lock:
            self._state = StateSnapshot(payload=payload, raw_state=raw_state, timestamp_s=time.monotonic())
            self._state_error = None
            self._state_count += 1

    def _poll_camera(self, camera: Any) -> None:
        from .camera import CameraError

        try:
            frame = camera.fetch()
        except CameraError as exc:
            with self._lock:
                self._camera_error = str(exc)
                self._camera_reconnects = camera.reconnects
            return
        with self._lock:
            self._frame = FrameSnapshot(frame=frame, timestamp_s=time.monotonic())
            self._camera_error = None
            self._frame_count += 1
            self._camera_reconnects = camera.reconnects
