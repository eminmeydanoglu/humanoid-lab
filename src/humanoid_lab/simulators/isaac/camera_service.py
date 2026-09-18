"""Head-camera ZMQ endpoint and headless reset control for the Isaac service.

The simulation loop owns every Isaac API call.  This module only ever moves
bytes: the loop pushes the latest 640x480 RGB frame into a thread-safe buffer,
and two small ZMQ REP services answer requests on their own threads.  Neither
service thread touches the stage, the robot, or the timeline, so a slow or
absent consumer can only ever cost a dropped frame, never a stalled physics
tick.

The camera endpoint answers exactly ``b"get_frame"`` with a three-frame
multipart:

    0. JPEG-encoded RGB (the head camera frame)
    1. IR placeholder (not captured yet)
    2. depth placeholder (not captured yet)

The reset endpoint answers ``b"reset"`` and ``b"status"``.  A reset request only
sets a flag; the simulation loop consumes it at its next boundary, so the reset
itself happens on the loop thread.  Defaults avoid the ports the SONIC and
GR00T side already use (5550, 5555, 5556, 5557): see the CLI help.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

DEFAULT_CAMERA_ENDPOINT = "tcp://*:5558"
DEFAULT_CONTROL_ENDPOINT = "tcp://*:5559"

GET_FRAME_REQUEST = b"get_frame"
RESET_REQUEST = b"reset"
STATUS_REQUEST = b"status"

#: Sent in place of a real frame until those sensors are enabled.  A JSON
#: sentinel rather than an empty frame so a consumer cannot mistake "not
#: captured" for a valid zero-sized image.
IR_PLACEHOLDER = b'{"kind":"ir","available":false}'
DEPTH_PLACEHOLDER = b'{"kind":"depth","available":false}'
ERROR_PREFIX = b"error:"


def encode_jpeg(rgb: Any, quality: int = 85) -> bytes:
    """Encode an HxWx3 uint8 RGB array as JPEG bytes.

    Prefers OpenCV, which the Isaac runtime ships; falls back to Pillow so the
    pure-Python contract stays testable outside the simulator.
    """
    try:
        import cv2
    except ImportError:
        pass
    else:
        # OpenCV reads and writes BGR; reversing keeps a decoded frame equal to
        # the RGB the sensor produced, matching the SONIC ego_view convention.
        ok, encoded = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed to encode the head-camera frame")
        return encoded.tobytes()

    try:
        import io

        from PIL import Image
    except ImportError as exc:  # pragma: no cover - both encoders missing
        raise RuntimeError("no JPEG encoder available (need opencv-python or Pillow)") from exc
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


@dataclass(frozen=True)
class FrameSnapshot:
    """One captured frame and the identity of the episode that produced it."""

    rgb: Any
    width: int
    height: int
    timestamp_s: float
    episode_id: int
    sequence: int


class HeadCameraBuffer:
    """Thread-safe latest-frame buffer with an explicit reset invalidation.

    Only the newest frame is retained: a consumer that falls behind should read
    the present, not replay the past.  ``invalidate`` drops the frame so a reset
    cannot serve a pose from before the robot and cubes were restored.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: FrameSnapshot | None = None
        self._sequence = 0
        self._invalidations = 0

    def publish(self, rgb: Any, *, timestamp_s: float | None = None, episode_id: int = 0) -> FrameSnapshot:
        # The simulator's sensor path may hand back a view onto a buffer it
        # reuses next render; this copy is what makes it safe to read from the
        # endpoint thread afterwards.
        import numpy as np

        rgb = np.array(rgb, copy=True)
        height = int(rgb.shape[0])
        width = int(rgb.shape[1])
        with self._lock:
            self._sequence += 1
            snapshot = FrameSnapshot(
                rgb=rgb,
                width=width,
                height=height,
                timestamp_s=time.monotonic() if timestamp_s is None else float(timestamp_s),
                episode_id=int(episode_id),
                sequence=self._sequence,
            )
            self._snapshot = snapshot
            return snapshot

    def invalidate(self, *, episode_id: int | None = None) -> None:
        """Drop the retained frame; a reset must not serve a stale pose."""
        with self._lock:
            self._snapshot = None
            self._invalidations += 1

    def snapshot(self) -> FrameSnapshot | None:
        with self._lock:
            return self._snapshot

    def frame_age_s(self, *, now: float | None = None) -> float | None:
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return None
        reference = time.monotonic() if now is None else float(now)
        return max(0.0, reference - snapshot.timestamp_s)

    @property
    def invalidations(self) -> int:
        with self._lock:
            return self._invalidations


class _RepServer:
    """A minimal ZMQ REP server on a daemon thread.

    Subclasses implement ``handle``: one request payload in, a list of reply
    frames out.  ZMQ is imported lazily so the pure payload logic can be
    exercised without the transport present.
    """

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.bound_endpoint: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: str | None = None

    @property
    def resolved_endpoint(self) -> str:
        """The endpoint a client can connect to, after a wildcard bind resolves."""
        return self.bound_endpoint or self.endpoint

    def handle(self, payload: bytes) -> list[bytes]:  # pragma: no cover - interface
        raise NotImplementedError

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._serve, name=f"rep{self.endpoint}", daemon=True)
        self._thread.start()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)

    def _serve(self) -> None:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(self.endpoint)
        except zmq.ZMQError as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
            context.term()
            return
        self.bound_endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._ready.set()
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if not poller.poll(timeout=100):
                    continue
                payload = socket.recv()
                try:
                    reply = self.handle(payload)
                except Exception as exc:  # noqa: BLE001 - a bad request must not kill the thread
                    reply = [ERROR_PREFIX + f"{type(exc).__name__}: {exc}".encode()]
                socket.send_multipart(reply)
        finally:
            socket.close()
            context.term()

    @property
    def error(self) -> str | None:
        return self._error

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            self._thread = None


class HeadCameraEndpoint(_RepServer):
    """Serves the latest RGB frame to one requesting consumer at a time."""

    def __init__(self, endpoint: str = DEFAULT_CAMERA_ENDPOINT, buffer: HeadCameraBuffer | None = None) -> None:
        super().__init__(endpoint)
        self.buffer = buffer if buffer is not None else HeadCameraBuffer()
        self._frames_published = 0

    def publish_frame(self, rgb: Any, *, timestamp_s: float | None = None, episode_id: int = 0) -> None:
        self.buffer.publish(rgb, timestamp_s=timestamp_s, episode_id=episode_id)
        self._frames_published += 1

    def invalidate(self, *, episode_id: int | None = None) -> None:
        self.buffer.invalidate(episode_id=episode_id)

    def frame_age_s(self, *, now: float | None = None) -> float | None:
        return self.buffer.frame_age_s(now=now)

    def handle(self, payload: bytes) -> list[bytes]:
        if payload != GET_FRAME_REQUEST:
            return [ERROR_PREFIX + b" expected get_frame"]
        snapshot = self.buffer.snapshot()
        if snapshot is None:
            return [ERROR_PREFIX + b" no frame captured yet"]
        return [encode_jpeg(snapshot.rgb), IR_PLACEHOLDER, DEPTH_PLACEHOLDER]

    def status(self) -> dict[str, Any]:
        snapshot = self.buffer.snapshot()
        return {
            "endpoint": self.endpoint,
            "error": self.error,
            "frames_published": self._frames_published,
            "frame_age_s": self.buffer.frame_age_s(),
            "invalidations": self.buffer.invalidations,
            "episode_id": snapshot.episode_id if snapshot is not None else None,
            "frame_shape": [snapshot.height, snapshot.width, 3] if snapshot is not None else None,
            "has_frame": snapshot is not None,
        }


class ResetControlEndpoint(_RepServer):
    """Accepts ``reset`` and ``status`` without ever calling an Isaac API.

    ``on_reset`` runs on this service thread.  It must only queue work for the
    simulation loop (the service's reset callback sets a flag) and return
    whether the request was accepted.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_CONTROL_ENDPOINT,
        on_reset: Callable[[], bool] | None = None,
        status_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(endpoint)
        self._on_reset = on_reset
        self._status_provider = status_provider

    def handle(self, payload: bytes) -> list[bytes]:
        if payload == RESET_REQUEST:
            accepted = bool(self._on_reset()) if self._on_reset is not None else False
            return [b"reset_queued" if accepted else b"reset_refused"]
        if payload == STATUS_REQUEST:
            provider = self._status_provider
            return [json.dumps(provider() if provider is not None else {}, sort_keys=True).encode()]
        return [ERROR_PREFIX + b" expected reset or status"]


def request_reply(endpoint: str, payload: bytes, *, timeout_ms: int = 2000) -> list[bytes]:
    """Send one request to a REP endpoint and return the multipart reply.

    A small helper for tests and for a later session service; it owns its own
    context and closes it, so it never leaks a socket into the caller.
    """
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(endpoint)
        socket.send(payload)
        if not socket.poll(timeout_ms):
            raise TimeoutError(f"no reply from {endpoint} within {timeout_ms} ms")
        return list(socket.recv_multipart())
    finally:
        socket.close()
        context.term()
