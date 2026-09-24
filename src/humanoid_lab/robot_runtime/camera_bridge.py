"""Expose one fresh robot RGB frame to NVIDIA's existing GR00T VLA client."""

from __future__ import annotations

import threading
import time

from humanoid_lab.psi0_bridge.camera import encode_jpeg


class GrootCameraBridge:
    """Publish the monitor's camera under SONIC's `ego_view` image key."""

    def __init__(self, monitor, endpoint: str = "tcp://*:5555", fps: float = 15.0):
        self.monitor = monitor
        self.endpoint = endpoint
        self.fps = fps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.sent = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="groot-camera", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _serve(self) -> None:
        import msgpack
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 2)
        try:
            socket.bind(self.endpoint)
            last_stamp = None
            while not self._stop.is_set():
                sample = self.monitor.frame(max_age_s=0.5)
                if sample is not None and sample.timestamp_s != last_stamp:
                    jpeg = encode_jpeg(sample.frame)
                    capture_wall = time.time() - (time.monotonic() - sample.timestamp_s)
                    payload = msgpack.packb({
                        "timestamps": {"ego_view": capture_wall},
                        "images": {"ego_view": jpeg},
                    }, use_bin_type=True)
                    try:
                        socket.send(payload, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        continue
                    last_stamp = sample.timestamp_s
                    self.sent += 1
                self._stop.wait(1.0 / self.fps)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            socket.close(linger=0)
            context.term()
