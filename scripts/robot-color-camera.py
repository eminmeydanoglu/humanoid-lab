#!/usr/bin/env python3
"""Read only the Unitree D435i color stream and serve fresh JPEG frames.

Run with the robot's existing pyrealsense2/Pillow environment. This process
does not import or publish DDS, controller commands, or robot SDK messages.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
import pyrealsense2 as rs


class ColorCamera:
    def __init__(self, fps: int) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.captured_at = 0.0
        self.frames = 0
        self.error: str | None = None
        self.stop = threading.Event()
        self.fps = fps

    def run(self) -> None:
        while not self.stop.is_set():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, self.fps)
            started = False
            try:
                pipeline.start(config)
                started = True
                while not self.stop.is_set():
                    frame = pipeline.wait_for_frames(2000).get_color_frame()
                    if not frame:
                        continue
                    captured_at = time.monotonic()
                    array = np.asanyarray(frame.get_data())
                    if array.shape != (480, 640, 3):
                        raise RuntimeError(f"unexpected color frame shape: {array.shape}")
                    buffer = io.BytesIO()
                    Image.fromarray(array, "RGB").save(buffer, format="JPEG", quality=80)
                    with self.lock:
                        self.jpeg = buffer.getvalue()
                        self.captured_at = captured_at
                        self.frames += 1
                        self.error = None
            except Exception as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.jpeg = None
            finally:
                if started:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
            self.stop.wait(1.0)

    def snapshot(self) -> tuple[bytes | None, float, int, str | None]:
        with self.lock:
            return self.jpeg, self.captured_at, self.frames, self.error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="robot Tailscale IPv4")
    parser.add_argument("--port", type=int, default=8558)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.fps <= 30:
        parser.error("fps must be between 1 and 30")
    camera = ColorCamera(args.fps)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            jpeg, captured_at, frames, error = camera.snapshot()
            age = None if jpeg is None else time.monotonic() - captured_at
            if self.path == "/healthz":
                payload = json.dumps({"frames": frames, "age_s": age, "error": error,
                                      "shape": [480, 640, 3]}).encode()
                self._reply(200, "application/json", payload)
            elif self.path == "/frame/color.jpg":
                if jpeg is None or age is None or age > 0.5:
                    self._reply(503, "text/plain", b"color frame unavailable or stale")
                else:
                    self._reply(200, "image/jpeg", jpeg, age)
            else:
                self._reply(404, "text/plain", b"not found")

        def _reply(self, status: int, content_type: str, payload: bytes,
                   age: float | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            if age is not None:
                self.send_header("X-Frame-Age-Ms", str(round(age * 1000)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    worker = threading.Thread(target=camera.run, name="realsense-color", daemon=True)
    worker.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        camera.stop.set()
        server.server_close()
        worker.join(timeout=3)


if __name__ == "__main__":
    main()
