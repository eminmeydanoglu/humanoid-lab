"""The robot camera client must reject stale frames and decode live RGB JPEG."""

from __future__ import annotations

import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from humanoid_lab.psi0_bridge.camera import CameraError, HttpColorCameraClient


@pytest.fixture
def camera_server():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., 0] = 180
    frame[..., 1] = 30
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG")
    jpeg = buffer.getvalue()
    response = {"age_ms": 40, "status": 200, "delay_s": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            time.sleep(response["delay_s"])
            self.send_response(response["status"])
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("X-Frame-Age-Ms", str(response["age_ms"]))
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", response
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_accepts_fresh_rgb_and_rejects_stale_or_unavailable(camera_server):
    endpoint, response = camera_server
    camera = HttpColorCameraClient(endpoint, timeout_ms=1000)
    try:
        frame = camera.fetch()
        assert frame.shape == (480, 640, 3)
        assert frame[200, 200, 0] > frame[200, 200, 1]
        response["age_ms"] = 501
        with pytest.raises(CameraError, match="age"):
            camera.fetch()
        response["status"] = 503
        with pytest.raises(CameraError, match="503"):
            camera.fetch()
        response["status"] = 200
        response["age_ms"] = 40
        response["delay_s"] = 0.52
        with pytest.raises(CameraError, match="age"):
            camera.fetch()
    finally:
        camera.close()
