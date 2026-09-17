#!/usr/bin/env python3
"""Plain-HTTP MJPEG viewer for unitree_sim_isaaclab camera frames.

Reads the live Isaac Sim camera frames from the shared-memory segments that
tools/shared_memory_utils.MultiImageWriter fills (isaac_{head,left,right}_image_shm)
and re-serves them as multipart/x-mixed-replace. This avoids WebRTC/ICE/TLS
entirely, so it works in any browser over plain http://host:PORT/.

Run inside the dev container with the unitree-sim environment and the project
root on sys.path (see ./dev.sh unitree-cam).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

IMAGES = ("head", "left", "right")

_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>unitree_sim_isaaclab cameras</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;margin:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:8px}
figure{margin:0}figcaption{font-size:12px;opacity:.7}img{width:100%;background:#000}</style>
</head><body>
<h3>unitree_sim_isaaclab &mdash; live cameras</h3>
<div class="grid">
__FIGS__
</div></body></html>
"""


def _page() -> bytes:
    figs = "\n".join(
        f'<figure><img src="/stream/{n}"><figcaption>{n}</figcaption></figure>'
        for n in IMAGES
    )
    return _PAGE.replace("__FIGS__", figs).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    reader = None  # set by main
    lock = threading.Lock()

    def log_message(self, *_a):  # keep the console quiet
        pass

    def _latest_jpeg(self, name: str) -> bytes | None:
        with Handler.lock:
            frame = Handler.reader.read_single_image(name)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else None

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = _page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/stream/"):
            name = self.path.rsplit("/", 1)[-1]
            if name not in IMAGES:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = self._latest_jpeg(name)
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1.0 / 30.0)
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        self.send_error(404)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--project-root", default="/opt/src/unitree-sim")
    args = ap.parse_args()

    sys.path.insert(0, args.project_root)
    from tools.shared_memory_utils import MultiImageReader  # noqa: E402

    Handler.reader = MultiImageReader()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[cam] MJPEG viewer on http://{args.host}:{args.port}/", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
