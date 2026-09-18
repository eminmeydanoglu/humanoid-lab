#!/usr/bin/env python3
"""Serve the SONIC ego_view ZMQ stream as a browser MJPEG endpoint."""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


class State:
    frame = b""
    lock = threading.Lock()


def capture(host: str, port: int) -> None:
    client = ComposedCameraClientSensor(server_ip=host, port=port)
    while True:
        message = client.read(blocking=True)
        if not message or not message.get("images"):
            continue
        image = message["images"].get("ego_view")
        if image is None:
            continue
        ok, encoded = cv2.imencode(".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with State.lock:
                State.frame = encoded.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = b'''<!doctype html><title>SONIC ego_view</title>
<style>body{margin:0;background:#111;display:grid;place-items:center;height:100vh}img{max-width:96vw;max-height:96vh}</style>
<img src="/stream.mjpg">'''
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path not in ("/stream.mjpg", "/snapshot.jpg"):
            self.send_error(404)
            return
        if self.path == "/snapshot.jpg":
            with State.lock:
                frame = State.frame
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        previous = b""
        while True:
            with State.lock:
                frame = State.frame
            if frame and frame != previous:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                self.wfile.flush()
                previous = frame
            time.sleep(0.03)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-host", default="localhost")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--http-port", type=int, default=8099)
    args = parser.parse_args()
    threading.Thread(target=capture, args=(args.camera_host, args.camera_port), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.http_port), Handler)
    print(f"SONIC camera viewer: http://0.0.0.0:{args.http_port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
