"""SONIC-compatible camera publisher without adding packages to the Isaac env."""

from __future__ import annotations

import struct
from typing import Any


def _pack(value: Any) -> bytes:
    if isinstance(value, dict):
        size = len(value)
        head = bytes([0x80 | size]) if size < 16 else b"\xde" + struct.pack(">H", size)
        return head + b"".join(_pack(key) + _pack(item) for key, item in value.items())
    if isinstance(value, str):
        raw = value.encode()
        head = bytes([0xA0 | len(raw)]) if len(raw) < 32 else b"\xd9" + bytes([len(raw)])
        return head + raw
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        return b"\xc6" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, float):
        return b"\xcb" + struct.pack(">d", value)
    raise TypeError(f"unsupported msgpack value: {type(value).__name__}")


class SonicCameraPublisher:
    def __init__(self, port: int) -> None:
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{port}")

    def publish(self, rgb: Any, timestamp: float) -> None:
        import cv2
        import zmq

        ok, encoded = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            payload = _pack({"timestamps": {"ego_view": timestamp}, "images": {"ego_view": encoded.tobytes()}})
            try:
                self._socket.send(payload, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

    def close(self) -> None:
        self._socket.close()
        self._context.term()
