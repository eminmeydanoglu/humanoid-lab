"""Publish Isaac head-camera frames in SONIC's native ``ego_view`` format."""

from __future__ import annotations

import struct
from typing import Any

DEFAULT_SONIC_CAMERA_ENDPOINT = "tcp://*:5555"


def _pack(value: Any) -> bytes:
    """Encode the small msgpack subset used by SONIC camera messages."""
    if isinstance(value, dict):
        size = len(value)
        head = bytes([0x80 | size]) if size < 16 else b"\xde" + struct.pack(">H", size)
        return head + b"".join(_pack(key) + _pack(item) for key, item in value.items())
    if isinstance(value, str):
        raw = value.encode()
        if len(raw) < 32:
            head = bytes([0xA0 | len(raw)])
        elif len(raw) < 256:
            head = b"\xd9" + bytes([len(raw)])
        else:
            head = b"\xda" + struct.pack(">H", len(raw))
        return head + raw
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        return b"\xc6" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, float):
        return b"\xcb" + struct.pack(">d", value)
    raise TypeError(f"unsupported msgpack value: {type(value).__name__}")


class SonicCameraPublisher:
    """Non-blocking latest-frame publisher for NVIDIA's SONIC VLA client."""

    def __init__(self, endpoint: str = DEFAULT_SONIC_CAMERA_ENDPOINT) -> None:
        import zmq

        self.endpoint = endpoint
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)

    def publish(self, rgb: Any, timestamp_s: float) -> bool:
        import zmq

        from .camera_service import encode_jpeg

        payload = _pack(
            {
                "timestamps": {"ego_view": float(timestamp_s)},
                "images": {"ego_view": encode_jpeg(rgb)},
            }
        )
        try:
            self._socket.send(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            return False
        return True

    def close(self) -> None:
        self._socket.close()
        self._context.term()
