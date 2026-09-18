"""Camera source: one ZMQ REQ frame per policy observation.

The bridge reads the camera only from ``get_frame`` on the REQ socket at
``tcp://localhost:5558``.  That endpoint has exactly one reply shape: a
three-frame multipart whose first frame is the JPEG-encoded 640x480 RGB head
camera image (the other two are the IR/depth placeholders).  The decoder below
implements that single protocol and fails closed on anything else.

Frames are returned exactly as decoded and are never resized client-side: the
Psi0 server owns the ``resize``/``center_crop`` transform (``/info`` pins it at
240x320), so the policy sees the same pixels either way.
"""

from __future__ import annotations

import io

import numpy as np

JPEG_SOI = b"\xff\xd8"

CAMERA_ENDPOINT = "tcp://localhost:5558"

#: The head-camera contract from the BlockStacking profile: 640x480 RGB.
FRAME_SHAPE = (480, 640, 3)

#: The camera service answers ``get_frame`` with a three-frame multipart.
REPLY_FRAME_COUNT = 3

GET_FRAME_REQUEST = b"get_frame"

ERROR_PREFIX = b"error:"


class CameraError(RuntimeError):
    """The camera service is unreachable or returned something undecodable."""


def _decode_jpeg(raw: bytes) -> np.ndarray:
    from PIL import Image  # Pillow only; cv2 stays out of the import path

    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def decode_camera_reply(reply: list[bytes]) -> np.ndarray:
    """Decode a ``get_frame`` multipart reply into the raw RGB frame.

    The reply must be the documented three-frame multipart; the first frame is
    the JPEG, and the remaining placeholder frames are ignored.  A single-frame
    ``error: ...`` reply and every other shape raise :class:`CameraError`.
    """
    parts = list(reply) if isinstance(reply, (list, tuple)) else [reply]
    if len(parts) == 1 and isinstance(parts[0], (bytes, bytearray)) and parts[0].startswith(ERROR_PREFIX):
        raise CameraError(f"camera service refused the request: {bytes(parts[0]).decode('utf-8', 'replace')}")
    if len(parts) != REPLY_FRAME_COUNT:
        raise CameraError(
            f"camera reply has {len(parts)} frames; expected the three-frame "
            f"JPEG/IR/depth multipart (first bytes: {bytes(parts[0])[:16]!r})"
        )
    jpeg = bytes(parts[0])
    if not jpeg.startswith(JPEG_SOI):
        raise CameraError(
            f"camera reply frame 0 is not a JPEG (first bytes: {jpeg[:16]!r})"
        )

    frame = _decode_jpeg(jpeg)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise CameraError(f"camera JPEG decoded to {frame.dtype} {frame.shape}, expected uint8 HxWx3")
    if frame.shape != FRAME_SHAPE:
        raise CameraError(
            f"camera frame is {frame.shape}, but this bridge is built for the "
            f"640x480 RGB head camera ({FRAME_SHAPE}); the profile and this client disagree"
        )
    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode an HxWx3 uint8 RGB frame for the UI preview.

    Prefers Pillow (already the decode-side dependency) and falls back to OpenCV,
    which the Isaac runtime ships.  The frame is encoded at its native size; the
    preview never resizes what the policy saw.
    """
    array = np.asarray(frame, dtype=np.uint8)
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    import cv2

    ok, encoded = cv2.imencode(".jpg", array[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CameraError("cv2.imencode failed to encode the camera frame")
    return encoded.tobytes()


class CameraClient:
    """REQ client for the ``get_frame`` camera endpoint.

    A ZMQ REQ socket that timed out (or answered an error) is stuck in its
    send/receive state machine and would fail every later request.  This client
    therefore throws the socket away and reconnects on any failed round trip, so
    one slow camera service cannot permanently blind the bridge; the next fetch
    uses a fresh socket.
    """

    def __init__(
        self,
        endpoint: str = CAMERA_ENDPOINT,
        *,
        request: bytes = GET_FRAME_REQUEST,
        timeout_ms: int = 2000,
    ) -> None:
        import zmq

        self.endpoint = endpoint
        self.request = request
        self.reconnects = 0
        self._timeout_ms = int(timeout_ms)
        self._context = zmq.Context()
        self._socket = self._new_socket()

    def _new_socket(self):
        import zmq

        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        socket.connect(self.endpoint)
        return socket

    def _reconnect(self) -> None:
        self._socket.close(linger=0)
        self._socket = self._new_socket()
        self.reconnects += 1

    def fetch(self) -> np.ndarray:
        """Request and decode exactly one frame, used verbatim by the policy."""
        import zmq

        try:
            self._socket.send(self.request)
            reply = self._socket.recv_multipart()
        except zmq.Again as exc:
            self._reconnect()
            raise CameraError(f"camera {self.endpoint} timed out") from exc
        except zmq.ZMQError as exc:
            self._reconnect()
            raise CameraError(f"camera {self.endpoint} error: {exc}") from exc
        return decode_camera_reply(reply)

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()
