"""Camera endpoint: the ``get_frame`` multipart decodes to the raw 640x480 RGB frame.

The bridge reads frames verbatim; the Psi0 server owns the 240x320 transform.
Only the real protocol is accepted, so a mis-shaped reply fails closed instead
of feeding the policy something the camera never produced.
"""

from __future__ import annotations

import io
import threading
import time
import unittest

import numpy as np
from PIL import Image

from humanoid_lab.psi0_bridge.camera import (
    FRAME_SHAPE,
    CameraClient,
    CameraError,
    decode_camera_reply,
    encode_jpeg,
)

try:
    import zmq
except ImportError:  # pragma: no cover - the psi0 environment ships pyzmq
    zmq = None

#: The two placeholders the Isaac camera service appends to every reply.
IR_PLACEHOLDER = b'{"kind":"ir","available":false}'
DEPTH_PLACEHOLDER = b'{"kind":"depth","available":false}'


def jpeg_bytes(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def synthetic_frame() -> np.ndarray:
    """640x480 with distinct quadrants; a channel or row flip would move them."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:240, :320] = (255, 0, 0)
    frame[:240, 320:] = (0, 255, 0)
    frame[240:, :320] = (0, 0, 255)
    frame[240:, 320:] = (255, 255, 255)
    return frame


class CameraRoundTripTest(unittest.TestCase):
    def test_multipart_reply_decodes_to_640x480_rgb(self) -> None:
        reply = [jpeg_bytes(synthetic_frame()), IR_PLACEHOLDER, DEPTH_PLACEHOLDER]
        frame = decode_camera_reply(reply)
        self.assertEqual(frame.shape, FRAME_SHAPE)
        self.assertEqual(frame.dtype, np.uint8)
        # JPEG is lossy, so compare each quadrant's dominant channel.
        self.assertGreater(int(frame[20:220, 20:300, 0].mean()), 200)   # red
        self.assertGreater(int(frame[20:220, 340:620, 1].mean()), 200)  # green
        self.assertGreater(int(frame[260:460, 20:300, 2].mean()), 200)  # blue
        self.assertGreater(int(frame[260:460, 340:620].min()), 200)     # white

    def test_placeholders_are_never_decoded_as_the_policy_frame(self) -> None:
        # A JPEG in the second slot is not the protocol; fail instead of guessing.
        reply = [IR_PLACEHOLDER, jpeg_bytes(synthetic_frame()), DEPTH_PLACEHOLDER]
        with self.assertRaises(CameraError):
            decode_camera_reply(reply)

    def test_error_reply_reports_the_camera_message(self) -> None:
        with self.assertRaises(CameraError) as ctx:
            decode_camera_reply([b"error: no frame captured yet"])
        self.assertIn("no frame captured yet", str(ctx.exception))

    def test_wrong_frame_count_is_rejected(self) -> None:
        for reply in ([jpeg_bytes(synthetic_frame())], [jpeg_bytes(synthetic_frame()), IR_PLACEHOLDER]):
            with self.subTest(count=len(reply)):
                with self.assertRaises(CameraError):
                    decode_camera_reply(reply)

    def test_non_jpeg_first_frame_is_rejected(self) -> None:
        with self.assertRaises(CameraError):
            decode_camera_reply([b"not an image", IR_PLACEHOLDER, DEPTH_PLACEHOLDER])

    def test_a_different_resolution_is_rejected_not_resized(self) -> None:
        small = np.zeros((240, 320, 3), dtype=np.uint8)
        with self.assertRaises(CameraError) as ctx:
            decode_camera_reply([jpeg_bytes(small), IR_PLACEHOLDER, DEPTH_PLACEHOLDER])
        self.assertIn("640x480", str(ctx.exception))


class PreviewEncodingTest(unittest.TestCase):
    def test_ui_preview_keeps_the_frame_size_and_orientation(self) -> None:
        frame = synthetic_frame()
        decoded = np.asarray(Image.open(io.BytesIO(encode_jpeg(frame))))
        self.assertEqual(decoded.shape, FRAME_SHAPE)
        self.assertGreater(int(decoded[:200, :300, 0].mean()), 200)
        self.assertGreater(int(decoded[260:, 340:, 1].mean()), 200)


@unittest.skipIf(zmq is None, "pyzmq is not installed in this environment")
class CameraSocketRecoveryTest(unittest.TestCase):
    """A stalled camera must not leave the REQ socket permanently unusable."""

    def _serve(self, stop: threading.Event, endpoint_ready: threading.Event, stall_first: bool) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind("tcp://127.0.0.1:*")
        self.endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        endpoint_ready.set()
        stalled = False
        try:
            while not stop.is_set():
                if not socket.poll(50):
                    continue
                socket.recv()
                if stall_first and not stalled:
                    stalled = True
                    time.sleep(0.4)  # longer than the client's 200 ms deadline
                socket.send_multipart([jpeg_bytes(synthetic_frame()), IR_PLACEHOLDER, DEPTH_PLACEHOLDER])
        finally:
            socket.close(linger=0)
            context.term()

    def _client(self, *, stall_first: bool):
        stop = threading.Event()
        endpoint_ready = threading.Event()
        thread = threading.Thread(target=self._serve, args=(stop, endpoint_ready, stall_first), daemon=True)
        thread.start()
        self.assertTrue(endpoint_ready.wait(2.0))
        client = CameraClient(self.endpoint, timeout_ms=200)
        return client, stop, thread

    def test_a_timeout_reconnects_instead_of_wedging_the_socket(self) -> None:
        client, stop, _thread = self._client(stall_first=True)
        try:
            with self.assertRaises(CameraError) as ctx:
                client.fetch()
            self.assertIn("timed out", str(ctx.exception))
            self.assertGreaterEqual(client.reconnects, 1)

            # The slow reply lands on the discarded socket; once the camera
            # service is idle again the next fetch must use the fresh socket and
            # return a real frame (the old socket would still be in EFSM).
            time.sleep(0.45)
            frame = client.fetch()
            self.assertEqual(frame.shape, FRAME_SHAPE)
        finally:
            client.close()
            stop.set()

    def test_a_missed_first_frame_is_not_replayed_as_a_later_frame(self) -> None:
        client, stop, _thread = self._client(stall_first=False)
        try:
            first = client.fetch()
            second = client.fetch()
            self.assertEqual(client.reconnects, 0)
            np.testing.assert_array_equal(first, second)
        finally:
            client.close()
            stop.set()


if __name__ == "__main__":
    unittest.main()
