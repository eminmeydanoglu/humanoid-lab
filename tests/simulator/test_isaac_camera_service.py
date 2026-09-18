"""Head-camera endpoint contract tests: synthetic RGB, no Isaac, no simulator.

The payload tests exercise the pure request/reply paths directly, so they run
wherever numpy is available.  A ZMQ round-trip test runs when pyzmq is present
and is skipped (not failed) in interpreters without it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.camera_service import (  # noqa: E402
    DEPTH_PLACEHOLDER,
    GET_FRAME_REQUEST,
    IR_PLACEHOLDER,
    RESET_REQUEST,
    STATUS_REQUEST,
    HeadCameraBuffer,
    HeadCameraEndpoint,
    ResetControlEndpoint,
    request_reply,
)

_ZMQ = importlib.util.find_spec("zmq") is not None


def synthetic_rgb(width: int = 640, height: int = 480, offset: int = 0) -> np.ndarray:
    """A deterministic 640x480 gradient that is not a solid color."""
    x = np.arange(width, dtype=np.int32)[None, :]
    y = np.arange(height, dtype=np.int32)[:, None]
    red = np.broadcast_to((x + offset) % 256, (height, width)).astype(np.uint8)
    green = np.broadcast_to(y % 256, (height, width)).astype(np.uint8)
    blue = np.broadcast_to((x // 2 + offset) % 256, (height, width)).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


class HeadCameraBufferTests(unittest.TestCase):
    def test_publish_keeps_latest_frame_and_reports_its_age(self) -> None:
        buffer = HeadCameraBuffer()
        self.assertIsNone(buffer.snapshot())
        self.assertIsNone(buffer.frame_age_s())

        buffer.publish(synthetic_rgb(offset=0), timestamp_s=100.0, episode_id=3)
        first = buffer.snapshot()
        buffer.publish(synthetic_rgb(offset=40), timestamp_s=101.0, episode_id=3)
        second = buffer.snapshot()

        assert first is not None and second is not None
        self.assertEqual((second.height, second.width), (480, 640))
        self.assertEqual(second.episode_id, 3)
        self.assertGreater(second.sequence, first.sequence)
        self.assertAlmostEqual(buffer.frame_age_s(now=102.5), 1.5)

    def test_a_reset_invalidation_drops_the_stale_frame(self) -> None:
        buffer = HeadCameraBuffer()
        buffer.publish(synthetic_rgb(), timestamp_s=1.0, episode_id=0)
        self.assertIsNotNone(buffer.snapshot())

        buffer.invalidate(episode_id=1)

        self.assertIsNone(buffer.snapshot())
        self.assertIsNone(buffer.frame_age_s())
        self.assertEqual(buffer.invalidations, 1)


class CameraEndpointPayloadTests(unittest.TestCase):
    def test_get_frame_returns_jpeg_rgb_plus_ir_and_depth_placeholders(self) -> None:
        endpoint = HeadCameraEndpoint("inproc://unused")
        source = synthetic_rgb(offset=10)
        endpoint.publish_frame(source, timestamp_s=time.monotonic(), episode_id=0)

        reply = endpoint.handle(GET_FRAME_REQUEST)

        self.assertEqual(len(reply), 3)
        self.assertEqual(reply[1], IR_PLACEHOLDER)
        self.assertEqual(reply[2], DEPTH_PLACEHOLDER)

        import cv2

        decoded = cv2.imdecode(np.frombuffer(reply[0], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (480, 640, 3))
        rgb = decoded[..., ::-1].astype(np.int16)
        self.assertLess(float(np.abs(rgb - source.astype(np.int16)).mean()), 8.0)

    def test_only_get_frame_is_answered_and_an_empty_buffer_is_an_error(self) -> None:
        endpoint = HeadCameraEndpoint("inproc://unused")

        empty = endpoint.handle(GET_FRAME_REQUEST)
        self.assertEqual(len(empty), 1)
        self.assertTrue(empty[0].startswith(b"error:"))

        self.assertTrue(endpoint.handle(b"get_frames")[0].startswith(b"error:"))

    def test_status_reports_frame_age_and_shape(self) -> None:
        endpoint = HeadCameraEndpoint("inproc://unused")
        idle = endpoint.status()
        self.assertFalse(idle["has_frame"])
        self.assertIsNone(idle["frame_age_s"])

        endpoint.publish_frame(synthetic_rgb(), timestamp_s=time.monotonic(), episode_id=5)
        status = endpoint.status()
        self.assertTrue(status["has_frame"])
        self.assertEqual(status["episode_id"], 5)
        self.assertEqual(status["frame_shape"], [480, 640, 3])
        self.assertGreaterEqual(status["frame_age_s"], 0.0)


class ResetControlPayloadTests(unittest.TestCase):
    def test_reset_and_status_requests_are_handled(self) -> None:
        accepted = {"value": True}
        calls: list[str] = []

        def on_reset() -> bool:
            calls.append("reset")
            return accepted["value"]

        endpoint = ResetControlEndpoint(
            "inproc://unused", on_reset=on_reset, status_provider=lambda: {"physics_tick": 7}
        )

        self.assertEqual(endpoint.handle(RESET_REQUEST), [b"reset_queued"])
        self.assertEqual(calls, ["reset"])

        accepted["value"] = False
        self.assertEqual(endpoint.handle(RESET_REQUEST), [b"reset_refused"])
        self.assertEqual(json.loads(endpoint.handle(STATUS_REQUEST)[0]), {"physics_tick": 7})
        self.assertTrue(endpoint.handle(b"nope")[0].startswith(b"error:"))


@unittest.skipUnless(_ZMQ, "pyzmq is not importable in this interpreter")
class CameraEndpointRoundTripTests(unittest.TestCase):
    def test_zmq_round_trip_serves_frames_and_invalidates_on_reset(self) -> None:
        endpoint = HeadCameraEndpoint("tcp://127.0.0.1:*")
        endpoint.start()
        self.assertTrue(endpoint.wait_ready())
        self.assertIsNone(endpoint.error)
        try:
            endpoint.publish_frame(synthetic_rgb(offset=20), timestamp_s=time.monotonic(), episode_id=7)
            reply = request_reply(endpoint.resolved_endpoint, GET_FRAME_REQUEST)
            self.assertEqual(len(reply), 3)
            self.assertEqual(reply[1], IR_PLACEHOLDER)
            self.assertEqual(reply[2], DEPTH_PLACEHOLDER)

            endpoint.invalidate(episode_id=8)
            after_reset = request_reply(endpoint.resolved_endpoint, GET_FRAME_REQUEST)
            self.assertTrue(after_reset[0].startswith(b"error:"))
        finally:
            endpoint.stop()

    def test_zmq_control_endpoint_queues_a_reset(self) -> None:
        queued: list[str] = []
        control = ResetControlEndpoint(
            "tcp://127.0.0.1:*",
            on_reset=lambda: (queued.append("reset"), True)[1],
            status_provider=lambda: {"physics_tick": 11},
        )
        control.start()
        self.assertTrue(control.wait_ready())
        self.assertIsNone(control.error)
        try:
            self.assertEqual(request_reply(control.resolved_endpoint, RESET_REQUEST), [b"reset_queued"])
            self.assertEqual(queued, ["reset"])
            status = json.loads(request_reply(control.resolved_endpoint, STATUS_REQUEST)[0])
            self.assertEqual(status, {"physics_tick": 11})
        finally:
            control.stop()


if __name__ == "__main__":
    unittest.main()
