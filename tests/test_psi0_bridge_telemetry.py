"""The bridge's opt-in rollout telemetry: decoding, recording, failing quiet.

The recorder is the only place the applied Protocol v4 messages and the policy's
raw actions are kept, so a decode that silently mis-reads a payload would put a
wrong action vector in front of whoever analyses the rollout afterwards.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.sonic.protocol_v4 import pack_latent_action_message  # noqa: E402
from humanoid_lab.psi0_bridge.telemetry import (  # noqa: E402
    SessionTelemetry,
    TelemetryError,
    decode_protocol_v4,
)


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DecodeProtocolV4Test(unittest.TestCase):
    def test_a_packed_message_round_trips(self) -> None:
        token = np.linspace(-0.625, 0.625, 64, dtype=np.float32)
        left = np.arange(7, dtype=np.float32) / 10.0
        right = -np.arange(7, dtype=np.float32) / 10.0
        payload = pack_latent_action_message(
            motion_token=token, frame_index=17, left_hand_joints=left, right_hand_joints=right
        )
        decoded = decode_protocol_v4(payload)
        self.assertEqual(decoded["topic"], "pose")
        self.assertEqual(decoded["fields"]["frame_index"], [17])
        np.testing.assert_allclose(decoded["fields"]["token_state"][0], token, rtol=0, atol=0)
        np.testing.assert_allclose(decoded["fields"]["left_hand_joints"][0], left, rtol=0, atol=0)
        np.testing.assert_allclose(decoded["fields"]["right_hand_joints"][0], right, rtol=0, atol=0)

    def test_a_token_only_message_decodes_without_hand_fields(self) -> None:
        payload = pack_latent_action_message(
            motion_token=np.zeros(64, dtype=np.float32), frame_index=0
        )
        decoded = decode_protocol_v4(payload)
        self.assertNotIn("left_hand_joints", decoded["fields"])

    def test_a_foreign_topic_is_refused(self) -> None:
        with self.assertRaises(TelemetryError):
            decode_protocol_v4(b"ctrl" + b"\0" * 1300)

    def test_a_truncated_payload_is_refused(self) -> None:
        with self.assertRaises(TelemetryError):
            decode_protocol_v4(b"pose" + b"{}\0")


class SessionTelemetryTest(unittest.TestCase):
    def test_events_are_one_json_object_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionTelemetry(Path(tmp), session_tag="t")
            recorder.transition("RUNNING", action_dim=80)
            recorder.target_action(np.arange(80, dtype=np.float32), published=True)
            recorder.close()
            rows = [json.loads(line) for line in recorder.events_path.read_text().splitlines()]
        self.assertEqual([row["kind"] for row in rows], ["session", "target_action"])
        self.assertEqual(rows[0]["state"], "RUNNING")
        self.assertEqual(rows[1]["action_dim"], 80)
        self.assertEqual(len(rows[1]["action"]), 80)
        self.assertTrue(rows[1]["published"])
        for row in rows:
            self.assertIsInstance(row["wall_time_ns"], int)
            self.assertEqual(row["session"], "t")

    def test_observations_keep_every_nth_frame_and_hash_them_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionTelemetry(Path(tmp), camera_every=2)
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            for index in range(5):
                recorder.observation(
                    state=np.full(45, index, dtype=np.float32), frame=frame + index * 10
                )
            recorder.close()
            rows = [json.loads(line) for line in recorder.events_path.read_text().splitlines()]
            saved = [row["camera_jpeg"] for row in rows]
            self.assertTrue(Path(saved[0]).is_file())
        self.assertEqual(len(rows), 5)
        self.assertEqual([path is not None for path in saved], [True, False, True, False, True])
        self.assertEqual(rows[0]["state_dim"], 45)
        self.assertEqual(len({row["frame_sha1"] for row in rows}), 5)

    def test_applied_actions_record_the_source_and_the_decoded_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionTelemetry(Path(tmp))
            payload = pack_latent_action_message(
                motion_token=np.zeros(64, dtype=np.float32), frame_index=3
            )
            recorder.applied_action("groot", payload)
            recorder.applied_action("psi", b"pose" + b"junk")
            recorder.close()
            rows = [json.loads(line) for line in recorder.events_path.read_text().splitlines()]
        self.assertEqual(rows[0]["source"], "groot")
        self.assertEqual(rows[0]["fields"]["frame_index"], [3])
        self.assertNotIn("decode_error", rows[0])
        # A payload the recorder cannot decode stays visible as a failed decode
        # instead of disappearing from the stream.
        self.assertIn("decode_error", rows[1])
        self.assertEqual(rows[1]["payload_bytes"], 8)

    def test_a_frame_provider_is_time_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionTelemetry(Path(tmp), frame_interval_s=3600.0)
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            recorder.set_frame_provider(lambda: (frame, 1.0))
            self.assertIsNotNone(recorder.sample_frame("groot"))
            self.assertIsNone(recorder.sample_frame("groot"))
            recorder.close()
            rows = [json.loads(line) for line in recorder.events_path.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "frame")
        self.assertEqual(rows[0]["source"], "groot")

    def test_the_monitors_snapshot_object_is_accepted_as_a_frame(self) -> None:
        # The bridge hands over the monitor's FrameSnapshot (an object with
        # ``.frame``/``.timestamp_s``), not a bare array; treating it as an array
        # once raised inside the action router thread and silently stopped a
        # whole GR00T episode.
        class Snapshot:
            def __init__(self, frame: np.ndarray, timestamp_s: float) -> None:
                self.frame = frame
                self.timestamp_s = timestamp_s

        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionTelemetry(Path(tmp), frame_interval_s=0.0)
            recorder.set_frame_provider(lambda: Snapshot(np.zeros((6, 8, 3), dtype=np.uint8), 12.5))
            record = recorder.sample_frame("groot")
            recorder.close()
        self.assertEqual(record["frame_shape"], [6, 8, 3])
        self.assertAlmostEqual(record["frame_source_time"], 12.5)
        self.assertIsNotNone(record["camera_jpeg"])

    def test_a_non_image_is_recorded_as_an_error_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionTelemetry(Path(tmp))
            record = recorder.frame(object(), source="groot")
            recorder.close()
        self.assertIn("frame_error", record)
        self.assertIsNone(record["camera_jpeg"])

    def test_a_write_failure_disables_the_recorder_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionTelemetry(Path(tmp))
            recorder.close()
            recorder.event("session", state="RUNNING")
            self.assertIsNotNone(recorder.disabled_reason)


if __name__ == "__main__":
    unittest.main()
