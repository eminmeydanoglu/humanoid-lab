"""UI + control API: the canonical-prompt gate, status wiring and the camera preview.

Drives the real FastAPI app with a stub session, so no sockets, GPU or policy
server are involved.  Runs in the psi0 environment (``./dev.sh psi0-tests``);
outside it the FastAPI import is reported as a skip.
"""

from __future__ import annotations

import io
import time
import unittest

import numpy as np

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - host python without the serve group
    TestClient = None

from humanoid_lab.psi0_bridge.contracts import validate_info
from humanoid_lab.psi0_bridge.prompt import CANONICAL_PROMPT
from humanoid_lab.psi0_bridge.session import ERROR, IDLE, RUNNING, STOPPED, SessionError

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
FRAME[:, :, 1] = 200


def server_info():
    return validate_info(
        {
            "run_dir": "/outputs/psi0-unitree-dex3-sonic-v1/finetune/run.2609171455",
            "ckpt_step": 40000,
            "dataset_name": "sonic",
            "transforms": [
                {"name": "resize", "size": [240, 320]},
                {"name": "center_crop", "size": [240, 320]},
            ],
            "expected_keys": {
                "image": {"observation.images.egocentric": "HxWx3 uint8 image array"},
                "state": {"states": "1x45 unnormalized state vector"},
            },
            "observation": {"state_dim": 45, "normalize_state": True},
            "action": {"action_dim": 80, "action_chunk_size": 30, "action_exec_horizon": 30},
            "rtc_enabled": True,
        }
    )


class FakeMonitor:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 3.0) -> None:
        self.stopped = True

    def status(self):
        return {
            "state": {"endpoint": "tcp://localhost:5557", "alive": True, "age_s": 0.1},
            "camera": {"endpoint": "tcp://localhost:5558", "alive": True, "shape": list(FRAME.shape)},
        }


class FakeSession:
    """The exact session surface app.create_app touches."""

    def __init__(self) -> None:
        self.monitor = FakeMonitor()
        self.state = IDLE
        self.error = None
        self.starts = 0
        self.stops = 0
        self.resets = 0
        self.closes = 0
        self.info_fails = False

    def close(self) -> None:
        self.closes += 1
        self.state = STOPPED

    def start(self):
        self.starts += 1
        if self.state == ERROR:
            raise SessionError("session is in ERROR; Reset before Start")
        self.state = RUNNING
        return self.status()

    def stop(self):
        self.stops += 1
        self.state = STOPPED
        return self.status()

    def reset(self):
        self.resets += 1
        self.state = IDLE
        return self.status()

    def status(self):
        return {
            "state": self.state,
            "error": self.error,
            "starting": False,
            "psi0": {"url": "ws://localhost:8014/ws", "connected": self.state == RUNNING,
                     "info": {"action_dim": 80, "action_chunk_size": 30}},
            "sonic_state": self.monitor.status()["state"],
            "camera": self.monitor.status()["camera"],
            "action": {"endpoint": "tcp://*:5556", "last_time": "2026-01-01T00:00:00Z",
                       "last_age_s": 0.2, "last_index": 3, "sent": 4},
        }

    def refresh_info(self):
        if self.info_fails:
            raise RuntimeError("/info unreachable")
        return server_info()

    def preview_frame(self):
        return FRAME, time.monotonic()


@unittest.skipIf(TestClient is None, "fastapi is not installed in this environment")
class UiApiTest(unittest.TestCase):
    def setUp(self) -> None:
        from humanoid_lab.psi0_bridge.app import create_app

        self.session = FakeSession()
        app = create_app(self.session, webrtc={"endpoint": "100.64.0.5:49100", "client": "/opt/client.AppImage"})
        self.client = TestClient(app)
        self.client.__enter__()
        self._closed = False

    def _close(self) -> None:
        if not self._closed:
            self._closed = True
            self.client.__exit__(None, None, None)

    def tearDown(self) -> None:
        self._close()

    def test_page_prefills_the_canonical_prompt_and_controls(self) -> None:
        body = self.client.get("/").text
        self.assertIn(CANONICAL_PROMPT, body)
        self.assertIn("BlockStacking", body)
        for control in ('id="start"', 'id="stop"', 'id="reset"', 'id="preview"', 'id="webrtc"'):
            self.assertIn(control, body)

    def test_meta_carries_the_prompt_and_the_webrtc_link(self) -> None:
        meta = self.client.get("/api/meta").json()
        self.assertEqual(meta["task"], "BlockStacking")
        self.assertEqual(meta["prompt"], CANONICAL_PROMPT)
        self.assertEqual(meta["webrtc"]["endpoint"], "100.64.0.5:49100")

    def test_start_accepts_only_the_exact_canonical_prompt(self) -> None:
        rejected = self.client.post("/api/start", json={"instruction": "Stack the blocks."})
        self.assertEqual(rejected.status_code, 400)
        self.assertIn(CANONICAL_PROMPT, rejected.json()["canonical_prompt"])
        self.assertEqual(self.session.starts, 0)

        # The literal contract: case or whitespace variants are refused too.
        for variant in (CANONICAL_PROMPT.upper(), " " + CANONICAL_PROMPT, CANONICAL_PROMPT + "\n"):
            with self.subTest(variant=variant[:20]):
                response = self.client.post("/api/start", json={"instruction": variant})
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.session.starts, 0)

        accepted = self.client.post("/api/start", json={"instruction": CANONICAL_PROMPT})
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(self.session.starts, 1)
        self.assertEqual(accepted.json()["state"], RUNNING)

    def test_start_reports_a_session_error_as_conflict(self) -> None:
        self.session.state = ERROR
        response = self.client.post("/api/start", json={"instruction": CANONICAL_PROMPT})
        self.assertEqual(response.status_code, 409)
        self.assertIn("Reset", response.json()["detail"])

    def test_stop_and_reset_are_wired_to_the_session(self) -> None:
        self.assertEqual(self.client.post("/api/stop").status_code, 200)
        self.assertEqual(self.session.stops, 1)
        self.assertEqual(self.client.post("/api/reset").status_code, 200)
        self.assertEqual(self.session.resets, 1)

    def test_status_exposes_state_streams_and_last_action(self) -> None:
        status = self.client.get("/api/status").json()
        self.assertEqual(status["state"], IDLE)
        self.assertTrue(status["psi0"]["info"]["action_dim"])
        self.assertTrue(status["sonic_state"]["alive"])
        self.assertTrue(status["camera"]["alive"])
        self.assertEqual(status["action"]["last_index"], 3)
        self.assertEqual(status["action"]["sent"], 4)

    def test_preview_serves_the_same_frame_as_the_policy(self) -> None:
        from PIL import Image

        response = self.client.get("/api/camera/frame")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        decoded = np.asarray(Image.open(io.BytesIO(response.content)))
        self.assertEqual(decoded.shape, FRAME.shape)
        self.assertGreater(int(decoded[:, :, 1].mean()), 150)

    def test_psi0_info_is_surfaced_and_failures_are_502(self) -> None:
        self.assertEqual(self.client.get("/api/psi0/info").json()["info"]["action_dim"], 80)
        self.session.info_fails = True
        self.assertEqual(self.client.get("/api/psi0/info").status_code, 502)

    def test_lifespan_owns_the_monitor_thread(self) -> None:
        self.assertTrue(self.session.monitor.started)
        self._close()
        self.assertTrue(self.session.monitor.stopped)
        self.assertEqual(self.session.state, STOPPED)


if __name__ == "__main__":
    unittest.main()
