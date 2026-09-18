"""Checkpoint selector: allowlist, safe switch order, identity, fail-closed cleanup.

The controller and the API are driven with fake session/policy-server objects, so
no GPU, no policy server and no real run directory are involved; the switch
semantics (stop before restart, verify before ready, rollback on failure) are
what these tests pin down.
"""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from humanoid_lab.psi0_bridge.checkpoints import (
    CheckpointController,
    CheckpointEntry,
    CheckpointError,
    CheckpointUnavailable,
    SwitchInProgress,
    UnknownCheckpoint,
)
from humanoid_lab.psi0_bridge.contracts import validate_info
from humanoid_lab.psi0_bridge.session import ERROR, IDLE, RUNNING

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - host python without the serve group
    TestClient = None

FINE_DIR = Path("/outputs/psi0-unitree-dex3-sonic-v1/finetune/run.2609171455")
BASE_DIR = Path("/outputs/psi0-unitree-dex3-sonic-v1/base/postpre.sonic1.0.unifolm.2609092156.40k")


def make_info(run_dir: Path, step: int):
    return validate_info({
        "run_dir": str(run_dir),
        "ckpt_step": step,
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
    })


def entry(entry_id: str, label: str, run_dir, step: int, *, available: bool = True,
          reason: str | None = None) -> CheckpointEntry:
    return CheckpointEntry(id=entry_id, label=label, run_dir=run_dir, step=step,
                           available=available, reason=reason)


class FakeMonitor:
    def start(self) -> None: pass
    def stop(self, timeout: float = 3.0) -> None: pass
    def set_active(self, active: bool) -> None: pass
    def status(self):
        return {"state": {"endpoint": "tcp://localhost:5557", "alive": True, "age_s": 0.1},
                "camera": {"endpoint": "tcp://localhost:5558", "alive": True, "shape": [480, 640, 3]}}


class FakeSession:
    """The session surface the controller and app use, with a call log."""

    def __init__(self, state: str = IDLE, info=None) -> None:
        self.monitor = FakeMonitor()
        self.state = state
        self.error = None
        self.info = info
        self.calls: list[str] = []
        self.closes = 0

    def stop(self):
        self.calls.append("stop")
        if self.state != ERROR:
            self.state = "STOPPED"
        return self.status()

    def mark_error(self, message: str):
        self.calls.append(f"mark_error:{message}")
        self.state = ERROR
        self.error = message
        return self.status()

    def adopt_policy(self, info):
        self.calls.append("adopt_policy")
        self.info = info
        self.error = None
        if self.state != RUNNING:
            self.state = IDLE

    def close(self):
        self.closes += 1
        self.calls.append("close")

    def status(self):
        return {
            "state": self.state,
            "error": self.error,
            "starting": False,
            "psi0": {"url": "ws://localhost:8014/ws", "connected": self.state == RUNNING,
                     "info": None if self.info is None else {"action_dim": self.info.action_dim}},
            "sonic_state": self.monitor.status()["state"],
            "camera": self.monitor.status()["camera"],
            "action": {"endpoint": "tcp://*:5556", "last_time": None, "last_age_s": None,
                       "last_index": None, "sent": 0},
        }


class FakeServer:
    def __init__(self) -> None:
        self.calls: list = []
        self.pid = 4242
        self.alive = False
        self.run_dir = None
        self.step = None
        self.start_error: Exception | None = None

    def start(self, run_dir, step):
        self.calls.append(("start", str(run_dir), step))
        if self.start_error is not None:
            raise self.start_error
        self.alive = True
        self.run_dir = Path(run_dir)
        self.step = step

    def stop(self):
        self.calls.append(("stop",))
        self.alive = False
        self.run_dir = None
        self.step = None


def build_controller(*, session=None, server=None, verify=None, base_available=True,
                     base_reason=None):
    session = session or FakeSession()
    server = server or FakeServer()
    entries = [
        entry("fine-tuned", "Fine-tuned (40k)", FINE_DIR, 40000),
        entry("base", "Base", BASE_DIR if base_available else None, 0,
              available=base_available, reason=base_reason),
    ]
    infos = {"fine-tuned": make_info(FINE_DIR, 40000), "base": make_info(BASE_DIR, 0)}

    def default_verify(candidate):
        return infos[candidate.id]

    controller = CheckpointController(
        session, server, entries,
        verify=verify or default_verify,
        initial_id="fine-tuned",
        log=lambda message: None,
    )
    session.info = infos["fine-tuned"]
    server.alive = True
    server.run_dir = FINE_DIR
    server.step = 40000
    return controller, session, server


class AllowlistTest(unittest.TestCase):
    def test_status_lists_exactly_the_two_allowlisted_options(self) -> None:
        controller, _, _ = build_controller()
        status = controller.status()
        self.assertEqual([opt["id"] for opt in status["options"]], ["fine-tuned", "base"])
        self.assertEqual([opt["label"] for opt in status["options"]],
                         ["Fine-tuned (40k)", "Base"])
        self.assertEqual(status["selected"]["id"], "fine-tuned")
        self.assertTrue(status["serving_selected"])
        self.assertEqual(status["active"]["step"], 40000)

    def test_an_unavailable_base_is_reported_never_substituted(self) -> None:
        controller, _, _ = build_controller(base_available=False, base_reason="no warm start")
        status = controller.status()
        base = status["options"][1]
        self.assertFalse(base["available"])
        self.assertEqual(base["reason"], "no warm start")
        self.assertIsNone(base["run_dir"])
        with self.assertRaises(CheckpointUnavailable):
            controller.switch("base")

    def test_unknown_ids_are_refused(self) -> None:
        controller, _, _ = build_controller()
        with self.assertRaises(UnknownCheckpoint):
            controller.switch("../../etc/passwd")
        with self.assertRaises(UnknownCheckpoint):
            controller.switch("base-2")

    def test_a_switch_in_progress_is_refused(self) -> None:
        release = threading.Event()

        def slow_verify(candidate):
            release.wait(timeout=5)
            return make_info(candidate.run_dir, candidate.step)

        controller, _, _ = build_controller(verify=slow_verify)
        errors: list = []
        thread = threading.Thread(target=lambda: self._switch(controller, "base", errors))
        thread.start()
        time.sleep(0.2)
        try:
            with self.assertRaises(SwitchInProgress):
                controller.switch("fine-tuned")
            self.assertTrue(controller.status()["switching"])
        finally:
            release.set()
            thread.join(timeout=5)
        self.assertEqual(errors, [])

    @staticmethod
    def _switch(controller, entry_id, sink):
        try:
            controller.switch(entry_id)
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            sink.append(exc)


class SwitchOrderTest(unittest.TestCase):
    def test_stop_then_restart_then_verify_then_ready(self) -> None:
        order: list[str] = []
        session = FakeSession(state=RUNNING, info=make_info(FINE_DIR, 40000))
        session.stop = lambda: (order.append("session.stop"), FakeSession.stop(session))[1]
        server = FakeServer()
        server.alive = True
        server.run_dir = FINE_DIR
        server.step = 40000
        server.stop = lambda: (order.append("server.stop"), FakeServer.stop(server))[1]
        server.start = lambda run_dir, step: (
            order.append(f"server.start:{Path(run_dir).name}"), FakeServer.start(server, run_dir, step))[1]
        infos = {"fine-tuned": make_info(FINE_DIR, 40000), "base": make_info(BASE_DIR, 0)}
        controller, _, _ = build_controller(
            session=session, server=server,
            verify=lambda candidate: (order.append(f"verify:{candidate.id}") or infos[candidate.id]),
        )
        session.state = RUNNING
        controller.switch("base")
        self.assertEqual(order, [
            "session.stop",
            "server.stop",
            f"server.start:{BASE_DIR.name}",
            "verify:base",
        ])
        self.assertEqual(session.state, IDLE)  # ready again, not RUNNING
        self.assertEqual(controller.selected_id, "base")
        self.assertEqual(controller.status()["active"]["step"], 0)

    def test_switch_to_the_serving_entry_is_a_noop(self) -> None:
        controller, session, server = build_controller()
        controller.switch("fine-tuned")
        self.assertEqual(session.calls, [])
        self.assertEqual(server.calls, [])


class FailClosedTest(unittest.TestCase):
    def test_identity_failure_rolls_back_and_marks_error(self) -> None:
        calls: list = []

        def verify(candidate):
            calls.append(candidate.id)
            if candidate.id == "base":
                raise RuntimeError("identity mismatch: served run_dir is another run")
            return make_info(candidate.run_dir, candidate.step)

        controller, session, server = build_controller(verify=verify)
        with self.assertRaises(CheckpointError) as ctx:
            controller.switch("base")
        self.assertIn("rolled back to Fine-tuned (40k)", str(ctx.exception))
        self.assertEqual(calls, ["base", "fine-tuned"])
        self.assertEqual([call[0] for call in server.calls],
                         ["stop", "start", "stop", "start"])
        self.assertEqual(session.state, ERROR)
        self.assertIn("failed", session.error)
        self.assertIn("rolled back", session.error)
        self.assertEqual(controller.selected_id, "fine-tuned")
        self.assertTrue(server.alive)  # the previous checkpoint is serving again

    def test_a_failed_rollback_says_so(self) -> None:
        def verify(candidate):
            raise RuntimeError(f"cannot serve {candidate.id}")

        controller, session, server = build_controller(verify=verify)
        with self.assertRaises(CheckpointError) as ctx:
            controller.switch("base")
        self.assertIn("rollback to Fine-tuned (40k) failed", str(ctx.exception))
        self.assertIn("no policy server is running", session.error)
        self.assertFalse(server.alive)

    def test_a_failed_server_start_never_marks_ready(self) -> None:
        controller, session, server = build_controller()
        server.start_error = RuntimeError("cannot spawn serve_psi0_sonic")
        with self.assertRaises(CheckpointError):
            controller.switch("base")
        self.assertEqual(session.state, ERROR)
        self.assertFalse(server.alive)
        self.assertEqual(controller.selected_id, "fine-tuned")

    def test_close_stops_the_owned_server(self) -> None:
        controller, _, server = build_controller()
        controller.close()
        self.assertEqual(server.calls, [("stop",)])
        self.assertFalse(server.alive)


@unittest.skipIf(TestClient is None, "fastapi is not installed in this environment")
class CheckpointApiTest(unittest.TestCase):
    def setUp(self) -> None:
        from humanoid_lab.psi0_bridge.app import create_app

        self.controller, self.session, self.server = build_controller()
        app = create_app(self.session, checkpoints=self.controller)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_the_page_carries_the_selector(self) -> None:
        body = self.client.get("/").text
        self.assertIn('id="cp-options"', body)
        self.assertIn('id="cp-detail"', body)

    def test_options_and_status(self) -> None:
        payload = self.client.get("/api/checkpoints").json()
        self.assertEqual([opt["label"] for opt in payload["options"]],
                         ["Fine-tuned (40k)", "Base"])
        self.assertEqual(payload["selected"]["run_dir"], str(FINE_DIR))
        self.assertEqual(payload["selected"]["step"], 40000)
        self.assertEqual(payload["active"]["step"], 40000)

    def test_selection_takes_an_allowlisted_id_only(self) -> None:
        response = self.client.post("/api/checkpoint", json={"id": "base"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.controller.selected_id, "base")
        self.assertEqual(response.json()["selected"]["label"], "Base")
        self.assertEqual(response.json()["active"]["step"], 0)

    def test_paths_and_unknown_ids_are_refused(self) -> None:
        cases = [
            ({"id": "/outputs/other-run"}, 400),
            ({"id": "base", "run_dir": "/outputs/other-run"}, 400),
            ({"id": "base", "step": 40000}, 400),
            ({"run_dir": "/outputs/other-run"}, 400),
            ({"id": 7}, 400),
            ([], 422),  # not an object at all: FastAPI refuses it before the handler
        ]
        for body, expected in cases:
            with self.subTest(body=body):
                response = self.client.post("/api/checkpoint", json=body)
                self.assertEqual(response.status_code, expected)
        self.assertEqual(self.controller.selected_id, "fine-tuned")

    def test_a_failed_switch_is_502_with_the_session_in_error(self) -> None:
        def verify(candidate):
            raise RuntimeError("identity mismatch")

        controller, session, _ = build_controller(verify=verify)
        from humanoid_lab.psi0_bridge.app import create_app

        client = TestClient(create_app(session, checkpoints=controller))
        with client:
            response = client.post("/api/checkpoint", json={"id": "base"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("failed", response.json()["detail"])
        self.assertEqual(session.state, ERROR)


if __name__ == "__main__":
    unittest.main()
