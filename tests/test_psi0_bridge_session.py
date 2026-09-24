"""Session lifecycle: IDLE/RUNNING/STOPPED/ERROR, Start gating, Stop, Reset.

These drive the real :class:`Session` state machine but replace only its two
transport seams -- the ``/info`` fetch, the WebSocket connection, the action
publisher, and the Isaac reset client -- plus its ZMQ poll thread (the
:class:`Monitor`, injected as a cached-snapshot stand-in).  No sockets, GPU or
Isaac are involved.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import unittest
from unittest import mock

import numpy as np

from humanoid_lab.psi0_bridge import session as session_mod
from humanoid_lab.psi0_bridge.contracts import validate_info
from humanoid_lab.psi0_bridge.monitor import FrameSnapshot, StateSnapshot
from humanoid_lab.psi0_bridge.policy_clock import PolicyTime
from humanoid_lab.psi0_bridge.prompt import CANONICAL_PROMPT
from humanoid_lab.psi0_bridge.reset_client import ResetError
from humanoid_lab.psi0_bridge.session import (
    ERROR,
    IDLE,
    RUNNING,
    STOPPED,
    Session,
    SessionConfig,
    SessionError,
)

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
RAW_STATE = np.zeros(43, dtype=np.float32)

#: Ordered events from the fakes, to pin the Reset sequence.
events: list[str] = []


def server_info(state_dim: int = 45, action_dim: int = 80):
    return validate_info(
        {
            "policy": "psi0",
            "run_dir": "/outputs/psi0-unitree-dex3-sonic-v1/finetune/run.2609171455",
            "ckpt_step": 40000,
            "dataset_name": "psi0-unitree-dex3-sonic-v1",
            "transforms": [
                {"name": "resize", "size": [240, 320]},
                {"name": "center_crop", "size": [240, 320]},
            ],
            "expected_keys": {
                "image": {"observation.images.egocentric": "HxWx3 uint8 image array"},
                "state": {"states": f"1x{state_dim} unnormalized state vector"},
            },
            "observation": {"state_dim": state_dim, "normalize_state": True},
            "action": {"action_dim": action_dim, "action_chunk_size": 30, "action_exec_horizon": 30},
            "rtc_enabled": False,
        }
    )


class FakeMonitor:
    """Cached-snapshot stand-in for the bridge's single ZMQ poll thread.

    Like the real :class:`Monitor`, it honours ``max_age_s`` and lets a Reset
    drop everything cached; ``refresh`` models the poll thread delivering the
    next observation.
    """

    def __init__(self, *, state: bool = True, frame: bool = True, age_s: float = 0.0) -> None:
        self.active: bool | None = None
        self.invalidations = 0
        self._state = None
        self._frame = None
        self._state_enabled = state
        self._frame_enabled = frame
        self._age_s = float(age_s)
        self.refresh()

    def refresh(self, *, age_s: float | None = None) -> None:
        age = self._age_s if age_s is None else float(age_s)
        now = time.monotonic()
        self._state = (
            StateSnapshot(payload={}, raw_state=RAW_STATE.copy(), timestamp_s=now - age)
            if self._state_enabled
            else None
        )
        self._frame = (
            FrameSnapshot(frame=FRAME.copy(), timestamp_s=now - age)
            if self._frame_enabled
            else None
        )

    def set_active(self, active: bool) -> None:
        self.active = bool(active)

    def invalidate(self) -> None:
        self._state = None
        self._frame = None
        self.invalidations += 1
        events.append("invalidate")

    @staticmethod
    def _fresh(snapshot, max_age_s):
        if snapshot is None:
            return None
        if max_age_s is not None and time.monotonic() - snapshot.timestamp_s > max_age_s:
            return None
        return snapshot

    def state(self, max_age_s=None):
        return self._fresh(self._state, max_age_s)

    def frame(self, max_age_s=None):
        return self._fresh(self._frame, max_age_s)

    def status(self) -> dict:
        return {
            "state": {"alive": self._state is not None, "endpoint": "tcp://localhost:5557"},
            "camera": {"alive": self._frame is not None, "endpoint": "tcp://localhost:5558"},
        }


def _action_message(action: np.ndarray, version: int) -> str:
    flat = np.ascontiguousarray(np.asarray(action, dtype=np.float32).reshape(1, -1))
    return json.dumps(
        {
            "action": {
                "__numpy__": base64.b64encode(flat.tobytes()).decode("ascii"),
                "dtype": "<f4",
                "shape": list(flat.shape),
            },
            "err": 0.0,
            "version": version,
        }
    )


class FakeConnection:
    """In-process WebSocket stand-in: answers each observation with one action."""

    instances: list["FakeConnection"] = []
    action_width = 80
    neck_values = (0.0, 0.0)
    messages_sent = 0

    def __init__(self, url: str, **_kwargs) -> None:
        self.url = url
        self.sent: list[str] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._version = 0
        FakeConnection.instances.append(self)

    async def __aenter__(self) -> "FakeConnection":
        return self

    async def __aexit__(self, *_exc) -> bool:
        await self.close()
        return False

    async def send(self, payload: str) -> None:
        self.sent.append(payload)
        FakeConnection.messages_sent += 1
        self._version += 1
        # A valid action: token on the FSQ grid, hands zero, neck a no-op zero.
        action = np.zeros((1, FakeConnection.action_width), dtype=np.float32)
        action[0, : min(64, FakeConnection.action_width)] = 0.1
        if FakeConnection.action_width == 80:
            action[0, 78:80] = FakeConnection.neck_values
        self._queue.put_nowait(_action_message(action, self._version))

    async def recv(self) -> str:
        return await self._queue.get()

    async def close(self) -> None:
        pass


class LiveFakeMonitor(FakeMonitor):
    def state(self, max_age_s=None):
        self.refresh()
        return super().state(max_age_s)

    def frame(self, max_age_s=None):
        self.refresh()
        return super().frame(max_age_s)


class ScaledSimulationClock:
    mode = "simulation"

    def __init__(self, speed: float = 0.67) -> None:
        self.speed = float(speed)
        self.started = time.monotonic()

    def now(self) -> PolicyTime:
        return PolicyTime((time.monotonic() - self.started) * self.speed, 0)


class SimulationPushConnection(FakeConnection):
    """Server pushes at 30 simulation Hz after a synchronous first-chunk delay."""

    async def send(self, payload: str) -> None:
        self.sent.append(payload)
        FakeConnection.messages_sent += 1

    async def recv(self) -> str:
        await asyncio.sleep(0.3 if self._version == 0 else 1.0 / (30.0 * 0.67))
        self._version += 1
        action = np.zeros((1, self.action_width), dtype=np.float32)
        action[0, : min(64, self.action_width)] = 0.1
        return _action_message(action, self._version)


class FakePublisher:
    """Mirrors PosePublisher: one socket per service, Stop only closes the gate."""

    instances: list["FakePublisher"] = []

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.sent: list[bytes] = []
        self.halted = True  # owned at construction; Start opens the send gate
        self.closed = False
        FakePublisher.instances.append(self)

    def publish(self, payload: bytes) -> bool:
        if self.halted or self.closed:
            return False
        self.sent.append(payload)
        return True

    def halt(self) -> None:
        self.halted = True

    def resume(self) -> None:
        self.halted = False

    def close(self) -> None:
        self.closed = True
        self.halted = True


class FakeResetClient:
    instances: list["FakeResetClient"] = []

    def __init__(self, endpoint: str, *, timeout_ms: int = 2000) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.calls = 0
        FakeResetClient.instances.append(self)

    def request_reset(self) -> None:
        self.calls += 1
        events.append("reset")
        if fail_flag[0]:
            raise ResetError("fake reset refused")


# Test-local switch so a single fake class can model success and failure.
fail_flag = [False]


class SessionLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeConnection.instances = []
        FakePublisher.instances = []
        FakeResetClient.instances = []
        FakeConnection.action_width = 80
        FakeConnection.neck_values = (0.0, 0.0)
        fail_flag[0] = False
        events.clear()
        self.monitor = FakeMonitor()
        self.info_ok = [True]
        self.info_action_dim = [80]

        def fake_fetch_info(ws_url, **_kwargs):
            if not self.info_ok[0]:
                raise session_mod.Psi0ClientError("server down")
            return server_info(action_dim=self.info_action_dim[0])

        self._patchers = [
            mock.patch.object(session_mod, "Psi0Connection", FakeConnection),
            mock.patch.object(session_mod, "PosePublisher", FakePublisher),
            mock.patch.object(session_mod, "IsaacResetClient", FakeResetClient),
            mock.patch.object(session_mod, "fetch_info", fake_fetch_info),
        ]
        for patcher in self._patchers:
            patcher.start()
        self.session = Session(
            SessionConfig(
                ws_url="ws://localhost:8014/ws",
                control_hz=50.0,
                recv_timeout_s=2.0,
                ready_timeout_s=0.3,
                start_timeout_s=2.0,
                stop_timeout_s=1.0,
            ),
            monitor=self.monitor,
        )

    def tearDown(self) -> None:
        try:
            self.session.stop()
        finally:
            for patcher in self._patchers:
                patcher.stop()

    def _wait_for(self, predicate, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def test_start_is_refused_until_state_and_camera_are_fresh(self) -> None:
        self.session.monitor = FakeMonitor(state=False, frame=False)
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == ERROR))
        status = self.session.status()
        self.assertIn("not ready", status["error"])
        # The socket exists (owned since construction) but nothing was published.
        self.assertEqual(FakePublisher.instances[0].sent, [])
        self.assertTrue(FakePublisher.instances[0].halted)

    def test_start_is_refused_when_the_server_info_is_invalid(self) -> None:
        self.info_ok[0] = False
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == ERROR))
        self.assertIn("server down", self.session.status()["error"])
        self.assertEqual(FakePublisher.instances[0].sent, [])
        self.assertTrue(FakePublisher.instances[0].halted)

    def test_running_publishes_and_stop_halts_immediately(self) -> None:
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == RUNNING))
        self.assertTrue(self._wait_for(lambda: FakePublisher.instances and FakePublisher.instances[0].sent))
        publisher = FakePublisher.instances[0]
        status = self.session.status()
        self.assertEqual(status["state"], RUNNING)
        self.assertEqual(status["action"]["last_index"], 0)
        self.assertGreaterEqual(status["action"]["sent"], 1)

        self.session.stop()
        self.assertEqual(self.session.state, STOPPED)
        # Stop closes the send gate but keeps owning the socket: a later Start
        # (or another publisher) must not be able to race the port.
        self.assertTrue(publisher.halted)
        self.assertFalse(publisher.closed)
        published = len(publisher.sent)
        time.sleep(0.1)
        self.assertEqual(len(publisher.sent), published)

        # Only the service shutdown releases the socket.
        self.session.close()
        self.assertTrue(publisher.closed)

    def test_masked_neck_values_do_not_stop_sonic_session(self) -> None:
        from dataclasses import replace

        FakeConnection.neck_values = (-0.03, -0.34)
        self.session.config = replace(self.session.config, neck_policy="discard")
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.status()["action"]["sent"] >= 1))
        status = self.session.status()
        self.assertEqual(status["state"], RUNNING)
        self.assertGreaterEqual(status["action"]["neck_padding_discards"], 1)

    def test_a_bad_action_width_moves_to_error_without_a_stale_repeat(self) -> None:
        FakeConnection.action_width = 77  # neither 78 nor 80
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == ERROR))
        self.assertTrue(FakePublisher.instances)
        publisher = FakePublisher.instances[0]
        self.assertEqual(publisher.sent, [])
        sent = len(publisher.sent)
        time.sleep(0.1)
        self.assertEqual(len(publisher.sent), sent)

    def test_a_server_declaring_80_but_sending_78_fails_closed(self) -> None:
        FakeConnection.action_width = 78  # /info promised 80
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == ERROR))
        self.assertIn("action_dim 80", self.session.status()["error"])
        self.assertEqual(FakePublisher.instances[0].sent, [])

    def test_a_server_declaring_78_but_sending_80_fails_closed(self) -> None:
        self.info_action_dim[0] = 78
        FakeConnection.action_width = 80  # /info promised 78
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == ERROR))
        self.assertIn("action_dim 78", self.session.status()["error"])
        self.assertEqual(FakePublisher.instances[0].sent, [])

    def test_a_78d_action_streams_when_the_server_declares_78(self) -> None:
        self.info_action_dim[0] = 78
        FakeConnection.action_width = 78
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == RUNNING))
        self.assertTrue(self._wait_for(lambda: FakePublisher.instances and FakePublisher.instances[0].sent))

    def test_error_requires_a_reset_before_start(self) -> None:
        FakeConnection.action_width = 77
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == ERROR))

        with self.assertRaises(SessionError):
            self.session.start()
        self.assertEqual(self.session.state, ERROR)

        self.session.reset()
        self.assertEqual(self.session.state, IDLE)
        FakeConnection.action_width = 80
        self.monitor.refresh()  # the real monitor keeps polling after a reset
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == RUNNING))

    def test_reset_stops_first_requests_reset_and_clears_the_sequence(self) -> None:
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == RUNNING))
        self.assertTrue(self._wait_for(lambda: FakePublisher.instances and FakePublisher.instances[0].sent))

        self.session.reset()
        self.assertEqual(self.session.state, IDLE)
        self.assertEqual(FakeResetClient.instances[-1].calls, 1)
        # Stop happened before the request: the gate is closed, the socket stays owned.
        self.assertTrue(FakePublisher.instances[0].halted)
        self.assertFalse(FakePublisher.instances[0].closed)
        # The scene reset is requested first; only then are the caches dropped.
        self.assertEqual(events, ["reset", "invalidate"])
        status = self.session.status()
        self.assertIsNone(status["action"]["last_index"])
        self.assertEqual(status["action"]["sent"], 0)

    def test_reset_requires_a_fresh_observation_before_the_next_start(self) -> None:
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == RUNNING))
        self.session.reset()
        self.assertEqual(self.session.state, IDLE)

        # Nothing has been observed since the reset: the pre-reset frame/state
        # must not be reused, so Start fails closed as "not ready".
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == ERROR))
        self.assertIn("not ready", self.session.status()["error"])

        # Once the poll thread delivers a new observation, Start works again.
        self.monitor.refresh()
        self.session.reset()
        self.monitor.refresh()
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == RUNNING))

    def test_stale_observations_are_not_ready(self) -> None:
        stale = FakeMonitor(age_s=1.0)  # older than the 0.5 s freshness contract
        self.session.monitor = stale
        self.session.start()
        self.assertTrue(self._wait_for(lambda: self.session.state == ERROR))
        self.assertIn("not ready", self.session.status()["error"])
        self.assertEqual(FakePublisher.instances[0].sent, [])

    def test_reset_waits_out_the_command_ttl_before_requesting_the_scene_reset(self) -> None:
        session = Session(
            SessionConfig(
                ws_url="ws://localhost:8014/ws",
                control_hz=50.0,
                recv_timeout_s=2.0,
                ready_timeout_s=0.3,
                start_timeout_s=2.0,
                stop_timeout_s=1.0,
                command_ttl_s=0.25,
            ),
            monitor=self.monitor,
        )
        session.start()
        self.assertTrue(self._wait_for(lambda: session.state == RUNNING))

        started = time.monotonic()
        session.reset()
        self.assertGreaterEqual(time.monotonic() - started, 0.25)
        self.assertEqual(session.state, IDLE)
        self.assertTrue(FakePublisher.instances[0].halted)  # stopped before the wait
        self.assertFalse(FakePublisher.instances[0].closed)
        self.assertEqual(FakeResetClient.instances[-1].calls, 1)  # requested after it

    def test_a_refused_reset_reports_error(self) -> None:
        fail_flag[0] = True
        self.session.reset()
        self.assertEqual(self.session.state, ERROR)
        self.assertIn("fake reset refused", self.session.status()["error"])
        self.assertEqual(FakeResetClient.instances[-1].calls, 1)

    def test_simulation_clock_skips_obsolete_request_opportunities_without_debt(self) -> None:
        session = Session(
            SessionConfig(
                ws_url="ws://localhost:8014/ws",
                control_hz=30.0,
                recv_timeout_s=2.0,
                ready_timeout_s=0.3,
                start_timeout_s=2.0,
                stop_timeout_s=1.0,
            ),
            monitor=LiveFakeMonitor(),
            policy_clock=ScaledSimulationClock(0.67),
        )
        with mock.patch.object(session_mod, "Psi0Connection", SimulationPushConnection):
            session.start()
            self.assertTrue(self._wait_for(lambda: session.state == RUNNING))
            publisher = FakePublisher.instances[-1]
            self.assertTrue(self._wait_for(lambda: len(publisher.sent) >= 20, timeout=2.0))
            session.stop()

        connection = SimulationPushConnection.instances[-1]
        self.assertGreaterEqual(len(connection.sent), len(publisher.sent))
        self.assertLessEqual(len(connection.sent) - len(publisher.sent), 8)
        self.assertGreaterEqual(len(publisher.sent), 20)
        self.assertEqual(session.status()["action"]["sent"], len(publisher.sent))

    def test_instruction_is_the_canonical_prompt_not_a_caller_string(self) -> None:
        self.assertEqual(self.session.config.instruction, CANONICAL_PROMPT)
        self.session.start()
        self.assertTrue(self._wait_for(lambda: FakeConnection.instances))
        connection = FakeConnection.instances[0]
        self.assertTrue(self._wait_for(lambda: connection.sent))
        payload = json.loads(connection.sent[0])
        self.assertEqual(payload["instruction"], CANONICAL_PROMPT)
        self.assertEqual(payload["history"], {})
        self.assertEqual(payload["state"]["states"]["shape"], [43])


if __name__ == "__main__":
    unittest.main()
