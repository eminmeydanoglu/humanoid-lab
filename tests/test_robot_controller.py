"""A motor-free dry run of operator transitions and fail-closed health gates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from humanoid_lab.robot_runtime.controller import Model, RobotController, RobotSessionError
from humanoid_lab.robot_runtime.sonic_status import SonicStatus


class FakeMonitor:
    def __init__(self):
        self.ready = True

    def start(self): pass
    def stop(self): pass
    def invalidate(self): self.ready = False
    def frame(self, **_kwargs): return object() if self.ready else None
    def state(self, **_kwargs): return object() if self.ready else None
    def status(self):
        return {name: {"alive": self.ready, "age_s": 0 if self.ready else None}
                for name in ("camera", "state")}


class FakeSonic:
    mode = "PLANNER"
    connected = True

    def start(self): pass
    def close(self): pass
    def latest(self):
        if not self.connected:
            return None
        return SonicStatus(self.mode, 10 if self.mode == "STREAMED_MOTION" else None,
                           1, time.monotonic())


class FakeRouter:
    def __init__(self, sonic):
        self.sonic = sonic
        self.sent = 0
        self.last_time = None
        self.error = None
        self.gate = False
        self.commands = []

    def select(self, _source): pass
    def resume(self): self.gate = True
    def halt(self): self.gate = False
    def reset_fault(self): self.error = None
    def close(self): pass
    def send_control(self, planner):
        self.commands.append(planner)
        self.sonic.mode = "PLANNER" if planner else "STREAMED_MOTION"
    def status(self):
        return {"sent": self.sent, "last_time": self.last_time, "error": self.error}


@dataclass(frozen=True)
class FakeConfig:
    instruction: str = ""


class FakePsiSession:
    config = FakeConfig()

    def __init__(self, router): self.router = router
    def start(self):
        self.router.sent += 2
        self.router.last_time = time.time()
        return {"state": "RUNNING"}
    def stop(self): pass
    def close(self): pass


class FakeServer:
    def start(self, *_args): pass
    def stop(self): pass


class FakeBridge:
    def start(self): pass
    def close(self): pass


@pytest.fixture
def robot():
    sonic = FakeSonic()
    router = FakeRouter(sonic)
    monitor = FakeMonitor()
    controller = RobotController(
        monitor=monitor, sonic=sonic, router=router, camera_bridge=FakeBridge(),
        psi_session=FakePsiSession(router), psi_server=FakeServer(), groot=FakeServer(),
        models=[Model("psi", "psi", Path("/unused"), 100)],
        tasks={f"task{i}": f"Prompt {i}" for i in range(13)},
        command_builder=lambda *, start, stop, planner: planner,
    )
    controller.open()
    try:
        yield controller, monitor, sonic, router
    finally:
        controller.close()


def test_start_stop_reset_and_no_observation(robot):
    controller, monitor, sonic, router = robot
    controller.stop()
    assert router.commands == []  # no robot start command on idle shutdown
    assert controller.start()["sonic"]["mode"] == "STREAMED_MOTION"
    assert controller.status()["session"] == "RUNNING"
    assert controller.stop()["sonic"]["mode"] == "PLANNER"
    assert not router.gate
    controller.reset()
    assert controller.status()["session"] == "IDLE"
    assert controller.status()["action"]["sent_this_session"] == 0
    with pytest.raises(RobotSessionError, match="RGB"):
        controller.start()


def test_model_silence_and_status_loss_return_to_planner(robot):
    controller, monitor, sonic, router = robot
    controller.start()
    router.last_time = time.time() - 1
    time.sleep(0.12)
    assert controller.status()["session"] == "ERROR"
    assert sonic.mode == "PLANNER"
    controller.reset()
    monitor.ready = True
    controller.start()
    sonic.connected = False
    time.sleep(0.12)
    assert controller.status()["session"] == "ERROR"
    assert not router.gate


def test_bad_router_packet_and_stale_camera_fail_closed(robot):
    controller, monitor, sonic, router = robot
    controller.start()
    router.error = "invalid groot pose packet"
    time.sleep(0.12)
    assert controller.status()["session"] == "ERROR"
    assert not router.gate
    controller.reset()
    monitor.ready = True
    controller.start()
    monitor.ready = False
    time.sleep(0.12)
    assert controller.status()["session"] == "ERROR"
    assert not router.gate
