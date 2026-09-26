"""Exercise Foxy service behavior with stand-in ROS messages and no robot publishers."""

import importlib
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, text):
        self.lines.append(("info", text))

    def warn(self, text):
        self.lines.append(("warn", text))

    def error(self, text):
        self.lines.append(("error", text))


class FakeNode:
    def __init__(self, name):
        self.name = name
        self.params = {}
        self.services = {}
        self.publishers = []
        self.logger = FakeLogger()

    def declare_parameter(self, name, default):
        self.params[name] = default

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=time.time_ns()))

    def get_logger(self):
        return self.logger

    def create_subscription(self, *args):
        return args

    def create_service(self, kind, name, callback):
        self.services[name] = callback

    def create_timer(self, period, callback):
        self.timer = callback

    def create_publisher(self, *args):
        self.publishers.append(args)
        raise AssertionError("dry-run must not create publishers")

    def destroy_node(self):
        pass


class FakeNetwork:
    def __init__(self, endpoint, events, timeout, public, secret, server):
        self.commands = []

    def start(self):
        pass

    def submit(self, *args):
        self.commands.append(args)

    def close(self):
        pass

    def join(self, timeout):
        pass


@pytest.fixture
def robot_node(monkeypatch):
    stubs = {
        "rclpy": {}, "rclpy.node": {"Node": FakeNode},
        "sensor_msgs": {}, "sensor_msgs.msg": {"Image": type("Image", (), {})},
        "std_srvs": {}, "std_srvs.srv": {"Trigger": type("Trigger", (), {})},
        "unitree_hg": {}, "unitree_hg.msg": {"LowState": type("LowState", (), {}),
                                         "HandState": type("HandState", (), {})},
        "flux_dex3_interfaces": {}, "flux_dex3_interfaces.srv": {
            "GetStatus": type("GetStatus", (), {}), "StartTask": type("StartTask", (), {})},
    }
    for name, attrs in stubs.items():
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("flux_dex3.node", None)
    module = importlib.import_module("flux_dex3.node")
    monkeypatch.setattr(module, "NetworkWorker", FakeNetwork)
    node = module.Dex3Node()
    yield node
    node.destroy_node()
    sys.modules.pop("flux_dex3.node", None)


def _image_message(stamp):
    return SimpleNamespace(encoding="rgb8", height=192, width=256, step=768,
                           data=bytes(192 * 256 * 3),
                           header=SimpleNamespace(stamp=SimpleNamespace(sec=int(stamp),
                                                                        nanosec=int((stamp % 1) * 1e9))))


def _motors(size, q):
    return SimpleNamespace(motor_state=[SimpleNamespace(q=float(q)) for _ in range(size)])


def _fresh_observation(node):
    stamp = time.time()
    node._image(_image_message(stamp))
    for key, size in (("low", 35), ("left", 7), ("right", 7)):
        node._store(key, _motors(size, 0.0))
    return stamp


def test_pause_repeats_last_target_until_stop_and_logs_transitions(robot_node):
    node = robot_node
    assert not node.publishers
    node.events.put(("status", {"status": "LOADING", "checkpoint": "", "error": ""}))
    node._tick()
    node.events.put(("status", {"status": "READY", "checkpoint": "sha256:" + "0" * 64, "error": ""}))
    node._tick()
    stamp = _fresh_observation(node)
    start = node._start(SimpleNamespace(prompt="stack three block"), SimpleNamespace())
    assert start.accepted and len(node.network.commands) == 1
    session = node.chunk_executor.session
    output = SimpleNamespace(calls=[], publish=lambda target, measured: output.calls.append(np.array(target, copy=True)))
    node.command_output = output
    node.events.put(("actions", session, 0, stamp, np.full((32, 28), 0.1, np.float32), time.monotonic()))
    node._tick()
    assert len(output.calls) == 1
    pause = node._pause(SimpleNamespace(), SimpleNamespace())
    assert pause.success and node.paused and node.chunk_executor.session is None
    latched = node.pause_target.copy()
    for _ in range(3):
        node._tick()
    assert len(output.calls) == 4
    for action in output.calls:
        np.testing.assert_array_equal(action, latched)
    assert node._status(SimpleNamespace(), SimpleNamespace()).state == "PAUSED"
    node.events.put(("status", {"status": "READY", "checkpoint": "sha256:" + "0" * 64, "error": ""}))
    node._tick()
    assert "paused; repeating" in node._status(SimpleNamespace(), SimpleNamespace()).reason
    node.events.put(("actions", session, 0, stamp, np.zeros((32, 28), np.float32), time.monotonic()))
    node._tick()
    np.testing.assert_array_equal(node.pause_target, latched)
    stopped = node._stop(SimpleNamespace(), SimpleNamespace())
    assert stopped.success and not node.paused and node.pause_target is None and node.last_target is None
    after_stop = len(output.calls)
    node._tick()
    assert len(output.calls) == after_stop
    text = "\n".join(line for _, line in node.logger.lines)
    for expected in ("model=LOADING", "GPU model status: READY", "StartTask accepted",
                     "Prediction accepted", "PauseTask", "Discarded stale prediction", "StopTask"):
        assert expected in text


def test_pause_blocks_resume_until_old_prediction_completes(robot_node):
    node = robot_node
    node.server_status = "READY"
    stamp = _fresh_observation(node)
    node._start(SimpleNamespace(prompt="stack three block"), SimpleNamespace())
    session = node.chunk_executor.session
    node.events.put(("actions", session, 0, stamp, np.full((32, 28), 0.05, np.float32), time.monotonic()))
    node._tick()
    node._request()
    assert node.inflight
    assert node._pause(SimpleNamespace(), SimpleNamespace()).success
    assert node.paused_request_pending
    blocked = node._start(SimpleNamespace(prompt="stack three block"), SimpleNamespace())
    assert not blocked.accepted and node.paused
    node.events.put(("error", session, 1, "prediction timed out"))
    node._tick()
    assert not node.paused_request_pending and node.paused
    resumed = node._start(SimpleNamespace(prompt="stack three block"), SimpleNamespace())
    assert resumed.accepted and node.chunk_executor.session != session


def test_invalid_image_logs_once_without_per_frame_spam(robot_node):
    node = robot_node
    invalid = SimpleNamespace(encoding="bgr8", height=192, width=256, step=768, data=bytes(192 * 256 * 3))
    node._image(invalid)
    node._image(invalid)
    warnings = [line for level, line in node.logger.lines if level == "warn"]
    assert len(warnings) == 1 and "Observation rejected" in warnings[0]


def test_snapshot_pairs_the_joint_sample_nearest_the_frame(robot_node):
    node = robot_node
    frame_age = 0.04
    node._image(_image_message(time.time() - frame_age))
    frame = time.monotonic() - frame_age
    for key, size in (("low", 35), ("left", 7), ("right", 7)):
        node.state_history[key].clear()
        node.state_history[key].append((frame - 0.010, _motors(size, -1.0)))
        node.state_history[key].append((frame + 0.020, _motors(size, 1.0)))
    _, state, _ = node._snapshot()
    assert state[0] == -1.0  # the sample closest to the frame, not the newest arrival
    assert node.pair_offsets["low"] == pytest.approx(0.010, abs=1e-3)
    node.state_history["low"].clear()
    node.state_history["low"].append((frame - 0.5, _motors(35, 1.0)))
    with pytest.raises(ValueError, match="within"):
        node._snapshot()


def test_exhausted_chunk_holds_the_last_target_until_the_late_chunk_lands(robot_node):
    node = robot_node
    node.server_status = "READY"
    stamp = _fresh_observation(node)
    node._start(SimpleNamespace(prompt="stack three block"), SimpleNamespace())
    session = node.chunk_executor.session
    first = np.repeat(np.arange(32, dtype=np.float32)[:, None] / 10, 28, axis=1)
    node.events.put(("actions", session, 0, stamp, first, time.monotonic()))
    node._tick()
    output = SimpleNamespace(calls=[], publish=lambda target, measured: output.calls.append(np.array(target, copy=True)))
    node.command_output = output
    node._request()
    assert node.inflight
    node.chunk_executor.started -= 2.0  # the chunk has run out; the next prediction is still in flight
    node._tick()
    np.testing.assert_allclose(output.calls[-1], first[31])
    assert node.chunk_executor.session == session and node.chunk_executor.held_since is not None
    late = np.full((32, 28), 2.0, np.float32)
    node.events.put(("actions", session, 1, stamp, late, time.monotonic()))
    node._tick()
    np.testing.assert_allclose(output.calls[-1], late[0])
    assert node.chunk_executor.held_since is None and not node.inflight
    text = "\n".join(line for _, line in node.logger.lines)
    assert "repeating the last target at 30 Hz" in text and "Hold ended after" in text


def test_resume_keeps_hold_until_fresh_chunk(robot_node):
    node = robot_node
    node.server_status = "READY"
    stamp = _fresh_observation(node)
    node._start(SimpleNamespace(prompt="stack three block"), SimpleNamespace())
    session = node.chunk_executor.session
    node.events.put(("actions", session, 0, stamp, np.full((32, 28), 0.05, np.float32), time.monotonic()))
    node._tick()
    assert node._pause(SimpleNamespace(), SimpleNamespace()).success
    wrong = node._start(SimpleNamespace(prompt="pour water"), SimpleNamespace())
    assert not wrong.accepted and node.paused
    resumed = node._start(SimpleNamespace(prompt="stack three block"), SimpleNamespace())
    assert resumed.accepted and not node.paused and node.hold_until_first_chunk
    assert node.chunk_executor.session != session
    node._tick()
    np.testing.assert_array_equal(node.pause_target, np.full(28, 0.05, np.float32))
    node._stop(SimpleNamespace(), SimpleNamespace())
