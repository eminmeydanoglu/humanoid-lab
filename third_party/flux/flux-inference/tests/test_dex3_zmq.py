"""Dex3 wire validation and ROUTER/DEALER tests without GPU or robot interfaces."""

import importlib
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "flux_dex3"))
protocol = importlib.import_module("flux_dex3.protocol")
server_module = importlib.import_module("examples.dex3.zmq_server")
Dex3Server = server_module.Dex3Server
checkpoint_identity = server_module.checkpoint_identity


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "immutable"
    root.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors", "policy_preprocessor.json",
                 "policy_postprocessor.json", "policy_preprocessor_step_0.safetensors",
                 "policy_postprocessor_step_0.safetensors"):
        path = root / name
        if name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            path.write_text(json.dumps({"steps": [{"state_file": name[:-5] + "_step_0.safetensors"}]}))
        else:
            path.write_bytes(name.encode())
        path.chmod(0o444)
    root.chmod(0o555)
    yield root
    root.chmod(0o755)


@pytest.fixture
def sample():
    return (np.arange(192 * 256 * 3, dtype=np.uint32).astype(np.uint8).reshape(192, 256, 3),
            np.linspace(-1, 1, 28, dtype=np.float32))


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeModel:
    def __init__(self):
        self.resets = 0
        self.calls = []
        self.block = None
        self.entered = threading.Event()

    def reset(self):
        self.resets += 1

    def predict(self, image, state, task):
        self.calls.append(task)
        self.entered.set()
        if self.block is not None and task != "stack three block":
            assert self.block.wait(5)
        assert image.dtype == np.uint8 and state.dtype == np.float32
        return np.full((32, 28), state[0], dtype=np.float32)


def client(context, number):
    sock = context.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.setsockopt(zmq.RCVHWM, 4)
    sock.setsockopt(zmq.SNDTIMEO, 1000)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.connect("tcp://127.0.0.1:{}".format(number))
    return sock


def exchange(sock, frames):
    sock.send_multipart(frames)
    return protocol.decode_reply(sock.recv_multipart())


def wait_status(sock, target):
    until = time.monotonic() + 3
    while time.monotonic() < until:
        reply = exchange(sock, protocol.encode_status_request())
        if reply["status"] == target:
            return reply
        time.sleep(0.02)
    pytest.fail("server did not enter " + target)


def run_server(checkpoint, loader):
    number = port()
    server = Dex3Server(str(checkpoint), port=number, model_loader=loader)
    ready = threading.Event()
    thread = threading.Thread(target=server.serve_forever, kwargs={"ready_event": ready}, daemon=True)
    thread.start()
    assert ready.wait(3)
    return server, thread, number


def test_wire_roundtrip_and_rejection(sample):
    image, state = sample
    for shape in protocol.IMAGE_SHAPES:
        rgb = np.zeros(shape, np.uint8)
        frames = protocol.encode_predict_request("session_1", 0, 123.25, "stack", rgb, state)
        decoded = protocol.decode_request(frames)
        np.testing.assert_array_equal(decoded["image"], rgb)
        np.testing.assert_array_equal(decoded["state"], state)
    actions = np.arange(32 * 28, dtype=np.float32).reshape(32, 28)
    reply = protocol.decode_reply(protocol.encode_predict_reply("session_1", 0, actions, 13.5))
    np.testing.assert_array_equal(reply["actions"], actions)
    assert reply["action_dtype"] == "<f4"
    assert protocol.decode_request(protocol.encode_status_request())["type"] == "STATUS"
    assert protocol.decode_reply(protocol.encode_status_reply("LOADING"))["status"] == "LOADING"
    frames = protocol.encode_predict_request("session_1", 0, 0, "stack", image, state)
    bad_meta = dict(json.loads(frames[0]))
    for name, value in (("version", 2), ("type", "OTHER"), ("image_shape", [640, 480, 3]),
                        ("state_dtype", ">f4"), ("task", ""), ("seq", True),
                        ("session_id", "bad session"), ("observation_timestamp", -1)):
        altered = dict(bad_meta, **{name: value})
        with pytest.raises(protocol.ProtocolError):
            protocol.decode_request([json.dumps(altered).encode(), *frames[1:]])
    for corrupted in ([frames[0]], [frames[0], frames[1][:-1], frames[2]],
                      [frames[0], frames[1], frames[2][:-1]],
                      [frames[0], frames[1], frames[2], b"extra"],
                      [b'{"version":1,"version":1,"type":"STATUS"}']):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode_request(corrupted)
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_request([frames[0], frames[1], np.full(28, np.nan, dtype="<f4").tobytes()])
    with pytest.raises(protocol.ProtocolError):
        protocol.encode_predict_request("session_1", 0, float("nan"), "stack", image, state)
    with pytest.raises(protocol.ProtocolError):
        protocol.encode_predict_reply("session_1", 0, np.full((32, 28), np.nan, np.float32), 1)
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_reply([protocol.encode_predict_reply("session_1", 0, actions, 1)[0], b"short"])
    with pytest.raises(protocol.ProtocolError):
        protocol.decode_request([b"x" * (protocol.MAX_METADATA + 1)])


def test_server_status_busy_sessions_and_roundtrip(checkpoint, sample):
    model = FakeModel()
    gate = threading.Event()

    def load(path, *, device):
        assert path == str(checkpoint) and device == "cuda"
        assert gate.wait(5)
        return model

    server, thread, number = run_server(checkpoint, load)
    context = zmq.Context()
    sock = client(context, number)
    try:
        assert exchange(sock, protocol.encode_status_request())["status"] == "LOADING"
        malformed = exchange(sock, [b'{"version":2,"type":"STATUS"}'])
        assert malformed["type"] == "ERROR"
        image, state = sample
        frames = protocol.encode_predict_request("one", 0, 12, "reach", image, state)
        assert exchange(sock, frames)["type"] == "ERROR"
        gate.set()
        status = wait_status(sock, "READY")
        assert status["checkpoint"] == checkpoint_identity(checkpoint)
        assert model.resets == 1 and model.calls == ["stack three block"]
        model.block = threading.Event()
        sock.send_multipart(frames)
        assert model.entered.wait(2)
        second = client(context, number)
        try:
            busy = exchange(second, frames)
            assert busy["type"] == "ERROR" and "pending" in busy["error"]
            assert exchange(second, protocol.encode_status_request())["status"] == "READY"
        finally:
            second.close()
        model.block.set()
        result = protocol.decode_reply(sock.recv_multipart())
        assert result["session_id"] == "one" and result["seq"] == 0
        np.testing.assert_array_equal(result["actions"], state[0])
        assert model.resets == 2
        assert "seq" in exchange(sock, frames)["error"]
        assert exchange(sock, protocol.encode_predict_request("one", 2, 13, "reach", image, state))["type"] == "ERROR"
        assert exchange(sock, protocol.encode_predict_request("two", 1, 13, "reach", image, state))["type"] == "ERROR"
        assert exchange(sock, protocol.encode_predict_request("two", 0, 14, "reach", image, state))["type"] == "PREDICT"
        assert model.resets == 3
        assert "stale session" in exchange(sock, frames)["error"]
        assert model.resets == 3
    finally:
        model.block.set()
        gate.set()
        server.stop()
        thread.join(3)
        sock.close()
        context.term()
    assert not thread.is_alive()


def test_load_error_still_serves_status(checkpoint):
    def fail(path, *, device):
        raise RuntimeError("deliberate load failure")

    server, thread, number = run_server(checkpoint, fail)
    context = zmq.Context()
    sock = client(context, number)
    try:
        result = wait_status(sock, "ERROR")
        assert "deliberate" in result["error"]
        assert exchange(sock, protocol.encode_status_request())["status"] == "ERROR"
        assert exchange(sock, protocol.encode_predict_request("one", 0, 1, "test",
                                                              np.zeros((192, 256, 3), np.uint8),
                                                              np.zeros(28, np.float32)))["type"] == "ERROR"
    finally:
        server.stop()
        thread.join(3)
        sock.close()
        context.term()


def test_prediction_error_keeps_status_available(checkpoint, sample):
    class BrokenModel(FakeModel):
        def predict(self, image, state, task):
            if task != "stack three block":
                raise RuntimeError("bad prediction")
            return super().predict(image, state, task)

    server, thread, number = run_server(checkpoint, lambda path, *, device: BrokenModel())
    context = zmq.Context()
    sock = client(context, number)
    try:
        wait_status(sock, "READY")
        image, state = sample
        error = exchange(sock, protocol.encode_predict_request("one", 0, 1, "reach", image, state))
        assert error["type"] == "ERROR" and "bad prediction" in error["error"]
        assert "bad prediction" in wait_status(sock, "ERROR")["error"]
    finally:
        server.stop()
        thread.join(3)
        sock.close()
        context.term()


def test_immutable_checkpoint_and_nonloopback_security(checkpoint):
    assert checkpoint_identity(checkpoint).startswith("sha256:")
    (checkpoint / "adapter_config.json").chmod(0o644)
    with pytest.raises(ValueError, match="writable"):
        checkpoint_identity(checkpoint)
    with pytest.raises(ValueError, match="wildcard"):
        Dex3Server(str(checkpoint), bind_ip="0.0.0.0", server_secret_key="secret", client_keys_dir="keys")
    with pytest.raises(ValueError, match="CURVE"):
        Dex3Server(str(checkpoint), bind_ip="192.0.2.1")
