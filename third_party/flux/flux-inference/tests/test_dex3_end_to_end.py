"""GPU ROUTER through the robot DEALER to the command-disabled local scheduler."""

import json
import queue
import socket
import threading
import time

import numpy as np
from flux_dex3.executor import ChunkExecutor
from flux_dex3.mapping import split_action
from flux_dex3.network import NetworkWorker

from examples.dex3.zmq_server import Dex3Server


class FakeModel:
    def reset(self):
        pass

    def predict(self, image, state, task):
        assert image.shape == (192, 256, 3) and state.shape == (28,)
        return np.repeat(state[None, :], 32, axis=0).astype(np.float32)


def test_gpu_to_robot_dry_run(tmp_path):
    checkpoint = tmp_path / "immutable"
    checkpoint.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors", "policy_preprocessor.json",
                 "policy_postprocessor.json", "policy_preprocessor_step_0.safetensors",
                 "policy_postprocessor_step_0.safetensors"):
        file = checkpoint / name
        if name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            file.write_text(json.dumps({"steps": [{"state_file": name[:-5] + "_step_0.safetensors"}]}))
        else:
            file.write_bytes(b"fixture")
        file.chmod(0o444)
    checkpoint.chmod(0o555)
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    server = Dex3Server(str(checkpoint), port=port, model_loader=lambda *args, **kwargs: FakeModel())
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    events = queue.Queue()
    client = NetworkWorker("tcp://127.0.0.1:{}".format(port), events, 1.0, "", "", "")
    serving.start()
    client.start()
    try:
        ready = False
        until = time.monotonic() + 4
        while time.monotonic() < until and not ready:
            try:
                event = events.get(timeout=max(0.01, until - time.monotonic()))
            except queue.Empty:
                break
            ready = event[0] == "status" and event[1]["status"] == "READY"
        assert ready, (server.status, server.error, client.is_alive(), serving.is_alive())
        state = np.arange(28, dtype=np.float32) / 20
        client.submit("episode", 0, time.time(), "stack three block",
                      np.zeros((192, 256, 3), np.uint8), state)
        until = time.monotonic() + 3
        result = None
        while time.monotonic() < until:
            event = events.get(timeout=max(0.01, until - time.monotonic()))
            if event[0] == "actions":
                result = event
                break
        assert result is not None and result[1:3] == ("episode", 0)
        executor = ChunkExecutor()
        executor.start("episode")
        executor.accept("episode", 0, 0.05, result[4])
        target = executor.tick()
        arm, left, right = split_action(target)
        assert arm[15] == state[0] and left[0] == state[14] and right[6] == state[27]
        executor.stop()
        assert executor.tick() is None
    finally:
        client.close()
        client.join(timeout=2)
        server.stop()
        serving.join(timeout=2)
        checkpoint.chmod(0o755)
