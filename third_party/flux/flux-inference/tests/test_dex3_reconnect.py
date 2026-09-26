"""A late GPU result cannot become a target after a local request timeout."""

import queue
import time

import numpy as np
import zmq
from flux_dex3 import protocol
from flux_dex3.network import NetworkWorker


def test_timeout_drops_late_result_and_reconnects():
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind("tcp://127.0.0.1:*")
    events = queue.Queue()
    worker = NetworkWorker(router.getsockopt_string(zmq.LAST_ENDPOINT), events, 0.25, "", "", "")
    worker.start()
    image, state = np.zeros((192, 256, 3), np.uint8), np.zeros(28, np.float32)
    try:
        assert router.poll(1500, zmq.POLLIN)
        address, *frames = router.recv_multipart()
        assert protocol.decode_request(frames)["type"] == "STATUS"
        router.send_multipart([address] + protocol.encode_status_reply("READY", "sha256:" + "0" * 64))
        assert events.get(timeout=1)[0] == "status"
        worker.submit("old", 0, time.time(), "stack three block", image, state)
        assert router.poll(1500, zmq.POLLIN)
        old_address, *frames = router.recv_multipart()
        assert protocol.decode_request(frames)["session_id"] == "old"
        failure = events.get(timeout=1.5)
        assert failure[:3] == ("error", "old", 0) and "timed out" in failure[3]
        router.send_multipart([old_address] + protocol.encode_predict_reply("old", 0, np.ones((32, 28), np.float32), 1))
        worker.submit("new", 0, time.time(), "stack three block", image, state)
        new_address = None
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and new_address is None:
            assert router.poll(1000, zmq.POLLIN)
            address, *frames = router.recv_multipart()
            request = protocol.decode_request(frames)
            if request["type"] == "STATUS":
                router.send_multipart([address] + protocol.encode_status_reply("READY", "sha256:" + "0" * 64))
            elif request["session_id"] == "new":
                new_address = address
        assert new_address is not None and new_address != old_address
        router.send_multipart([new_address] + protocol.encode_predict_reply("new", 0, np.zeros((32, 28), np.float32), 1))
        deadline = time.monotonic() + 2
        received = []
        while time.monotonic() < deadline:
            try:
                received.append(events.get(timeout=0.2))
            except queue.Empty:
                continue
            if received[-1][0] == "actions":
                break
        assert not any(event[0] == "actions" and event[1] == "old" for event in received)
        assert any(event[0] == "actions" and event[1] == "new" for event in received)
    finally:
        worker.close()
        worker.join(timeout=2)
        router.close()
        context.term()
