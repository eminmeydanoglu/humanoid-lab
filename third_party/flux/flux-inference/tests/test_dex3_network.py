"""The Foxy-side worker exchanges multipart frames without importing ROS."""

import queue
import time

import numpy as np
import zmq
from flux_dex3 import protocol
from flux_dex3.network import NetworkWorker


def test_network_worker_status_and_predict_round_trip():
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    router.bind("tcp://127.0.0.1:*")
    endpoint = router.getsockopt_string(zmq.LAST_ENDPOINT)
    events = queue.Queue()
    worker = NetworkWorker(endpoint, events, 1.0, "", "", "")
    worker.start()
    try:
        assert router.poll(2500, zmq.POLLIN)
        address, *frames = router.recv_multipart()
        assert protocol.decode_request(frames)["type"] == "STATUS"
        router.send_multipart([address] + protocol.encode_status_reply("READY", "sha256:" + "0" * 64))
        assert events.get(timeout=2)[1]["status"] == "READY"
        worker.submit("episode", 0, time.time(), "stack three block",
                      np.zeros((192, 256, 3), np.uint8), np.zeros(28, np.float32))
        assert router.poll(2500, zmq.POLLIN)
        address, *frames = router.recv_multipart()
        req = protocol.decode_request(frames)
        assert req["type"] == "PREDICT" and req["session_id"] == "episode" and req["seq"] == 0
        router.send_multipart([address] + protocol.encode_predict_reply("episode", 0, np.ones((32, 28), np.float32), 5.0))
        result = events.get(timeout=2)
        assert result[:3] == ("actions", "episode", 0)
        np.testing.assert_array_equal(result[4], 1)
    finally:
        worker.close()
        worker.join(timeout=2)
        router.close()
        context.term()
    assert not worker.is_alive()
