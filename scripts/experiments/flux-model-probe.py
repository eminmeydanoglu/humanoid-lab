#!/usr/bin/env python3
"""Ask the GPU server what it predicts for the live simulator observation.

This sits between the model and the node's motor contract: it builds the same
28-value observation the node sends (from /lowstate and the hand states, paired
with the newest camera frame), predicts one chunk with the real protocol, and
reports where the chunk falls outside the motor config's joint limits -- the
exact check that decides whether the node may publish.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from flux_dex3 import protocol
from flux_dex3.command_output import load_motor_config
from flux_dex3.executor import ChunkExecutor
from flux_dex3.mapping import JOINT_NAMES, measured_state

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from unitree_hg.msg import HandState, LowState


def rgb_image(msg):
    if msg.encoding != "rgb8" or (msg.height, msg.width) not in ((480, 640), (192, 256)):
        raise ValueError("camera must provide RGB8 480x640 or 192x256")
    return np.ndarray((msg.height, msg.width, 3), dtype=np.uint8, buffer=bytes(msg.data),
                      strides=(msg.step, 3, 1)).copy(order="C")


class Probe(Node):
    def __init__(self, camera_topic):
        super().__init__("flux_model_probe")
        self.state = {}
        self.image = None
        self.create_subscription(Image, camera_topic, self._image, 2)
        self.create_subscription(LowState, "/lowstate", lambda msg: self._store("low", msg), 10)
        self.create_subscription(HandState, "/dex3/left/state", lambda msg: self._store("left", msg), 10)
        self.create_subscription(HandState, "/dex3/right/state", lambda msg: self._store("right", msg), 10)

    def _store(self, key, msg):
        self.state[key] = (msg, time.monotonic())

    def _image(self, msg):
        try:
            self.image = (rgb_image(msg), time.monotonic())
        except ValueError:
            pass

    def wait(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.image is not None and all(key in self.state for key in ("low", "left", "right")):
                return True
        return False


def predict(endpoint, state, image, prompt, timeout_s):
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    socket.connect(endpoint)
    try:
        frames = protocol.encode_predict_request("probe-" + str(int(time.time())), 0,
                                                 time.time(), prompt, image, state)
        socket.send_multipart(frames)
        reply = protocol.decode_reply(socket.recv_multipart())
    finally:
        socket.close(linger=0)
        context.term()
    if reply["type"] != "PREDICT":
        raise RuntimeError("unexpected reply: %s" % json.dumps({k: v for k, v in reply.items()
                                                                if k != "actions"}))
    return reply


def report(chunk, limits, names):
    low, high = np.asarray(limits, dtype=np.float64).T
    chunk_min = chunk.min(axis=0)
    chunk_max = chunk.max(axis=0)
    violations = []
    for index, name in enumerate(names):
        below = low[index] - chunk_min[index]
        above = chunk_max[index] - high[index]
        if below > 0 or above > 0:
            violations.append({
                "joint": name,
                "chunk_min": round(float(chunk_min[index]), 4),
                "chunk_max": round(float(chunk_max[index]), 4),
                "limit": [round(float(low[index]), 4), round(float(high[index]), 4)],
                "max_excess_rad": round(float(max(below, above)), 4),
            })
    return {
        "chunk_shape": list(chunk.shape),
        "chunk_min": [round(float(value), 4) for value in chunk_min],
        "chunk_max": [round(float(value), 4) for value in chunk_max],
        "violations": violations,
        "passes_node_gate": not violations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--camera-topic", default="/camera/color/image_raw")
    parser.add_argument("--motor-config", required=True)
    parser.add_argument("--prompt", default="Put the apple into the plate.")
    parser.add_argument("--wait-s", type=float, default=30.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    rclpy.init()
    node = None
    try:
        node = Probe(args.camera_topic)
        if not node.wait(args.wait_s):
            print(json.dumps({"error": "no fresh camera/joint observation within %.0fs" % args.wait_s}),
                  file=sys.stderr)
            return 2
        image, image_at = node.image
        state = np.asarray(measured_state(*(node.state[key][0] for key in ("low", "left", "right"))),
                           dtype=np.float32)
        reply = predict(args.model_endpoint, state, image, args.prompt, args.timeout_s)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

    config = load_motor_config(args.motor_config)
    names = JOINT_NAMES
    result = {
        "state": [round(float(value), 4) for value in state],
        "image_shape": list(image.shape),
        "image_age_s": round(time.monotonic() - image_at, 3),
        "inference_ms": round(float(reply["inference_ms"]), 1),
        "limits_check": report(reply["actions"], config["joint_limits_rad"], names),
    }
    # The node's own executor check, verbatim: this is what aborted the task.
    try:
        executor = ChunkExecutor(limits=config["joint_limits_rad"])
        executor.start("probe")
        executor.accept("probe", 0, 0.05, reply["actions"])
        result["node_gate"] = "accepted"
    except ValueError as exc:
        result["node_gate"] = "rejected: %s" % exc
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    return 0 if result["limits_check"]["passes_node_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
