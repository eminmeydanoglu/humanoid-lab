#!/usr/bin/env python3
"""Produce one isolated v4 pose packet via the pinned upstream VLA packer."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import zmq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from sonic_isolated_lifecycle import (  # noqa: E402
    ACTION_RATE_HZ,
    INFERENCE_RATE_HZ,
    IsolatedLifecycle,
    State,
)


def load_upstream_packer(sonic_root: Path):
    path = sonic_root / "gear_sonic/scripts/run_vla_inference.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    if set(("InferenceConfig", "pack_latent_action_message")) - set(nodes):
        raise RuntimeError(f"pinned VLA serialization entry point changed: {path}")
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

    config_fields = {
        field.target.id: field.value
        for field in nodes["InferenceConfig"].body
        if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name) and field.value is not None
    }
    action_rate = eval(compile(ast.Expression(config_fields["action_publish_rate"]), str(path), "eval"), {"__builtins__": {}}, {})
    inference_rate = eval(compile(ast.Expression(config_fields["rate"]), str(path), "eval"), {"__builtins__": {}}, {})
    if action_rate != ACTION_RATE_HZ or inference_rate != INFERENCE_RATE_HZ:
        raise RuntimeError("upstream VLA rates are not 50 Hz action / 2.5 Hz inference")
    namespace = {"np": np, "pack_pose_message": pack_pose_message}
    exec(compile(ast.Module([nodes["pack_latent_action_message"]], []), str(path), "exec"), namespace)
    return namespace["pack_latent_action_message"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.reference.read_text())
    token = np.asarray(record["motion_token"], dtype=np.float32)
    left = np.asarray(record["left_hand"], dtype=np.float32)
    right = np.asarray(record["right_hand"], dtype=np.float32)
    if hashlib.sha256(args.decoder.read_bytes()).hexdigest() != record["decoder_sha256"]:
        raise RuntimeError("decoder hash does not match safe reference")

    lifecycle = IsolatedLifecycle()
    lifecycle.initialize(token)
    if lifecycle.hold(0.0) != tuple(token):
        raise RuntimeError("initial validated hold failed")
    lifecycle.start(1.0)
    lifecycle.validate_action(token, left, right, 1.0)
    lifecycle.pause()
    lifecycle.hold(1.1)
    lifecycle.start(1.2)
    lifecycle.validate_action(token, left, right, 1.2)
    if lifecycle.tick(1.2 + lifecycle.watchdog_seconds + 0.001) is not State.TIMED_OUT:
        raise RuntimeError("watchdog timeout failed")
    lifecycle.hold(1.6)
    lifecycle.stop()
    lifecycle.reset()
    if lifecycle.state is not State.RESET:
        raise RuntimeError("reset failed")

    pack = load_upstream_packer(args.sonic_root)
    packet = pack(token, np.array([43], dtype=np.int64), left, right)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://127.0.0.1:{args.port}")
    time.sleep(0.25)
    for _ in range(8):
        socket.send(packet)
        time.sleep(0.025)
    socket.close()
    context.term()
    print("PRODUCER upstream_pack_latent_action_message lifecycle=hold,running,pause,timeout,stop,reset rates=50,2.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
