#!/usr/bin/env python3
"""Bridge live Isaac observations to the pinned upstream GR00T PolicyServer."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cloudwalk_adapter import EMBODIMENT, PROMPT, validate_observation  # noqa: E402
from sonic_isaac_inspire_adapter import ContractError, split_groot_action_chunk  # noqa: E402


def _reply(socket: zmq.Socket, **value: object) -> None:
    socket.send_json(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="tcp://127.0.0.1:56110")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=56111)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.prompt != PROMPT:
        raise ContractError(f"CloudWalk checkpoint requires literal prompt {PROMPT!r}")

    from gr00t.policy.server_client import PolicyClient

    def prepare_observation_for_eval(state: np.ndarray, projected_gravity: np.ndarray) -> dict:
        # Exact UNITREE_G1_SONIC dataset replay groups from upstream vla_utils.
        return {
            "left_leg": state[..., 0:6], "right_leg": state[..., 6:12], "waist": state[..., 12:15],
            "left_arm": state[..., 15:22], "left_hand": state[..., 22:29], "right_arm": state[..., 29:36],
            "right_hand": state[..., 36:43], "projected_gravity": projected_gravity[np.newaxis, np.newaxis],
        }
    policy = PolicyClient(host=args.policy_host, port=args.policy_port)
    deadline = time.monotonic() + args.startup_timeout
    while not policy.ping():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"upstream GR00T PolicyServer is unavailable at {args.policy_host}:{args.policy_port}")
        time.sleep(1.0)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    print(json.dumps({"worker": "vla", "policy": "upstream PolicyClient", "prompt": args.prompt, "bind": args.bind}), flush=True)
    try:
        while True:
            request = socket.recv_json()
            if request == {"op": "stop"}:
                _reply(socket, ok=True)
                return 0
            try:
                rgb = np.frombuffer(base64.b64decode(request["rgb"], validate=True), dtype=np.uint8).reshape(480, 640, 3)
                state = np.asarray(request["state"], dtype=np.float32)
                base_quat = np.asarray(request["base_quat"], dtype=np.float64)
                validate_observation(rgb, state, args.prompt)
                if base_quat.shape != (4,) or not np.isfinite(base_quat).all():
                    raise ContractError("base quaternion must be four finite values")
                # Isaac and the upstream G1 sensor path both provide WXYZ base quaternions.
                w, x, y, z = base_quat
                gravity = np.asarray((2 * (x * z - w * y), 2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), dtype=np.float32)
                observation = {
                    "video": {"ego_view": rgb[np.newaxis, np.newaxis]},
                    "state": prepare_observation_for_eval(state[np.newaxis, np.newaxis], gravity),
                    "language": {"annotation.human.task_description": [[args.prompt]]},
                    "timestamps": float(request["timestamp"]),
                }
                action, _ = policy.get_action(observation)
                action.pop("task_progress", None)
                action.pop("action.task_progress", None)
                processed = {key.replace("action.", ""): value for key, value in action.items()}
                chunk = np.concatenate((processed["motion_token"], processed["left_hand_joints"], processed["right_hand_joints"]), axis=-1)
                if chunk.shape != (1, 40, 78) or not np.isfinite(chunk).all() or np.abs(chunk[:, :, :64]).max() > 1.25:
                    raise ContractError("upstream GR00T output is not a finite bounded [1,40,78] CloudWalk chunk")
                split_groot_action_chunk(chunk[0].tolist())
                _reply(socket, ok=True, sequence=request["sequence"], actions=chunk[0].tolist(), latency_seconds=time.monotonic() - float(request["sent_monotonic"]))
            except (KeyError, TypeError, ValueError, ContractError, zmq.ZMQError) as error:
                _reply(socket, ok=False, error=str(error))
    finally:
        socket.close(); context.term()


if __name__ == "__main__":
    raise SystemExit(main())
