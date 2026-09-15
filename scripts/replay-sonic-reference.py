#!/usr/bin/env python3
"""Publish a prepared episode to SONIC's official ZMQ Protocol v1 endpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import msgpack
import zmq

from humanoid_lab.datasets.sonic.protocol_v1 import pack_command_message, pack_pose_message, staged_windows
from humanoid_lab.datasets.sonic.schema import CanonicalEpisode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--endpoint", default="tcp://*:5556")
    parser.add_argument("--pre-roll", type=float, default=2.0)
    parser.add_argument("--post-roll", type=float, default=2.0)
    parser.add_argument("--stop-after", action="store_true",
                        help="stop SONIC after replay instead of returning to IDLE planner")
    parser.add_argument("--token-output", type=Path)
    args = parser.parse_args()
    data = np.load(args.reference)
    episode = CanonicalEpisode(**{name: data[name] for name in CanonicalEpisode.__dataclass_fields__})
    context = zmq.Context.instance(); socket = context.socket(zmq.PUB); socket.bind(args.endpoint)
    debug = context.socket(zmq.SUB); debug.setsockopt(zmq.SUBSCRIBE, b"g1_debug"); debug.connect("tcp://localhost:5557")
    tokens: list[np.ndarray] = []

    def latest_token(wait_s: float = 0.0) -> np.ndarray | None:
        result = None
        deadline = time.monotonic() + wait_s
        while debug.poll(max(0, int((deadline - time.monotonic()) * 1000)) if wait_s else 0):
            raw = debug.recv()
            payload = msgpack.unpackb(raw[len(b"g1_debug"):], raw=False)
            value = np.asarray(payload.get("token_state", []), dtype=np.float32)
            if value.shape == (64,) and np.isfinite(value).all(): result = value
            if result is not None: break
        return result
    try:
        time.sleep(1.0)
        windows = list(staged_windows(episode))
        first = pack_pose_message(windows[0][1], windows[0][0])
        # ZMQManager starts in planner mode. Start control there so the official
        # IDLE planner owns the safe takeover, then switch to streamed motion.
        # A streamed-mode start command is not consumed by the delegated pose
        # endpoint in the pinned upstream implementation.
        planner_ready = None
        ready_deadline = time.monotonic() + 5.0
        while planner_ready is None and time.monotonic() < ready_deadline:
            socket.send(pack_command_message(start=True, stop=False, planner=True))
            planner_ready = latest_token(0.1)
        if planner_ready is None:
            raise RuntimeError("official SONIC IDLE planner did not become token-ready")
        until = time.monotonic() + args.pre_roll
        while time.monotonic() < until: socket.send(first); time.sleep(0.02)
        for _ in range(10):
            socket.send(pack_command_message(start=False, stop=False, planner=False)); time.sleep(0.02)
        initial_token = None
        ready_deadline = time.monotonic() + 5.0
        while initial_token is None and time.monotonic() < ready_deadline:
            socket.send(first)
            socket.send(pack_command_message(start=False, stop=False, planner=False))
            initial_token = latest_token(0.1)
        if initial_token is None:
            raise RuntimeError("official C++ encoder did not become token-ready after explicit start")
        started = time.monotonic()
        for frame_index, window in windows:
            socket.send(pack_pose_message(window, frame_index))
            deadline = started + (int(frame_index[0]) + 1) / 50.0
            time.sleep(max(0.0, deadline - time.monotonic()))
            token = latest_token(0.5)
            if token is None:
                raise RuntimeError(f"missing official C++ token at replay frame {int(frame_index[0])}")
            tokens.append(token)
        last = pack_pose_message(windows[-1][1], windows[-1][0])
        until = time.monotonic() + args.post_roll
        while time.monotonic() < until: socket.send(last); time.sleep(0.02)
        if args.stop_after:
            socket.send(pack_command_message(start=False, stop=True, planner=False))
        else:
            # Keep one controller in charge after the episode. Switching back
            # to mode-0 IDLE is the safe reset for a free-standing robot.
            for _ in range(10):
                socket.send(pack_command_message(start=False, stop=False, planner=True))
                time.sleep(0.02)
        if args.token_output:
            args.token_output.parent.mkdir(parents=True, exist_ok=True)
            action = np.concatenate((np.asarray(tokens), episode.left_hand_joints, episode.right_hand_joints), axis=1)
            np.savez_compressed(args.token_output, motion_token=np.asarray(tokens), left_hand_joints=episode.left_hand_joints,
                right_hand_joints=episode.right_hand_joints, action=action, frame_index=np.arange(len(tokens), dtype=np.int64))
            metadata = {"backend": "official SONIC v1.1 C++ TensorRT encoder", "frames": len(tokens),
                "motion_token_dim": 64, "action_dim": 78, "finite": bool(np.isfinite(action).all()),
                "terminal_state": "stopped" if args.stop_after else "official_idle_planner"}
            args.token_output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        socket.close(linger=0)
        debug.close(linger=0)


if __name__ == "__main__":
    main()
