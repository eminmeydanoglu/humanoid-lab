#!/usr/bin/env python3
"""Replay offline SONIC motion tokens through the official Protocol v4 input."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

from humanoid_lab.datasets.sonic.protocol_v1 import pack_command_message
from humanoid_lab.datasets.sonic.protocol_v4 import pack_latent_action_message
from humanoid_lab.datasets.sonic.fidelity import simulation_frame_due


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", type=Path, help="action_tokens.npz produced by encode-sonic-episode.py")
    parser.add_argument("--endpoint", default="tcp://*:5556")
    parser.add_argument("--pre-roll", type=float, default=2.0)
    parser.add_argument("--post-roll", type=float, default=2.0)
    parser.add_argument("--stop-after", action="store_true")
    parser.add_argument("--timeline-output", type=Path, help="save publisher send times for A/B/C alignment")
    parser.add_argument("--sim-clock", type=Path, help="pace each token by Isaac simulation time")
    args = parser.parse_args()

    with np.load(args.tokens) as payload:
        arrays = {name: payload[name] for name in payload.files}
    tokens = np.asarray(arrays["motion_token"], dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 64 or len(tokens) < 2 or not np.isfinite(tokens).all():
        raise ValueError(f"motion_token must be finite [frames,64], got {tokens.shape}")
    frame_index = np.asarray(arrays.get("frame_index", np.arange(len(tokens))), dtype=np.int64).reshape(-1)
    if frame_index.shape != (len(tokens),) or np.any(np.diff(frame_index) != 1):
        raise ValueError("frame_index must be contiguous, strictly increasing, and match motion_token")

    left = arrays.get("left_hand_joints")
    right = arrays.get("right_hand_joints")
    if (left is None) != (right is None):
        raise ValueError("left and right hand targets must be present together")
    if left is not None:
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        if left.shape != (len(tokens), 7) or right.shape != (len(tokens), 7):
            raise ValueError("hand targets must be [frames,7]")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("hand targets contain NaN or Inf")

    def message(index: int) -> bytes:
        return pack_latent_action_message(
            tokens[index], frame_index[index],
            None if left is None else left[index],
            None if right is None else right[index],
        )

    context = zmq.Context.instance()

    def sim_clock() -> float | None:
        if args.sim_clock is None:
            return None
        try:
            return float(args.sim_clock.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None

    socket = context.socket(zmq.PUB)
    socket.bind(args.endpoint)
    debug = context.socket(zmq.SUB)
    debug.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
    debug.connect("tcp://localhost:5557")

    def latest_debug_token(wait_s: float) -> np.ndarray | None:
        deadline = time.monotonic() + wait_s
        result = None
        while time.monotonic() < deadline:
            if not debug.poll(max(0, int((deadline - time.monotonic()) * 1000))):
                break
            raw = debug.recv()
            payload = msgpack.unpackb(raw[len(b"g1_debug"):], raw=False)
            value = np.asarray(payload.get("token_state", []), dtype=np.float32)
            if value.shape == (64,) and np.isfinite(value).all():
                result = value
        return result

    try:
        time.sleep(1.0)
        planner_ready = None
        deadline = time.monotonic() + 8.0
        while planner_ready is None and time.monotonic() < deadline:
            socket.send(pack_command_message(start=True, stop=False, planner=True))
            planner_ready = latest_debug_token(0.1)
        if planner_ready is None:
            raise RuntimeError("official SONIC IDLE planner did not become token-ready")

        first = message(0)
        until = time.monotonic() + args.pre_roll
        while time.monotonic() < until:
            socket.send(first)
            time.sleep(0.02)
        for _ in range(10):
            socket.send(first)
            socket.send(pack_command_message(start=False, stop=False, planner=False))
            time.sleep(0.02)

        observed = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            socket.send(first)
            socket.send(pack_command_message(start=False, stop=False, planner=False))
            candidate = latest_debug_token(0.1)
            if candidate is not None and np.allclose(candidate, tokens[0], rtol=0.0, atol=1e-5):
                observed = candidate
                break
        if observed is None:
            raise RuntimeError("deployment did not report the staged offline token")

        started = time.monotonic()
        max_lateness = 0.0
        sent_wall_time_ns = np.empty(len(tokens), dtype=np.int64)
        sent_sim_s = np.empty(len(tokens), dtype=np.float64)
        start_sim = sim_clock()
        if args.sim_clock is not None and start_sim is None:
            raise RuntimeError(f"Isaac simulation clock is unavailable: {args.sim_clock}")
        for row in range(len(tokens)):
            if start_sim is not None:
                target_sim = start_sim + row / 50.0
                stalled_at = time.monotonic()
                previous_sim = None
                while True:
                    current_sim = sim_clock()
                    if current_sim is not None and simulation_frame_due(current_sim, start_sim, row):
                        break
                    if current_sim != previous_sim:
                        stalled_at = time.monotonic()
                        previous_sim = current_sim
                    if time.monotonic() - stalled_at > 5.0:
                        raise RuntimeError(f"Isaac simulation clock stalled before token {row}/{len(tokens)}")
                    time.sleep(0.002)
                sent_sim_s[row] = current_sim
                max_lateness = max(max_lateness, current_sim - target_sim)
            else:
                sent_sim_s[row] = np.nan
            sent_wall_time_ns[row] = time.time_ns()
            socket.send(message(row))
            if start_sim is None:
                deadline = started + (row + 1) / 50.0
                now = time.monotonic()
                max_lateness = max(max_lateness, now - deadline)
                time.sleep(max(0.0, deadline - now))

        if args.timeline_output is not None:
            args.timeline_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.timeline_output,
                frame_index=frame_index,
                sent_wall_time_ns=sent_wall_time_ns,
                sent_sim_s=sent_sim_s,
                expected_frames=np.asarray([len(tokens)], dtype=np.int64),
            )

        last = message(len(tokens) - 1)
        until = time.monotonic() + args.post_roll
        while time.monotonic() < until:
            socket.send(last)
            time.sleep(0.02)
        if args.stop_after:
            socket.send(pack_command_message(start=False, stop=True, planner=False))
            terminal_state = "stopped"
        else:
            for _ in range(10):
                socket.send(pack_command_message(start=False, stop=False, planner=True))
                time.sleep(0.02)
            terminal_state = "official_idle_planner"
        print(json.dumps({
            "backend": "offline SONIC v1.1 ONNX tokens via official Protocol v4",
            "frames": len(tokens),
            "hands": left is not None,
            "publish_hz": 50.0,
            "first_token_echo_max_abs": float(np.abs(observed - tokens[0]).max()),
            "max_publish_lateness_s": max(0.0, float(max_lateness)),
            "pacing_clock": "Isaac simulation time" if start_sim is not None else "wall time",
            "terminal_state": terminal_state,
        }, indent=2))
    except Exception:
        socket.send(pack_command_message(start=False, stop=True, planner=False))
        raise
    finally:
        socket.close(linger=0)
        debug.close(linger=0)


if __name__ == "__main__":
    main()
