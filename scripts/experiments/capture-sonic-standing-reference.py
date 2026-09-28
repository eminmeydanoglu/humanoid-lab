#!/usr/bin/env python3
"""Capture stable planner targets from SONIC's official g1_debug stream."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER
from humanoid_lab.datasets.sonic.joints import reorder
from humanoid_lab.datasets.sonic.protocol_v1 import pack_command_message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="IDLE trajectory duration retained after settling")
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--endpoint", default="tcp://*:5556")
    args = parser.parse_args()
    context = zmq.Context.instance(); socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"g1_debug"); socket.connect("tcp://localhost:5557")
    command = context.socket(zmq.PUB); command.bind(args.endpoint)
    # ZMQ PUB/SUB needs a short subscription warm-up. Repeating the idempotent
    # planner start also makes this capture self-contained and deterministic.
    time.sleep(1.0)
    samples = []; receipt_times = []; started = time.monotonic(); deadline = started + args.settle_seconds + args.seconds
    next_start = started
    while time.monotonic() < deadline:
        if time.monotonic() >= next_start:
            command.send(pack_command_message(start=True, stop=False, planner=True))
            next_start = time.monotonic() + 0.1
        if not socket.poll(100): continue
        raw = socket.recv(); payload = msgpack.unpackb(raw[len(b"g1_debug"):], raw=False)
        if len(payload.get("body_q_target", [])) == 29:
            samples.append(payload)
            receipt_times.append(time.monotonic() - started)
    socket.close(linger=0); command.close(linger=0)
    if len(samples) < 50:
        raise RuntimeError(f"only captured {len(samples)} planner samples")
    times = np.asarray(receipt_times)
    eligible = times >= args.settle_seconds
    if not np.any(eligible):
        raise RuntimeError("no planner samples remain after settling")
    source_t = times[eligible]
    source_t = source_t - source_t[0]
    target_t = np.arange(int(np.floor(min(args.seconds, source_t[-1]) * 50.0)) + 1) / 50.0
    if target_t[-1] < args.seconds - 0.04:
        raise RuntimeError(
            f"only {target_t[-1]:.3f}s of post-settle IDLE data arrived; requested {args.seconds:.3f}s"
        )
    def resample(key: str) -> np.ndarray:
        values = np.asarray([sample[key] for sample, keep in zip(samples, eligible) if keep], dtype=np.float64)
        return np.stack([np.interp(target_t, source_t, values[:, column]) for column in range(values.shape[1])], axis=1)
    # g1_debug exposes body_q_target after C++ has remapped the motion to
    # hardware/MuJoCo order.  SONIC motion files and Protocol v1 require the
    # original IsaacLab reference order, so undo that remap before persisting.
    q_debug_mujoco = resample("body_q_target")
    q = reorder(q_debug_mujoco, BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER)
    body_pos = resample("base_trans_target")
    body_quat = resample("base_quat_target")
    body_quat /= np.linalg.norm(body_quat, axis=1, keepdims=True)
    qd = np.zeros_like(q)
    qd[:-1] = np.diff(q, axis=0) * 50.0
    if np.any((body_pos[:, 2] < 0.65) | (body_pos[:, 2] > 1.05)) or np.max(np.abs(np.linalg.norm(body_quat, axis=1) - 1.0)) > 1e-3:
        raise RuntimeError("captured planner trajectory has an invalid standing body pose")
    args.output.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.output / "joint_pos.csv", q, delimiter=",")
    np.savetxt(args.output / "timestamps.csv", target_t, delimiter=",")
    np.savetxt(args.output / "joint_vel.csv", qd, delimiter=",")
    np.savetxt(args.output / "body_pos.csv", body_pos, delimiter=",")
    np.savetxt(args.output / "body_quat.csv", body_quat, delimiter=",")
    settled_range = np.ptp(q[:, :15], axis=0)
    provenance = {"source": "official SONIC g1_debug body_q_target", "captured_samples": len(samples),
        "source_joint_order": "MuJoCo/Unitree (g1_debug output)",
        "stored_joint_order": "official SONIC reference / IsaacLab",
        "selection": "complete post-settle IDLE trajectory resampled to 50 Hz", "settle_seconds": args.settle_seconds,
        "retained_frames": len(target_t), "retained_duration_s": float(target_t[-1]),
        "settled_max_lower_body_range_rad": float(np.max(settled_range)),
        "lower_body_is_time_series": True}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "metadata.txt").write_text("SONIC v1.1 resmî IDLE planner trajectory, mode_id=0, 50 Hz\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__": main()
