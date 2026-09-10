#!/usr/bin/env python3
"""Run the SONIC C++ deploy with --input-type zmq under a PTY.

The deploy only accepts its keyboard commands on a real TTY and needs
ENTER (toggle to the ZMQ stream) + ']' (arm control) to start consuming
Protocol-v3 SMPL poses. This wrapper spawns it on a pty, mirrors its output to
stdout/VRAM log, and injects those keystrokes once it is initialised.

Usage (inside the container, deployed libs on LD_LIBRARY_PATH):
  python run_deploy_zmq.py [--enter-at 6] [--arm-at 8] [--log /tmp/deploy_zmq.log]
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import signal
import subprocess
import sys
import time

BINARY = os.environ.get("DEPLOY_BINARY", "/data/models/sonic-deploy/g1_deploy_onnx_ref")
MODELS = os.environ.get("SONIC_DEPLOY_MODELS", "/data/models/sonic-isaac/sonic_v1_1")
REFERENCE = os.environ.get("SONIC_DEPLOY_REFERENCE", "/data/models/sonic-isaac/reference/example")
PLANNER = os.environ.get(
    "SONIC_DEPLOY_PLANNER", "/data/models/sonic-isaac/planner/target_vel/V2/planner_sonic.onnx")


def build_argv(port: int, out_port: int) -> list[str]:
    return [
        BINARY, "lo", f"{MODELS}/model_decoder.onnx", REFERENCE,
        "--obs-config", f"{MODELS}/observation_config.yaml",
        "--encoder-file", f"{MODELS}/model_encoder.onnx",
        "--planner-file", PLANNER,
        "--input-type", "zmq",
        "--zmq-host", "localhost", "--zmq-port", str(port), "--zmq-topic", "pose",
        "--zmq-out-port", str(out_port),
        "--disable-crc-check",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--out-port", type=int, default=5559)
    ap.add_argument("--enter-at", type=float, default=0.0,
                    help="seconds after init to send ENTER (0 = auto-detect)")
    ap.add_argument("--arm-at", type=float, default=0.0,
                    help="seconds after ENTER to send ']' (0 = ENTER+3s)")
    ap.add_argument("--log", default="/tmp/deploy_zmq.log")
    ap.add_argument("--print", action="store_true", help="mirror deploy output to stdout")
    args = ap.parse_args()

    argv = build_argv(args.port, args.out_port)
    print("[deploy-pty] exec:", " ".join(argv), flush=True)

    master, slave = pty.openpty()
    logf = open(args.log, "wb")
    proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave,
                            start_new_session=True, env=dict(os.environ))
    os.close(slave)

    start = time.time()
    buf = bytearray()
    enter_sent = False
    arm_time = None
    running = True
    try:
        while running:
            r, _, _ = select.select([master], [], [], 0.2)
            if r:
                try:
                    data = os.read(master, 8192)
                except OSError:
                    break
                if not data:
                    break
                logf.write(data)
                logf.flush()
                buf += data
                if args.print:
                    sys.stdout.write(data.decode(errors="replace"))
                    sys.stdout.flush()
                if len(buf) > 200000:
                    del buf[:-100000]

            now = time.time()
            if not enter_sent:
                if (args.enter_at and now - start >= args.enter_at) or \
                   (not args.enter_at and b"Press ENTER to toggle" in buf) or \
                   (not args.enter_at and now - start >= 8.0):
                    os.write(master, b"\n")
                    enter_sent = True
                    arm_time = now + (args.arm_at if args.arm_at else 3.0)
                    print("[deploy-pty] sent ENTER (toggle to ZMQ stream)", flush=True)
            if enter_sent and arm_time and now >= arm_time:
                os.write(master, b"]")
                print("[deploy-pty] sent ']' (arm control)", flush=True)
                arm_time = None

            if proc.poll() is not None:
                running = False
    except KeyboardInterrupt:
        print("[deploy-pty] interrupted", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        logf.close()
        print("[deploy-pty] stopped", flush=True)


if __name__ == "__main__":
    main()
