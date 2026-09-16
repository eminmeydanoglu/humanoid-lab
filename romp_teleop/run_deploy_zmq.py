#!/usr/bin/env python3
"""Run the SONIC C++ deploy with --input-type zmq under a PTY.

The deploy only accepts its keyboard commands on a real TTY. Following NVIDIA's
documented operator order, it needs ']' (start control) then ENTER (toggle to
the ZMQ / SMPL stream) to start consuming Protocol-v3 poses. This wrapper spawns
it on a pty, mirrors its output to stdout/log, and injects those keystrokes once
it is initialised.

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
DEPLOY_LIB = os.environ.get("SONIC_DEPLOY_LIB", "/data/models/sonic-deploy/lib")


def build_argv(port: int, out_port: int, input_type: str = "zmq") -> list[str]:
    return [
        BINARY, "lo", f"{MODELS}/model_decoder.onnx", REFERENCE,
        "--obs-config", f"{MODELS}/observation_config.yaml",
        "--encoder-file", f"{MODELS}/model_encoder.onnx",
        "--planner-file", PLANNER,
        "--input-type", input_type,
        "--zmq-host", "localhost", "--zmq-port", str(port), "--zmq-topic", "pose",
        "--zmq-out-port", str(out_port),
        "--disable-crc-check",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--out-port", type=int, default=5559)
    ap.add_argument("--input-type", choices=["zmq", "zmq_manager"], default="zmq",
                    help="use zmq_manager for --control-mode upper-body")
    ap.add_argument("--enter-at", type=float, default=0.0,
                    help="seconds after init to send ENTER (0 = auto-detect)")
    ap.add_argument("--arm-at", type=float, default=0.0,
                    help="seconds after ']' to enable ZMQ (0 = 6s)")
    ap.add_argument("--drop-after-arm", type=float, default=2.0,
                    help="seconds after ']' to release MuJoCo's elastic band")
    ap.add_argument("--drop-marker", default="/tmp/romp_drop_robot",
                    help="marker consumed by sim_mujoco_domain42.py (empty disables)")
    ap.add_argument("--log", default="/tmp/deploy_zmq.log")
    ap.add_argument("--print", action="store_true", help="mirror deploy output to stdout")
    args = ap.parse_args()

    argv = build_argv(args.port, args.out_port, args.input_type)
    print("[deploy-pty] exec:", " ".join(argv), flush=True)

    if args.drop_marker:
        try:
            os.unlink(args.drop_marker)
        except FileNotFoundError:
            pass

    # The staged binary ships its ONNX Runtime shared library next to the
    # executable. Make the wrapper self-contained when the container entrypoint
    # did not export LD_LIBRARY_PATH.
    child_env = dict(os.environ)
    if os.path.isdir(DEPLOY_LIB):
        old_ld = child_env.get("LD_LIBRARY_PATH", "")
        child_env["LD_LIBRARY_PATH"] = ":".join(
            part for part in (DEPLOY_LIB, old_ld) if part
        )
        print(f"[deploy-pty] LD_LIBRARY_PATH={child_env['LD_LIBRARY_PATH']}", flush=True)

    master, slave = pty.openpty()
    logf = open(args.log, "wb")
    proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave,
                            start_new_session=True, env=child_env)
    os.close(slave)

    start = time.time()
    buf = bytearray()
    arm_sent = False
    drop_time = None
    enter_time = None
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
            # NVIDIA documented operator order: ']' to start control, then
            # ENTER to toggle into ZMQ (camera/SMPL) streaming.
            if not arm_sent:
                ready = (args.enter_at and now - start >= args.enter_at) or \
                        (not args.enter_at and b"Init Done" in buf)
                if ready:
                    os.write(master, b"]")
                    arm_sent = True
                    drop_time = now + args.drop_after_arm if args.drop_marker else None
                    enter_time = now + (args.arm_at if args.arm_at else 6.0)
                    print("[deploy-pty] sent ']' (arm control)", flush=True)
            if drop_time and now >= drop_time:
                fd = os.open(args.drop_marker, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                os.close(fd)
                print(f"[deploy-pty] requested MuJoCo drop via {args.drop_marker}", flush=True)
                drop_time = None
            if arm_sent and enter_time and now >= enter_time:
                os.write(master, b"\n")
                print("[deploy-pty] sent ENTER (toggle to ZMQ stream)", flush=True)
                enter_time = None

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
