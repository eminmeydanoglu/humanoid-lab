#!/usr/bin/env python3
"""Run GR00T action scheduling separately from the Isaac Timeline process."""
from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from sonic_isaac_inspire_adapter import SafeReference, pack_protocol_v4, split_groot_action_chunk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-endpoint", default="tcp://127.0.0.1:56110")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:56114")
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:56112")
    parser.add_argument("--control-endpoint", default="tcp://127.0.0.1:56115")
    parser.add_argument("--observation-endpoint", default="tcp://127.0.0.1:56116")
    parser.add_argument("--safe-reference", type=Path, default=ROOT / "configs" / "sonic_v1_1_safe_standing_reference.json")
    parser.add_argument("--safe-reference-only", action="store_true", help="Run native SONIC hold without GR00T inference for stability testing.")
    args = parser.parse_args()

    try:
        import zmq
    except ModuleNotFoundError:
        sys.path.insert(0, "/opt/venvs/sonic-sim/lib/python3.11/site-packages")
        import zmq

    reference = SafeReference.load(args.safe_reference)
    reference.validate()
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: stop.set())

    context = zmq.Context()
    action_pub = context.socket(zmq.PUB); action_pub.setsockopt(zmq.LINGER, 0); action_pub.bind(args.action_endpoint)
    state_sub = context.socket(zmq.SUB); state_sub.setsockopt(zmq.LINGER, 0); state_sub.setsockopt(zmq.SUBSCRIBE, b""); state_sub.connect(args.state_endpoint)
    observation_sub = context.socket(zmq.SUB); observation_sub.setsockopt(zmq.LINGER, 0); observation_sub.setsockopt(zmq.SUBSCRIBE, b""); observation_sub.connect(args.observation_endpoint)
    control = context.socket(zmq.REQ); control.setsockopt(zmq.LINGER, 0); control.setsockopt(zmq.RCVTIMEO, 5000); control.setsockopt(zmq.SNDTIMEO, 5000); control.connect(args.control_endpoint)
    inference_requests: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=1)
    inference_results: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)

    def inference_worker() -> None:
        worker_context = zmq.Context()
        socket = worker_context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0); socket.setsockopt(zmq.RCVTIMEO, 30000); socket.setsockopt(zmq.SNDTIMEO, 30000)
        socket.connect(args.vla_endpoint)
        try:
            while not stop.is_set():
                item = inference_requests.get()
                if item is None:
                    return
                socket.send_json(item)
                reply = socket.recv_json()
                reply["inference_started_monotonic"] = item["sent_monotonic"]
                try:
                    inference_results.put_nowait(reply)
                except queue.Full:
                    inference_results.get_nowait(); inference_results.put_nowait(reply)
        finally:
            socket.close(); worker_context.term()

    worker = threading.Thread(target=inference_worker, name="cloudwalk-controller-groot", daemon=True)
    worker.start()
    time.sleep(0.5)
    control.send_json({"op": "start"})
    response = control.recv_json()
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "simulator rejected controller start"))
    print(json.dumps({"event": "controller_armed", **response}, sort_keys=True), flush=True)

    policy_activated = False
    if args.safe_reference_only:
        control.send_json({"op": "run"})
        activation = control.recv_json()
        if not activation.get("ok"):
            raise RuntimeError(activation.get("error", "simulator rejected safe-reference activation"))
        policy_activated = True
        print(json.dumps({"event": "controller_safe_reference_active", **activation}, sort_keys=True), flush=True)

    sequence = 0
    chunk = None
    chunk_index = 0
    inference_pending = False
    latest_observation = None
    poller = zmq.Poller()
    poller.register(state_sub, zmq.POLLIN)
    poller.register(observation_sub, zmq.POLLIN)
    try:
        while not stop.is_set():
            ready = dict(poller.poll(100))
            if observation_sub in ready:
                while True:
                    try:
                        latest_observation = observation_sub.recv_json(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
            try:
                reply = inference_results.get_nowait()
            except queue.Empty:
                reply = None
            if reply is not None:
                inference_pending = False
                if not reply.get("ok"):
                    raise RuntimeError(reply.get("error", "GR00T inference failed"))
                chunk = split_groot_action_chunk(reply["actions"])
                chunk_index = 0
                if not policy_activated:
                    control.send_json({"op": "run"})
                    activation = control.recv_json()
                    if not activation.get("ok"):
                        raise RuntimeError(activation.get("error", "simulator rejected policy activation"))
                    policy_activated = True
                    print(json.dumps({"event": "controller_policy_active", **activation}, sort_keys=True), flush=True)
                print(json.dumps({"event": "controller_groot_reply", "sequence": reply.get("sequence"), "chunk_index": chunk_index}, sort_keys=True), flush=True)
            if not args.safe_reference_only and latest_observation is not None and not inference_pending:
                request = dict(latest_observation)
                request["sent_monotonic"] = time.monotonic()
                inference_requests.put_nowait(request)
                inference_pending = True
                latest_observation = None
                print(json.dumps({"event": "controller_groot_request", "sequence": request.get("sequence")}, sort_keys=True), flush=True)
            if state_sub not in ready:
                continue
            while True:
                try:
                    state_sub.recv(flags=zmq.NOBLOCK)
                    sequence += 1
                except zmq.Again:
                    break
            action = reference.action if chunk is None else chunk[min(chunk_index, len(chunk) - 1)]
            action_pub.send(pack_protocol_v4(action, sequence - 1))
            if chunk is not None:
                chunk_index = min(chunk_index + 1, len(chunk) - 1)
    finally:
        try:
            control.send_json({"op": "stop"})
            reply = control.recv_json()
            print(json.dumps({"event": "controller_disarmed", **reply}, sort_keys=True), flush=True)
        except Exception as error:
            print(json.dumps({"event": "controller_disarm_failed", "error": str(error)}, sort_keys=True), flush=True)
        stop.set()
        try:
            inference_requests.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=2.0)
        action_pub.close(); state_sub.close(); observation_sub.close(); control.close(); context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
