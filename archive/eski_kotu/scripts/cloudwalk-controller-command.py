#!/usr/bin/env python3
"""Send a lifecycle command to the running CloudWalk Isaac simulator."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("start", "stop", "status"))
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:56115")
    args = parser.parse_args()
    try:
        import zmq
    except ModuleNotFoundError:
        sys.path.insert(0, "/opt/venvs/sonic-sim/lib/python3.11/site-packages")
        import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0); socket.setsockopt(zmq.SNDTIMEO, 3000); socket.setsockopt(zmq.RCVTIMEO, 3000)
    socket.connect(args.endpoint)
    try:
        socket.send_json({"op": args.operation})
        response = socket.recv_json()
        print(json.dumps(response, sort_keys=True))
        return 0 if response.get("ok") else 2
    finally:
        socket.close(); context.term()


if __name__ == "__main__":
    raise SystemExit(main())
