#!/usr/bin/env python3
"""Diagnostic: check whether the official SONIC deployment is publishing.

The deployment only emits ``g1_debug`` once control has started, so this is a
liveness probe for a deployment that is already under way (for example while a
replay is running), not a startup gate.
"""

from __future__ import annotations

import argparse
import sys
import time

import msgpack
import zmq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://localhost:5557")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--subscribe", default="g1_debug")
    args = parser.parse_args()
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, args.subscribe.encode())
    socket.connect(args.endpoint)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            if not socket.poll(500):
                continue
            raw = socket.recv()
            payload = msgpack.unpackb(raw[len(args.subscribe) :], raw=False)
            keys = ",".join(sorted(payload)[:6])
            print(f"deployment debug stream is live on {args.endpoint} ({keys})")
            return 0
        print(f"error: no {args.subscribe!r} sample within {args.timeout:.0f}s on {args.endpoint}", file=sys.stderr)
        return 1
    finally:
        socket.close(linger=0)


if __name__ == "__main__":
    raise SystemExit(main())
