"""Save head-camera frames published by ``./dev.sh flux-isaac`` (SONIC ego_view PUB).

Run inside the dev container with the Isaac venv (it has pyzmq):

    python scripts/experiments/grab_head_camera.py --out /outputs/realism/frame.jpg --wait 240

The payload is a small msgpack map whose ``images.ego_view`` value is a JPEG;
the JPEG is cut out by its SOI/EOI markers instead of decoding msgpack.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import zmq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait", type=float, default=240.0, help="seconds to wait for the first frame")
    parser.add_argument("--settle", type=float, default=5.0, help="seconds to keep reading after the first frame")
    args = parser.parse_args()

    socket = zmq.Context().socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.connect(args.endpoint)
    deadline = time.monotonic() + args.wait
    first = None
    payload = None
    while time.monotonic() < deadline:
        try:
            payload = socket.recv()
        except zmq.Again:
            continue
        if first is None:
            first = time.monotonic()
            deadline = first + args.settle
    if payload is None:
        print("no frame received")
        return 1
    start = payload.index(b"\xff\xd8")
    end = payload.rindex(b"\xff\xd9") + 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload[start:end])
    print(f"wrote {args.out} ({end - start} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
