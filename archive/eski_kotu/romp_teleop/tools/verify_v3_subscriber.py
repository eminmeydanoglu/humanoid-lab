#!/usr/bin/env python3
"""Mock deploy: subscribe to the SONIC v3 'pose' topic and decode it exactly
like the C++ ZMQEndpointInterface (topic + 1280-byte JSON header + binary).

Use this to validate the bridge's wire format without the C++ deploy.

    python tools/verify_v3_subscriber.py --port 5556 --count 5
"""

from __future__ import annotations

import argparse
import json
import struct
import time

import numpy as np
import zmq

HEADER_SIZE = 1280
_DTYPES = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64}


def decode(msg: bytes, topic: str = "pose"):
    assert msg.startswith(topic.encode()), f"bad topic prefix: {msg[:8]!r}"
    rest = msg[len(topic):]
    header = json.loads(rest[:HEADER_SIZE].rstrip(b"\x00").decode("utf-8"))
    payload = rest[HEADER_SIZE:]
    out, off = {}, 0
    for f in header["fields"]:
        dt = _DTYPES[f["dtype"]]
        n = int(np.prod(f["shape"]))
        arr = np.frombuffer(payload, dtype=dt, count=n, offset=off)
        out[f["name"]] = arr.reshape(f["shape"]).copy()
        off += n * np.dtype(dt).itemsize
    return header, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{args.port}")
    sub.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
    sub.setsockopt(zmq.RCVTIMEO, int(args.timeout * 1000))
    print(f"[verify] subscribed tcp://localhost:{args.port} topic={args.topic}")

    got = 0
    try:
        while got < args.count:
            try:
                msg = sub.recv()
            except zmq.Again:
                print("[verify] timeout, no message")
                break
            header, fields = decode(msg, args.topic)
            got += 1
            print(f"[verify] msg {got}: v={header['v']} fields={list(fields)}")
            for k, v in fields.items():
                extra = ""
                if v.dtype.kind == "f":
                    extra = f" min={v.min():.4f} max={v.max():.4f}"
                print(f"         {k:12s} shape={v.shape} dtype={v.dtype}{extra}")
            q = fields.get("body_quat")
            if q is not None:
                print(f"         quat_norm={float(np.linalg.norm(q)):.5f}")
            if got >= args.count:
                break
            time.sleep(0.1)
    finally:
        sub.close(0)
        ctx.term()
    print(f"[verify] received {got} valid v3 message(s)")


if __name__ == "__main__":
    main()
