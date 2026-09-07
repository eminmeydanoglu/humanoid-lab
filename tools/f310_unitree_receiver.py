#!/usr/bin/env python3
"""Loopback-only diagnostic receiver for f310_unitree_bridge.py.

It is intentionally a monitor: it does not import Unitree SDKs, DDS, or SONIC
and cannot send motor commands.  Run it on the G1 before installing/enabling
the SONIC adapter to prove the SSH tunnel, frame integrity, mapping, and stale
link behaviour.
"""

from __future__ import annotations

import argparse
import socket
import struct
import time

FRAME = struct.Struct("!4sBBHQ40s")
MAGIC, VERSION, FLAG_ARMED = b"F31B", 1, 0x01


def recv_exact(connection: socket.socket, size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def describe(raw: bytes) -> str:
    buttons = struct.unpack_from("<H", raw, 2)[0]
    lx, rx, ry, l2, ly = struct.unpack_from("<5f", raw, 4)
    return f"buttons=0x{buttons:04x} lx={lx:+.3f} ly={ly:+.3f} rx={rx:+.3f} ry={ry:+.3f} l2={l2:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=49051)
    parser.add_argument("--timeout-ms", type=int, default=250)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.timeout_ms < 50:
        parser.error("invalid port or timeout")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(1)
    listener.settimeout(1.0)
    print(f"F310 diagnostic receiver listening on 127.0.0.1:{args.port}; no DDS/motor output")
    latest_at, last_sequence, stale = 0.0, -1, True
    try:
        while True:
            try:
                connection, peer = listener.accept()
            except socket.timeout:
                if latest_at and not stale and time.monotonic() - latest_at > args.timeout_ms / 1000:
                    stale = True
                    print("STALE_LINK: no frame received; diagnostic receiver remains neutral")
                continue
            # Sequences are scoped to one bridge process. A restarted sender
            # reconnects at zero, so do not retain replay state across TCP
            # sessions.
            last_sequence = -1
            print(f"connected from {peer[0]}:{peer[1]}")
            with connection:
                while True:
                    connection.settimeout(args.timeout_ms / 1000)
                    try:
                        wire = recv_exact(connection, FRAME.size)
                    except socket.timeout:
                        stale = True
                        print("STALE_LINK: sender timed out; diagnostic receiver remains neutral")
                        break
                    if wire is None:
                        print("sender disconnected; diagnostic receiver remains neutral")
                        break
                    magic, version, flags, reserved, sequence, raw = FRAME.unpack(wire)
                    if magic != MAGIC or version != VERSION or reserved or sequence <= last_sequence:
                        print("rejected invalid, replayed, or malformed frame")
                        continue
                    last_sequence, latest_at, stale = sequence, time.monotonic(), False
                    state = "ARMED" if flags & FLAG_ARMED else "UNARMED"
                    print(f"frame={sequence} {state} {describe(raw)}")
    except KeyboardInterrupt:
        return 0
    finally:
        listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
