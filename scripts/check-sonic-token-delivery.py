#!/usr/bin/env python3
"""Verify that the SONIC Protocol v4 receiver logged every latent frame."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Upstream writes from several threads to stdout without a logging mutex.  A
# valid receiver message can therefore be split by reset/debug output between
# its prefix and ``frame_index`` (seen reproducibly at a temporary-motion
# boundary).  The controller log has no other frame_index producer, so parse
# the stable field itself instead of requiring one physically intact line.
RECEIVED = re.compile(r"frame_index: (\d+)")


def receiver_coverage(log: Path, expected_frames: int) -> dict:
    if expected_frames < 2:
        raise ValueError("expected_frames must be at least two")
    indices = [int(value) for value in RECEIVED.findall(log.read_text(encoding="utf-8", errors="replace"))]
    received = set(indices)
    expected = set(range(expected_frames))
    missing = sorted(expected - received)
    extras = sorted(received - expected)
    complete = bool(indices and not missing and not extras)
    return {
        "expected_frames": expected_frames,
        "received_messages": len(indices),
        "received_unique_frames": len(received),
        "first_frame": min(indices) if indices else None,
        "last_frame": max(indices) if indices else None,
        "missing_frames": missing,
        "unexpected_frames": extras,
        "complete": complete,
        "source": str(log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = receiver_coverage(args.log, args.expected_frames)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("complete", "expected_frames", "received_unique_frames", "missing_frames")}, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
