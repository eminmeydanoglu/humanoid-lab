#!/usr/bin/env python3
"""Verify that the SONIC Protocol v4 receiver logged every latent frame."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RECEIVED = re.compile(r"Protocol v4: Received 64D token.*frame_index: (\d+)")


def receiver_coverage(log: Path, expected_frames: int) -> dict:
    if expected_frames < 2:
        raise ValueError("expected_frames must be at least two")
    indices = [int(match.group(1)) for line in log.open(encoding="utf-8", errors="replace")
               if (match := RECEIVED.search(line))]
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
