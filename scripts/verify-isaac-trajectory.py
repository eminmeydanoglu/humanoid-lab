#!/usr/bin/env python3
"""Compare recording-on/off Isaac G1 tensor trajectories by physics tick."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_metrics(run_dir: Path) -> dict[int, dict[str, Any]]:
    path = run_dir / "simulator-metrics.jsonl"
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[int(row["physics_tick"])] = row
    if not rows:
        raise ValueError(f"no simulator metrics in {path}")
    return rows


def compare(first: Path, second: Path, tolerance: float) -> dict[str, Any]:
    left = load_metrics(first)
    right = load_metrics(second)
    ticks = sorted(set(left) & set(right))
    if len(ticks) < 2:
        raise ValueError("runs do not contain at least two matching physics ticks")
    maximum = 0.0
    finite = True
    for tick in ticks:
        for field in ("root_position", "root_rotation_wxyz", "body_position", "body_velocity"):
            a, b = left[tick][field], right[tick][field]
            if len(a) != len(b):
                raise ValueError(f"{field} shape differs at tick {tick}")
            for lhs, rhs in zip(a, b, strict=True):
                finite = finite and math.isfinite(float(lhs)) and math.isfinite(float(rhs))
                maximum = max(maximum, abs(float(lhs) - float(rhs)))
    return {
        "result": "PASS" if finite and maximum <= tolerance else "FAIL",
        "recording_on_run": str(first),
        "recording_off_run": str(second),
        "compared_ticks": len(ticks),
        "last_compared_tick": ticks[-1],
        "maximum_absolute_tensor_error": maximum,
        "tolerance": tolerance,
        "finite": finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_on", type=Path)
    parser.add_argument("recording_off", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.recording_on, args.recording_off, args.tolerance)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
