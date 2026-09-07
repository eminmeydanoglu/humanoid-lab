#!/usr/bin/env python3
"""Fail-closed verifier for two independently recorded CloudWalk GR00T replays."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cloudwalk_adapter import CHECKPOINT_REVISION, EMBODIMENT, PROMPT


def _records(path: Path) -> list[dict]:
    if not path.is_file(): raise ValueError(f"missing replay log: {path}")
    found = []
    for line in path.read_text(errors="replace").splitlines():
        try: value = json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(value, dict): found.append(value)
    return found


def _validate(path: Path) -> dict:
    records = _records(path)
    passed = [item for item in records if item.get("result") == "PASS"]
    if len(passed) != 1: raise ValueError(f"{path}: requires exactly one complete PASS record")
    item = passed[0]
    required = {"embodiment": EMBODIMENT, "literal_prompt": PROMPT, "checkpoint_revision": CHECKPOINT_REVISION, "shape": [1, 40, 78], "dtype": "float32", "finite": True}
    for key, expected in required.items():
        if item.get(key) != expected: raise ValueError(f"{path}: {key} is not {expected!r}")
    splits = item.get("splits")
    if not isinstance(splits, dict) or [splits.get(key) for key in ("motion_token", "left_hand_joints", "right_hand_joints")] != [[1,40,64],[1,40,7],[1,40,7]]:
        raise ValueError(f"{path}: missing exact 64+7+7 split")
    latency = item.get("latency_seconds")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(latency) or latency <= 0:
        raise ValueError(f"{path}: latency_seconds must be finite and positive")
    return {"log": str(path), "latency_seconds": latency}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("log", type=Path, nargs=2); parser.add_argument("--checkpoint-revision", default=CHECKPOINT_REVISION)
    args = parser.parse_args()
    if args.checkpoint_revision != CHECKPOINT_REVISION: raise ValueError("unexpected checkpoint revision")
    runs = [_validate(path) for path in args.log]
    if args.log[0].resolve() == args.log[1].resolve(): raise ValueError("two independent replay logs are required")
    print(json.dumps({"result":"PASS","embodiment":EMBODIMENT,"literal_prompt":PROMPT,"checkpoint_revision":CHECKPOINT_REVISION,"action_shape":[40,78],"split":[64,7,7],"runs":runs}, sort_keys=True))
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except ValueError as error: print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)
