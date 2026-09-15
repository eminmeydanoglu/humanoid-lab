#!/usr/bin/env python3
"""Encode a prepared pilot's 1751D observation with the pinned SONIC v1.1 encoder.

Runs in the environment that pins onnxruntime (``./dev.sh sonic-encode`` uses the
sonic-sim venv).  Output is ``action_tokens.npz`` (``motion_token[64]`` plus the
7+7 Dex3 hand targets) and ``encoder_manifest.json`` with the model checksum, the
validated IO contract and the repeatability result.  A source whose hand schema is
unresolved writes the 64D body latent only and records why the 78D action is
blocked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.sonic.production import (  # noqa: E402
    PINNED_ENCODER_SHA256,
    encode_prepared_episode,
)

DEFAULT_MODEL_DIR = Path("/data/models/sonic-isaac/sonic_v1_1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--expected-sha256", default=PINNED_ENCODER_SHA256,
                        help="fail closed unless model_encoder.onnx has this checksum")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    args = parser.parse_args()

    pilot = args.pilot_dir.resolve()
    try:
        content = encode_prepared_episode(
            pilot,
            model_dir=args.model_dir,
            expected_sha256=args.expected_sha256,
            providers=args.providers,
            lock_path=ROOT / "versions.lock.yaml",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: content[key] for key in (
        "frames", "motion_token_dim", "action_dim", "final_action_78d", "finite", "repeatability",
        "token_norm")}, indent=2))
    print(f"tokens: {pilot / 'action_tokens.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
