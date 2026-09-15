#!/usr/bin/env python3
"""Validate pinned SONIC contracts and source metadata without converting data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humanoid_lab.datasets.sonic.adapters.nvidia_fruits import action_slices
from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import validate_metadata
from humanoid_lab.datasets.sonic.provenance import assert_processed_destination, sonic_model_checksums


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/data/datasets/first_tur_ham"))
    parser.add_argument("--processed-root", type=Path, default=Path("/data/datasets/first_tur_processed/sonic_v1_1"))
    parser.add_argument("--models", type=Path, default=Path("/data/models/sonic/sonic_v1_1"))
    args = parser.parse_args()
    assert_processed_destination(args.raw_root, args.processed_root)
    unitree = args.raw_root / "unitree-g1-dex3/G1_Dex3_ObjectPlacement_Dataset"
    fruits = args.raw_root / "nvidia-g1-fruits-1k/g1-pick-apple"
    report = {
        "result": "PASS",
        "sonic_model_checksums": sonic_model_checksums(args.models),
        "unitree_object_placement": validate_metadata(unitree),
        "fruits_pick_apple_action_slices": {
            name: [value.start, value.stop] for name, value in action_slices(fruits).items()
        },
        "apple_hand_schema": "unresolved",
        "bulk_conversion_allowed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
