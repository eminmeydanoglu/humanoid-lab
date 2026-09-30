#!/usr/bin/env python3
"""Validate a produced GR00T Unitree Dex3 / SONIC LeRobot v2.1 pack.

Every episode is re-derived from the frozen raw collection and the frozen 50 Hz
SONIC corpus with the validator's own name tables and interpolation, then
compared field by field against what is on disk. The run also opens each split
with the pinned LeRobot loader and, when the training image is mounted, matches
the pack's modality keys against the pre-registered ``UNITREE_G1_SONIC``
embodiment config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.groot_sonic.contract import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from humanoid_lab.datasets.groot_sonic.split import (  # noqa: E402
    assert_manifest_matches_config,
    load_split_manifest,
)
from humanoid_lab.datasets.groot_sonic.validate import validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--root", type=Path, help="pack root holding the split directories")
    parser.add_argument("--split-manifest", type=Path, help="Psi0 split manifest (default: the contract's)")
    parser.add_argument("--split", action="append", choices=["train", "val"], default=[])
    parser.add_argument("--no-lerobot", action="store_true", help="skip opening the splits with LeRobot")
    parser.add_argument("--no-groot", action="store_true", help="skip the pre-registered embodiment check")
    parser.add_argument("--groot-source", type=Path, help="pinned Isaac-GR00T checkout (default /opt/src/isaac-groot)")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="also require every episode of the split manifest to be present",
    )
    parser.add_argument("--report", type=Path, help="write the machine-readable report here")
    parser.add_argument("--quiet", action="store_true", help="print only failing and skipped checks")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.root) if args.root else config.output_root
    manifest_path = Path(args.split_manifest) if args.split_manifest else config.split_manifest
    manifest = load_split_manifest(manifest_path)
    assert_manifest_matches_config(manifest, config)

    report = validate_dataset(
        config,
        root,
        manifest,
        splits=tuple(args.split) or None,
        check_lerobot=not args.no_lerobot,
        check_groot=not args.no_groot,
        require_complete=args.require_complete,
        groot_source=args.groot_source,
    )
    if args.quiet:
        for entry in report["failed"] + report["skipped"]:
            print(f"{entry['status']:4} {entry['check']} {entry['detail']}".rstrip())
    else:
        for entry in report["checks"]:
            print(f"{entry['status']:4} {entry['check']} {entry['detail']}".rstrip())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": report["status"],
        "root": report["root"],
        "episodes_checked": report["episodes_checked"],
        "frames_checked": report["frames_checked"],
        "repaired_source_samples": report["repaired_source_samples"],
        "failures": [entry["check"] for entry in report["failed"]],
        "skipped": [entry["check"] for entry in report["skipped"]],
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
