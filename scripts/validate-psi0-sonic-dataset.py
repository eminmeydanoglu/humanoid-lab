#!/usr/bin/env python3
"""Validate a produced Psi0 Unitree Dex3 / SONIC v1.1 LeRobot dataset.

Every episode is re-derived from the frozen raw collection and the frozen 50 Hz
SONIC corpus, then compared field by field against what is on disk. The run also
opens each split with the pinned LeRobot loader so "the video opens in the ready
LeRobot tooling" is checked rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.psi0_dex3.contract import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from humanoid_lab.datasets.psi0_dex3.split import assert_manifest_matches_config, load_split_manifest  # noqa: E402
from humanoid_lab.datasets.psi0_dex3.validate import validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--root", type=Path, help="dataset root holding the split directories")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", action="append", choices=["train", "val"], default=[])
    parser.add_argument("--no-lerobot", action="store_true", help="skip opening the split with LeRobot")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="also require every episode of the split manifest to be present (full corpus run)",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quiet", action="store_true", help="print only the failing checks")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.root) if args.root else config.output_root
    manifest_path = Path(args.split_manifest) if args.split_manifest else root / "split_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"split manifest not found: {manifest_path}")
    manifest = load_split_manifest(manifest_path)
    assert_manifest_matches_config(manifest, config)

    report = validate_dataset(
        config,
        root,
        manifest,
        splits=tuple(args.split) or None,
        check_lerobot=not args.no_lerobot,
        require_complete=args.require_complete,
    )
    if not args.quiet:
        for entry in report["checks"]:
            print(f"{entry['status']:4} {entry['check']} {entry['detail']}".rstrip())
    else:
        for entry in report["failed"]:
            print(f"FAIL {entry['check']} {entry['detail']}".rstrip())
    print(
        json.dumps(
            {
                "status": report["status"],
                "root": report["root"],
                "episodes_checked": report["episodes_checked"],
                "frames_checked": report["frames_checked"],
                "failures": len(report["failed"]),
            }
        )
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
