#!/usr/bin/env python3
"""Gate 0: build the deterministic episode-level train/validation split.

Reads the machine-readable conversion contract, checks that the frozen sources
still hold the declared collections and exclusions, and writes
``split_manifest.json`` (seed included) without touching any source data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.psi0_dex3.contract import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from humanoid_lab.datasets.psi0_dex3.split import build_split, write_split_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, help="split manifest path (default: <output root>/split_manifest.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = build_split(config)
    path = write_split_manifest(config, args.output)

    for name, entry in sorted(manifest["collections"].items()):
        print(
            f"{name}: {entry['usable']}/{entry['total']} usable -> "
            f"train {len(entry['train'])}, val {len(entry['val'])}"
        )
    totals = manifest["totals"]
    print(
        f"total: {totals['usable']} episodes -> train {totals['train']}, val {totals['val']} "
        f"({totals['val_fraction_realized']:.4%})"
    )
    print(f"seed: {manifest['seed']}")
    print(f"wrote {path}")

    failures = []
    if totals["usable"] != 3150:
        failures.append(f"expected 3150 usable episodes, got {totals['usable']}")
    if sum(1 for entry in manifest["collections"].values() if entry["val"]) != len(config.collections):
        failures.append("not every collection is represented in validation")
    for name, entry in manifest["collections"].items():
        if set(entry["train"]) & set(entry["val"]):
            failures.append(f"{name}: train and validation overlap")
        if sorted(entry["train"] + entry["val"]) != sorted(config.usable_episodes(name)):
            failures.append(f"{name}: the split does not cover the usable episodes")
        if set(entry["train"] + entry["val"]) & set(entry["excluded"]):
            failures.append(f"{name}: an excluded episode entered the split")
    print(json.dumps({"status": "FAIL" if failures else "PASS", "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
