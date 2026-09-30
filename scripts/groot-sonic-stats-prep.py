#!/usr/bin/env python3
"""Statistics preparation for the GR00T Unitree Dex3 / SONIC pack.

Two jobs, kept apart on purpose:

* ``--own`` recomputes this pipeline's own statistics
  (``meta/stats_groot_sonic.json``) from the produced parquet, so the file the
  converter wrote can be re-derived and checked after the fact. It is our
  summary of our own data and is labelled as such.
* ``--print-command`` (default) prints the official command that produces the
  loader-facing ``meta/stats.json`` and ``meta/relative_stats.json`` --
  ``gr00t/data/stats.py`` from the pinned Isaac-GR00T checkout, invoked as
  ``--dataset-path <split> --embodiment-tag UNITREE_G1_SONIC`` (both flags are
  required: the tool resolves its modality config from the tag, and it accepts
  the enum member name, not the lower-case value) -- and ``--run-official`` runs
  it, but only inside the training image where ``/opt/src/isaac-groot`` is
  mounted. This pipeline never writes those files itself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.groot_sonic.contract import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    official_stats_command,
)

#: Fields carrying the statistics the loader normalises, with their widths.
STAT_FIELDS = {
    "observation.state": 43,
    "observation.projected_gravity": 3,
    "action.motion_token": 64,
    "teleop.left_hand_joints": 7,
    "teleop.right_hand_joints": 7,
}


def split_dirs(root: Path, repos: tuple[str, ...], requested: list[str]) -> list[Path]:
    return [root / repo for repo in (requested or list(repos)) if (root / repo).is_dir()]


def stat_block(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def compute_own(split_dir: Path, fields: dict[str, int]) -> tuple[dict, int]:
    """Recompute the statistics of one split from its parquet files."""
    import pyarrow.parquet as pq

    collected: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    frames = 0
    for path in sorted((split_dir / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(path, columns=list(fields))
        frames += int(table.num_rows)
        for name in fields:
            collected[name].append(np.asarray(table.column(name).to_pylist(), dtype=np.float64))
    stats: dict[str, dict] = {}
    for name, width in fields.items():
        if not collected[name]:
            raise SystemExit(f"{split_dir}: no parquet episodes found")
        values = np.concatenate(collected[name], axis=0)
        if values.shape[1] != width:
            raise SystemExit(f"{split_dir}: {name} is {values.shape[1]}D, expected {width}D")
        stats[name] = stat_block(values)
    return stats, frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--root", type=Path, help="pack root holding the split directories")
    parser.add_argument("--split", action="append", choices=["train", "val"], default=[])
    parser.add_argument("--print-command", action="store_true", help="print the official stats.py command (default)")
    parser.add_argument("--own", action="store_true", help="recompute and rewrite meta/stats_groot_sonic.json")
    parser.add_argument("--check", action="store_true", help="compare the stored own stats with a fresh computation")
    parser.add_argument("--run-official", action="store_true", help="run the official tool in the training image")
    parser.add_argument("--official-python", default="python3", help="interpreter that has Isaac-GR00T installed")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.root) if args.root else config.output_root
    splits = split_dirs(root, (config.train_repo, config.val_repo), args.split)
    if not splits:
        raise SystemExit(f"no split directories under {root}")

    if args.own or args.check:
        for split_dir in splits:
            stats, frames = compute_own(split_dir, STAT_FIELDS)
            path = split_dir / f"meta/{config.stats_own_file}"
            if args.check:
                stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
                if stored != stats:
                    raise SystemExit(
                        f"{path}: the stored statistics do not match a fresh computation of {frames} frames"
                    )
                print(f"PASS {path} matches a fresh computation of {frames} frames")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(stats, indent=4, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {path} from {frames} frames")
    else:
        print("no statistics rewritten; pass --own to recompute meta/" + config.stats_own_file)

    for split_dir in splits:
        command = official_stats_command(config, split_dir, python=args.official_python)
        print("official statistics (not produced by this pipeline): " + " ".join(command))
        if not args.run_official:
            continue
        tool = Path(config.stats_official_tool)
        if not tool.is_file():
            raise SystemExit(
                f"the official tool is not mounted at {tool}; run this inside the training image "
                "(it reads the pack's modality.json and writes meta/stats.json)"
            )
        subprocess.run(command, check=True)
        print(f"ran the official tool for {split_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
