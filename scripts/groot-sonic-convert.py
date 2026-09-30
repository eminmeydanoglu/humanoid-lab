#!/usr/bin/env python3
"""Convert Unitree Dex3 episodes into the GR00T N1.7 LeRobot v2.1 pack.

Mini mode (the default) converts the first ``--train-per-task``/``--val-per-task``
episodes of every task in both splits so the contract can be exercised before the
whole corpus runs; ``--all-episodes`` converts exactly the Psi0 split manifest.
Sources are only ever read, and the episode split is not re-decided: it is read
from the Psi0 ``split_manifest.json``.

The pack's loader-facing ``meta/stats.json`` is not written here -- it comes from
GR00T's own ``gr00t/data/stats.py`` later. This script prints that command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.groot_sonic.contract import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    official_stats_command,
)
from humanoid_lab.datasets.groot_sonic.convert import (  # noqa: E402
    convert_episode,
    load_sonic_episode,
    read_raw_episode,
    sonic_episode_path,
)
from humanoid_lab.datasets.groot_sonic.split import (  # noqa: E402
    assert_instructions_match_psi0,
    assert_manifest_matches_config,
    load_split_manifest,
    selection,
    selection_summary,
)
from humanoid_lab.datasets.groot_sonic.writer import SplitWriter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split-manifest", type=Path, help="Psi0 split manifest (default: the contract's)")
    parser.add_argument("--output-root", type=Path, help="pack root (default: <contract root>, or its -mini sibling)")
    parser.add_argument("--train-per-task", type=int, default=1, help="mini size per task (train)")
    parser.add_argument("--val-per-task", type=int, default=1, help="mini size per task (validation)")
    parser.add_argument("--all-episodes", action="store_true", help="convert the whole split manifest")
    parser.add_argument("--collection", action="append", default=[], help="restrict to this collection (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-encode clips that already exist")
    parser.add_argument("--summary", type=Path, help="conversion manifest path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    assert_instructions_match_psi0(config)

    manifest_path = Path(args.split_manifest) if args.split_manifest else config.split_manifest
    manifest = load_split_manifest(manifest_path)
    assert_manifest_matches_config(manifest, config)

    mini = not args.all_episodes
    output_root = Path(args.output_root) if args.output_root else (
        config.output_root if not mini else Path(f"{config.output_root}-mini")
    )
    episodes_map = selection(
        manifest,
        config,
        train_per_task=None if args.all_episodes else args.train_per_task,
        val_per_task=None if args.all_episodes else args.val_per_task,
        all_episodes=args.all_episodes,
    )
    collections = args.collection or list(config.collection_names)
    unknown = sorted(set(collections) - set(config.collection_names))
    if unknown:
        raise SystemExit(f"unknown collections: {unknown}")
    chosen = {
        split: {name: values for name, values in by_task.items() if name in collections}
        for split, by_task in episodes_map.items()
    }
    if not any(chosen[split] for split in chosen):
        raise SystemExit("nothing selected to convert")

    provenance = {
        "name": config.name,
        "mode": "full" if args.all_episodes else "mini",
        "config": {"path": str(config.path), "sha256": config.sha256},
        "split_manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "name": str(manifest.get("name")),
        },
        "sources": {"raw_root": str(config.raw_root), "sonic_root": str(config.sonic_root)},
        "fields": {
            "state": "observation.state",
            "gravity": "observation.projected_gravity",
            "motion_token": "action.motion_token",
            "left_hand": "teleop.left_hand_joints",
            "right_hand": "teleop.right_hand_joints",
            "camera": config.camera_target_key,
            "instruction": "annotation.human.task_description",
        },
        "hand_orders": {
            "state": {side: list(config.raw["state"]["hand_order"][side]) for side in ("left", "right")},
            "action": {side: list(config.raw["action"]["hand_order"][side]) for side in ("left", "right")},
        },
        "trim": {"trailing_invalid_rows": config.trailing_invalid_rows},
    }
    writers = {
        split: SplitWriter(output_root / split, config, split, force=args.force, provenance=provenance)
        for split in (config.train_repo, config.val_repo)
        if chosen[split]
    }

    episodes: list[dict] = []
    for split in (config.train_repo, config.val_repo):
        for name in collections:
            for episode in chosen[split].get(name, []):
                corpus_path = sonic_episode_path(config.sonic_root, name, episode)
                if not corpus_path.is_file():
                    raise SystemExit(f"missing frozen SONIC episode: {corpus_path}")
                raw = read_raw_episode(config, name, episode)
                converted = convert_episode(raw, load_sonic_episode(corpus_path), config)
                record = writers[split].add(converted)
                episodes.append(
                    {
                        "split": split,
                        "collection": name,
                        "source_episode_index": episode,
                        "episode_index": record.episode_index,
                        "rows": record.length,
                        "source_frames": record.source_frames,
                        "corpus_frames": record.corpus_frames,
                        "trimmed_rows": record.trimmed_rows,
                        "hand_repair": record.repair,
                        "corpus_hand_max_error_rad": record.corpus_hand_max_error,
                        "parquet": str(record.parquet.relative_to(output_root)),
                        "video": str(record.video.relative_to(output_root)),
                    }
                )
                print(
                    f"[{split}] {name} ep{episode:06d}: {record.length} rows "
                    f"({record.trimmed_rows} clamped tail rows dropped, "
                    f"{record.repair['invalid_source_samples']} repaired hand samples)",
                    flush=True,
                )

    splits = {split: writer.finalize() for split, writer in writers.items()}
    summary = {
        **provenance,
        "output_root": str(output_root),
        "selection": selection_summary(chosen),
        "splits": splits,
        "episodes": episodes,
        "hand_repair": {
            "invalid_source_samples": sum(entry["hand_repair"]["invalid_source_samples"] for entry in episodes),
            "channels": {
                channel: sum(entry["hand_repair"]["channels"].get(channel, 0) for entry in episodes)
                for channel in sorted({key for entry in episodes for key in entry["hand_repair"]["channels"]})
            },
        },
        "official_stats_commands": {
            split: official_stats_command(config, output_root / split) for split in splits
        },
    }
    summary_path = Path(args.summary) if args.summary else output_root / "conversion_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": str(summary_path), "splits": splits}, indent=2))
    print("official statistics are not written here; run later, in the training image:")
    for command in summary["official_stats_commands"].values():
        print(f"  {' '.join(command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
