#!/usr/bin/env python3
"""Gate 1: convert Unitree Dex3 episodes into the Psi0 30 Hz LeRobot dataset.

The default run converts a mini dataset (one or two episodes per task in each
split) so the 30 Hz contract can be validated before the whole corpus is
processed. Sources are only ever read; output goes to the contract's output
root (or ``--output-root``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.psi0_dex3.contract import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from humanoid_lab.datasets.psi0_dex3.convert import convert_episode, read_raw_episode  # noqa: E402
from humanoid_lab.datasets.psi0_dex3.split import assert_manifest_matches_config, load_split_manifest  # noqa: E402
from humanoid_lab.datasets.psi0_dex3.writer import SplitWriter  # noqa: E402


def sonic_episode_path(root: Path, collection: str, episode: int) -> Path:
    return Path(root) / collection / "episodes" / f"episode_{episode:06d}" / "action.npz"


def load_sonic_episode(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"action", "timestamp", "training_valid_mask"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path}: frozen SONIC episode is missing {missing}")
        return (
            np.asarray(payload["action"], dtype=np.float32),
            np.asarray(payload["timestamp"], dtype=np.float64),
            np.asarray(payload["training_valid_mask"], dtype=bool),
        )


def select(
    manifest: dict,
    *,
    collections: list[str],
    train_per_task: int | None,
    val_per_task: int | None,
    all_episodes: bool,
) -> dict[str, dict[str, list[int]]]:
    selection: dict[str, dict[str, list[int]]] = {"train": {}, "val": {}}
    for split, limit in (("train", train_per_task), ("val", val_per_task)):
        for name in collections:
            episodes = list(manifest["collections"][name][split])
            if all_episodes:
                chosen = episodes
            elif limit is None:
                chosen = []
            else:
                chosen = episodes[:limit]
            if chosen:
                selection[split][name] = chosen
    return selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--train-per-task", type=int, default=1, help="mini dataset size per task (train)")
    parser.add_argument("--val-per-task", type=int, default=1, help="mini dataset size per task (validation)")
    parser.add_argument("--all-episodes", action="store_true", help="convert the whole split manifest")
    parser.add_argument("--collection", action="append", default=[], help="restrict to this collection (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-encode videos that already exist")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(args.output_root) if args.output_root else config.output_root
    manifest_path = Path(args.split_manifest) if args.split_manifest else output_root / "split_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"split manifest not found: {manifest_path} (run scripts/psi0-build-split.py first)")
    manifest = load_split_manifest(manifest_path)
    assert_manifest_matches_config(manifest, config)

    collections = args.collection or list(config.collection_names)
    unknown = sorted(set(collections) - set(config.collection_names))
    if unknown:
        raise SystemExit(f"unknown collections: {unknown}")
    if args.all_episodes and (args.train_per_task != 1 or args.val_per_task != 1):
        raise SystemExit("--all-episodes cannot be combined with a per-task limit")
    selection = select(
        manifest,
        collections=collections,
        train_per_task=args.train_per_task,
        val_per_task=args.val_per_task,
        all_episodes=args.all_episodes,
    )
    if not selection["train"] and not selection["val"]:
        raise SystemExit("nothing selected to convert")

    writers = {
        split: SplitWriter(output_root / split, config, split, force=args.force)
        for split in ("train", "val")
        if selection[split]
    }
    episodes: list[dict] = []
    for split in ("train", "val"):
        for name in collections:
            for episode in selection[split].get(name, []):
                sonic_path = sonic_episode_path(config.sonic_root, name, episode)
                if not sonic_path.is_file():
                    raise SystemExit(f"missing frozen SONIC episode: {sonic_path}")
                raw = read_raw_episode(config, name, episode)
                action, timestamp, valid = load_sonic_episode(sonic_path)
                converted = convert_episode(raw, action, timestamp, valid, config)
                record = writers[split].add(converted)
                episodes.append(
                    {
                        "split": split,
                        "collection": name,
                        "source_episode_index": episode,
                        "episode_index": record.episode_index,
                        "frames": record.length,
                        "frames_valid": record.frames_valid,
                        "anchors_valid": record.anchors_valid,
                        "video_start_s": record.video_start_s,
                        "video_stop_s": record.video_stop_s,
                    }
                )
                print(
                    f"[{split}] {name} ep{episode:06d}: {record.length} frames, "
                    f"{record.frames_valid} valid, {record.anchors_valid} strict anchors",
                    flush=True,
                )

    summaries = {split: writer.finalize() for split, writer in writers.items()}
    summary = {
        "name": config.name,
        "config": {"path": str(config.path), "sha256": config.sha256},
        "split_manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
        "output_root": str(output_root),
        "model_contract": {
            "state_dim": config.raw["model_contract"]["state_dim"],
            "action_dim": config.raw["model_contract"]["action_dim"],
            "action_chunk_size": config.raw["model_contract"]["action_chunk_size"],
        },
        "fields": {
            "state": config.state_field,
            "body_token": config.body_token_field,
            "hand_action": config.action_field,
            "camera": config.camera_target_key,
            "instruction": "task_description",
        },
        "selection": {split: {name: len(values) for name, values in by_task.items()} for split, by_task in selection.items()},
        "splits": summaries,
        "episodes": episodes,
    }
    summary_path = Path(args.summary) if args.summary else output_root / "conversion_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": str(summary_path), "splits": summaries}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
