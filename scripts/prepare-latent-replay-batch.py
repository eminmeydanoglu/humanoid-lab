#!/usr/bin/env python3
"""Turn production SONIC-latent episodes into pilot-shaped replay directories.

The bulk corpus keeps exactly one 78D ``action.npz`` per episode.  The replay
pipeline (``scripts/run-sonic-pilot-sim.sh``) consumes a pilot directory instead:
the canonical 50 Hz reference, the token file in its pilot layout, the recorded
head-camera clip, and the two manifests it validates against the pinned SONIC
bundle.  This script derives those from the stored artifacts.  No motion is
re-encoded and no token is recomputed: the 64D body latent and the 7+7 hand
targets are sliced straight out of the corpus action.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import load_episode  # noqa: E402
from humanoid_lab.datasets.sonic.pilot import (  # noqa: E402
    PilotSpec,
    extract_source_video,
    write_canonical_reference,
)
from humanoid_lab.datasets.sonic.provenance import sha256_file  # noqa: E402
from humanoid_lab.datasets.sonic.reference import deployment_standing_pose  # noqa: E402

FPS = 50.0


def sampled_episodes(corpus: Path, collection: str, *, per_collection: int,
                     max_seconds: float, seed: str) -> list[int]:
    """Random, re-drawable sample of one collection's episodes.

    Episodes longer than ``max_seconds`` are left out so one outlier cannot
    dominate the batch wall time; the sample is drawn per collection from a
    seed, so adding or dropping a collection does not move the other draws.
    """
    candidates: list[int] = []
    for manifest_path in sorted((corpus / collection / "episodes").glob("episode_*/manifest.json")):
        frames = int(json.loads(manifest_path.read_text())["frames"]["total"])
        if frames / FPS <= max_seconds:
            candidates.append(int(manifest_path.parent.name.split("_")[1]))
    if not candidates:
        raise SystemExit(f"{collection}: no episode at or below {max_seconds}s")
    if len(candidates) <= per_collection:
        return sorted(candidates)
    return sorted(random.Random(f"{seed}:{collection}").sample(candidates, per_collection))


def prepare(corpus: Path, raw_root: Path, output_root: Path, collection: str, episode: int,
            standing, stamp: str) -> dict:
    production_dir = corpus / collection / "episodes" / f"episode_{episode:06d}"
    production = json.loads((production_dir / "manifest.json").read_text())
    with np.load(production_dir / "action.npz") as payload:
        action = np.asarray(payload["action"], dtype=np.float32)
        timestamp = np.asarray(payload["timestamp"], dtype=np.float64)
        frame_index = np.asarray(payload["frame_index"], dtype=np.int64)
    if action.shape[1] != 78 or len(action) != len(timestamp):
        raise SystemExit(f"{production_dir}: unexpected action contract {action.shape}")

    run_dir = output_root / collection / f"episode_{episode:06d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        run_dir / "action_tokens.npz",
        motion_token=action[:, :64],
        left_hand_joints=action[:, 64:71],
        right_hand_joints=action[:, 71:78],
        frame_index=frame_index,
        timestamp=timestamp,
    )

    dataset = raw_root / collection
    spec = PilotSpec(kind="unitree", dataset=dataset, episode_index=episode,
                     pilot_name=f"unitree_{collection}_ep{episode:03d}")
    build = load_episode(dataset, episode, standing)
    write_canonical_reference(run_dir, build)
    source = extract_source_video(spec, run_dir / "source.mp4")

    frames = int(production["frames"]["total"])
    if frames != len(action):
        raise SystemExit(f"{run_dir}: frame count mismatch {frames} != {len(action)}")
    run_manifest = {
        "pilot": spec.pilot_name,
        "kind": "unitree",
        "dataset": str(dataset),
        "dataset_name": collection,
        "episode_index": episode,
        "frames": frames,
        "duration_s": frames / FPS,
        "source_fps": float(production["fps"]["source"]),
        "processed_fps": float(production["fps"]["target"]),
        "created_utc": stamp,
        "encoder_orientation_policy": "reference_root_current_frame",
        "final_action_78d": {"status": "available", "reason": "hand channels resolved"},
        "source_video": source,
        "notes": [
            "bulk-corpus episode: the reference is re-derived by the same adapter the corpus used, "
            "and the tokens are sliced out of the stored 78D action; nothing is re-encoded",
            "lower body, waist and root are synthetic for the whole episode (arm-only collection)",
        ],
        "production": {
            "action_npz": str(production_dir / "action.npz"),
            "action_npz_sha256": sha256_file(production_dir / "action.npz"),
            "manifest": str(production_dir / "manifest.json"),
            "converter_commit": production["converter"]["commit"],
            "code_bundle_sha256": production["converter"]["code_bundle_sha256"],
            "encoder_sha256": production["sonic"]["encoder_sha256"],
            "observation_config_sha256": production["sonic"]["observation_config_sha256"],
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")

    norms = np.linalg.norm(action[:, :64], axis=1)
    encoder_manifest = {
        "action_dim": 78,
        "motion_token_dim": 64,
        "frames": frames,
        "encoder": {
            "path": "/data/models/sonic-isaac/sonic_v1_1/model_encoder.onnx",
            "sha256": production["sonic"]["encoder_sha256"],
        },
        "observation_config_sha256": production["sonic"]["observation_config_sha256"],
        "encoder_orientation_policy": "reference_root_current_frame",
        "final_action_78d": {"status": "available", "reason": "hand channels resolved"},
        "token_norm": {"p50": float(np.percentile(norms, 50)), "max": float(norms.max())},
        "source": "production corpus slice (unitree-sonic-v1.1-78d); not a pilot re-encode",
    }
    (run_dir / "encoder_manifest.json").write_text(json.dumps(encoder_manifest, indent=2, sort_keys=True) + "\n")

    return {
        "run_dir": str(run_dir),
        "pilot": spec.pilot_name,
        "collection": collection,
        "episode_index": episode,
        "frames": frames,
        "duration_s": frames / FPS,
        "action_npz_sha256": run_manifest["production"]["action_npz_sha256"],
        "source_video": source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=Path("/data/datasets/unitree-sonic-v1.1-78d"))
    parser.add_argument("--raw-root", type=Path, default=Path("/data/datasets/first_tur_ham/unitree-g1-dex3"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/datasets/sonic/pilots.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-collection", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--seed", default="20260917")
    parser.add_argument("--episodes", action="append", default=[], metavar="COLLECTION:INDEX",
                        help="pin one episode instead of sampling (repeatable)")
    args = parser.parse_args()

    declared = [Path(name).name for name in json.loads(args.config.read_text())["unitree"]["bulk_datasets"]]
    pinned: dict[str, list[int]] = {}
    for item in args.episodes:
        collection, _, index = item.partition(":")
        if collection not in declared:
            raise SystemExit(f"pinned collection is not declared: {collection}")
        pinned.setdefault(collection, []).append(int(index))
    if pinned and (args.per_collection or len(pinned) != len(declared)):
        pass  # a pinned run only converts what it lists

    standing = deployment_standing_pose()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    collections = sorted(pinned) if pinned else declared
    for collection in collections:
        episodes = sorted(set(pinned[collection])) if collection in pinned else sampled_episodes(
            args.corpus_root, collection, per_collection=args.per_collection,
            max_seconds=args.max_seconds, seed=args.seed)
        for episode in episodes:
            run = prepare(args.corpus_root, args.raw_root, args.output_root, collection, episode,
                          standing, stamp)
            runs.append(run)
            print(f"[prepare] {run['pilot']} frames={run['frames']} ({run['duration_s']:.1f}s)", flush=True)

    batch = {
        "created_utc": stamp,
        "corpus_root": str(args.corpus_root),
        "per_collection": args.per_collection,
        "max_seconds": args.max_seconds,
        "seed": args.seed,
        "pinned": {key: sorted(value) for key, value in sorted(pinned.items())},
        "collections": len(collections),
        "runs": runs,
        "total_frames": sum(item["frames"] for item in runs),
    }
    (args.output_root / "batch_manifest.json").write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")
    print(f"[prepare] {len(runs)} runs, {batch['total_frames'] / FPS / 60:.1f} motion-minutes -> {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
