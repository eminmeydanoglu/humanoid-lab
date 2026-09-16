#!/usr/bin/env python3
"""Lean, resumable Unitree Dex3 -> SONIC v1.1 78D production conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.sonic.adapters.unitree_dex3 import (  # noqa: E402
    episode_row_index,
    load_episode,
    validate_metadata,
)
from humanoid_lab.datasets.sonic.encoder_observation import load_encoder_layout  # noqa: E402
from humanoid_lab.datasets.sonic.encoder_runner import G1EncoderRunner, compose_pilot_action  # noqa: E402
from humanoid_lab.datasets.sonic.production import PINNED_ENCODER_SHA256, build_encoder_input  # noqa: E402
from humanoid_lab.datasets.sonic.provenance import sha256_file  # noqa: E402
from humanoid_lab.datasets.sonic.reference import deployment_standing_pose  # noqa: E402
from humanoid_lab.datasets.sonic.training import DEFAULT_ACTION_HORIZON, SonicTrainingEpisode  # noqa: E402

OBS_CONFIG_SHA256 = "4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0"
CONVERTER_SCHEMA = 1


def git_identity() -> dict[str, object]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout)
    relevant = [
        Path(__file__),
        ROOT / "src/humanoid_lab/datasets/sonic/adapters/unitree_dex3.py",
        ROOT / "src/humanoid_lab/datasets/sonic/encoder_observation.py",
        ROOT / "src/humanoid_lab/datasets/sonic/encoder_runner.py",
        ROOT / "src/humanoid_lab/datasets/sonic/production.py",
        ROOT / "src/humanoid_lab/datasets/sonic/reference.py",
        ROOT / "src/humanoid_lab/datasets/sonic/timeline.py",
        ROOT / "src/humanoid_lab/datasets/sonic/training.py",
    ]
    bundle = hashlib.sha256()
    for path in relevant:
        bundle.update(str(path.relative_to(ROOT)).encode())
        bundle.update(path.read_bytes())
    return {"commit": commit, "dirty": dirty, "converter_sha256": file_sha256(Path(__file__)),
            "code_bundle_sha256": bundle.hexdigest()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compatible(manifest: dict, *, source: dict, encoder_sha: str, config_sha: str,
               converter_bundle_sha: str, output: Path) -> bool:
    return bool(
        manifest.get("schema_version") == CONVERTER_SCHEMA
        and manifest.get("source") == source
        and manifest.get("sonic", {}).get("encoder_sha256") == encoder_sha
        and manifest.get("sonic", {}).get("observation_config_sha256") == config_sha
        and manifest.get("converter", {}).get("code_bundle_sha256") == converter_bundle_sha
        and output.is_file()
        and manifest.get("output", {}).get("sha256") == file_sha256(output)
    )


def parse_range(text: str) -> tuple[int, int]:
    start, stop = (int(value) for value in text.split(":"))
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("range must be START:STOP")
    return start, stop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Unitree collection directory name")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episodes", type=parse_range)
    group.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=Path("/data/datasets/first_tur_ham/unitree-g1-dex3"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/datasets/unitree-sonic-v1.1-78d"))
    parser.add_argument("--model-dir", type=Path, default=Path("/data/models/sonic-isaac/sonic_v1_1"))
    parser.add_argument("--qc-manifest", type=Path, default=Path("/data/datasets/unitree-sonic-v1.1-78d/qc/unitree-production-qc.json"))
    parser.add_argument("--qc-mode", action="store_true", help="permit at most five explicitly selected episodes before QC approval")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    allowed = {Path(name).name for name in json.loads((ROOT / "configs/datasets/sonic/pilots.json").read_text())["unitree"]["bulk_datasets"]}
    if args.dataset not in allowed:
        raise SystemExit(f"dataset is not in the pinned Unitree allowlist: {args.dataset}")
    dataset = args.raw_root / args.dataset
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = validate_metadata(dataset)
    total_episodes = int(metadata["total_episodes"])
    if args.all_episodes:
        selection = range(total_episodes)
    else:
        start, stop = args.episodes
        if stop > total_episodes:
            raise SystemExit(f"episode range ends at {stop}, dataset has {total_episodes}")
        selection = range(start, stop)
    if args.qc_mode:
        if args.all_episodes or len(selection) > 5:
            raise SystemExit("--qc-mode requires an explicit selection of at most five episodes")
    else:
        if not args.qc_manifest.is_file() or json.loads(args.qc_manifest.read_text()).get("status") != "PASS":
            raise SystemExit(f"production conversion blocked until five-episode QC passes: {args.qc_manifest}")

    config_sha = sha256_file(args.model_dir / "observation_config.yaml")
    if config_sha != OBS_CONFIG_SHA256:
        raise SystemExit(f"wrong SONIC v1.1 observation config: {config_sha}")
    load_encoder_layout(args.model_dir / "observation_config.yaml")
    runner = G1EncoderRunner(args.model_dir / "model_encoder.onnx", expected_sha256=PINNED_ENCODER_SHA256)
    converter = git_identity()
    standing = deployment_standing_pose()
    rows = episode_row_index(dataset)
    data_sha_cache: dict[Path, str] = {}
    converted: list[dict] = []
    skipped: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    total_valid = total_frames = total_bytes = 0

    for ordinal, episode_index in enumerate(selection, 1):
        episode_started = time.perf_counter()
        row = rows[episode_index]
        data_file = dataset / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
        if data_file not in data_sha_cache:
            data_sha_cache[data_file] = file_sha256(data_file)
        source = {
            "collection": args.dataset,
            "episode_index": episode_index,
            "data_file": str(data_file.relative_to(dataset)),
            "data_file_sha256": data_sha_cache[data_file],
            "dataset_from_index": int(row["dataset_from_index"]),
            "dataset_to_index": int(row["dataset_to_index"]),
        }
        target = args.output_root / args.dataset / "episodes" / f"episode_{episode_index:06d}"
        artifact = target / "action.npz"
        manifest_path = target / "manifest.json"
        if not args.force and manifest_path.is_file():
            try:
                old = json.loads(manifest_path.read_text())
                if compatible(old, source=source, encoder_sha=runner.info.sha256, config_sha=config_sha,
                              converter_bundle_sha=str(converter["code_bundle_sha256"]), output=artifact):
                    skipped.append({"episode_index": episode_index, "valid_frames": old["frames"]["valid_training"]})
                    total_frames += int(old["frames"]["total"])
                    total_valid += int(old["frames"]["valid_training"])
                    total_bytes += artifact.stat().st_size + manifest_path.stat().st_size
                    print(f"[{ordinal}/{len(selection)}] skip {episode_index}: compatible", flush=True)
                    continue
            except Exception:
                pass
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".episode_{episode_index:06d}.", dir=target.parent))
        try:
            build = load_episode(dataset, episode_index, standing)
            encoder_input = build_encoder_input(build.episode)
            tokens = runner.encode_frames(encoder_input.observation)
            action = compose_pilot_action(tokens, build.episode.left_hand_joints, build.episode.right_hand_joints)
            if action is None or action.shape[1] != 78:
                raise ValueError("Unitree production action must be 78D")
            valid = encoder_input.future_clamp_fraction == 0.0
            temporary.mkdir(parents=True, exist_ok=True)
            np.savez(
                temporary / "action.npz",
                action=action,
                timestamp=build.episode.timestamps,
                frame_index=np.arange(len(action), dtype=np.int64),
                training_valid_mask=valid,
            )
            loaded = SonicTrainingEpisode.load(temporary / "action.npz")
            chunk_anchors, _ = loaded.valid_action_chunks(DEFAULT_ACTION_HORIZON)
            norms = np.linalg.norm(tokens, axis=1)
            output_sha = file_sha256(temporary / "action.npz")
            manifest = {
                "schema_version": CONVERTER_SCHEMA,
                "representation": "SONIC v1.1 64D body latent + Unitree Dex3 source action 14D",
                "source": source,
                "converter": converter,
                "sonic": {"encoder_sha256": runner.info.sha256, "observation_config_sha256": config_sha,
                          "model_revision": "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2"},
                "fps": {"source": 30.0, "target": 50.0},
                "frames": {"total": len(action), "valid_training": int(valid.sum()),
                           "excluded_tail": int((~valid).sum()), "valid_chunk_anchors_h40": len(chunk_anchors)},
                "shapes": {"action": list(action.shape), "motion_token": list(tokens.shape),
                           "hands": [len(action), 14], "training_valid_mask": list(valid.shape)},
                "action_slices": {"motion_token": [0, 64], "left_hand": [64, 71], "right_hand": [71, 78]},
                "token_sanity": {"finite": bool(np.isfinite(tokens).all()), "norm_p50": float(np.percentile(norms, 50)),
                                 "norm_p95": float(np.percentile(norms, 95)), "norm_max": float(norms.max())},
                "output": {"file": "action.npz", "sha256": output_sha},
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, target)
            elapsed = time.perf_counter() - episode_started
            size = artifact.stat().st_size + manifest_path.stat().st_size
            converted.append({"episode_index": episode_index, "seconds": elapsed, "frames": len(action),
                              "valid_frames": int(valid.sum()), "bytes": size})
            total_frames += len(action); total_valid += int(valid.sum()); total_bytes += size
            print(f"[{ordinal}/{len(selection)}] converted {episode_index}: {len(action)} frames, {valid.sum()} valid, {elapsed:.3f}s", flush=True)
        except Exception as error:
            shutil.rmtree(temporary, ignore_errors=True)
            failures.append({"episode_index": episode_index, "error": f"{type(error).__name__}: {error}"})
            print(f"[{ordinal}/{len(selection)}] FAILED {episode_index}: {error}", file=sys.stderr, flush=True)

    elapsed = time.perf_counter() - started
    summary = {"dataset": args.dataset, "requested": len(selection), "converted": len(converted), "skipped": len(skipped),
               "failed": len(failures), "total_frames": total_frames, "valid_training_frames": total_valid,
               "excluded_tail_frames": total_frames - total_valid, "output_bytes": total_bytes, "elapsed_s": elapsed,
               "throughput_frames_s": total_frames / elapsed if elapsed else None, "episodes": converted,
               "skipped_episodes": skipped, "failures": failures, "converter": converter,
               "encoder_sha256": runner.info.sha256, "observation_config_sha256": config_sha}
    summary_path = args.summary or args.output_root / args.dataset / "conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
