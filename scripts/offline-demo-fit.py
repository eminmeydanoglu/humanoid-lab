#!/usr/bin/env python3
"""Bounded open-loop predictions on recorded BlockStacking demonstrations.

Run inside the model's own container environment. The GR00T path loads the
checkpoint directly. The psi0 path connects to a locally served evaluated
checkpoint and uses Psi0's canonical no-augmentation dataset/request helpers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROMPT = "Stack the three cubic blocks on the black tape in the order red, yellow, blue."
PHASES = ("early", "middle", "late")


def phase_offsets(length: int, per_phase: int) -> list[tuple[str, int]]:
    centers = (0.15, 0.50, 0.85)
    result: list[tuple[str, int]] = []
    for phase, center in zip(PHASES, centers):
        offsets = np.linspace(-0.015, 0.015, per_phase)
        result.extend((phase, min(length - 1, max(0, int(round((center + x) * (length - 1)))))) for x in offsets)
    return result


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def group_metrics(rows: list[dict[str, Any]], groups: list[tuple[str, slice]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, span in groups:
        target = np.stack([np.asarray(r["target"])[span] for r in rows])
        result[name] = {}
        for method in ("model", "hold", "mean"):
            pred = np.stack([np.asarray(r[method])[span] for r in rows])
            result[name][method] = metrics(pred, target)
    return result


def summarize(rows: list[dict[str, Any]], groups: list[tuple[str, slice]]) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": len(rows), "groups": group_metrics(rows, groups)}
    out["by_phase"] = {}
    for phase in PHASES:
        subset = [r for r in rows if r["phase"] == phase]
        out["by_phase"][phase] = group_metrics(subset, groups) if subset else {}
    return out


def run_groot(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag.resolve("unitree_g1_sonic")
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(args.checkpoint), device="cuda")
    policy.model.action_head.num_inference_timesteps = 4
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=str(args.dataset), modality_configs=modality)
    action_keys = modality["action"].modality_keys
    state_only = dict(modality)
    state_only.pop("action")

    episodes_meta = [json.loads(x) for x in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()]
    candidates = [e["episode_index"] for e in episodes_meta if e["tasks"] == [PROMPT]]
    selected = [candidates[i] for i in np.linspace(0, len(candidates) - 1, args.episodes, dtype=int)]

    # Mean baseline is computed over the full BlockStacking population used to train GR00T.
    sums = None
    count = 0
    for episode_id in candidates:
        traj = loader[episode_id]
        arr = np.concatenate([np.vstack(traj[f"action.{key}"]) for key in action_keys], axis=1)
        sums = arr.sum(axis=0, dtype=np.float64) if sums is None else sums + arr.sum(axis=0, dtype=np.float64)
        count += len(arr)
    mean_action = sums / count

    rows = []
    for episode_id in selected:
        traj = loader[episode_id]
        for phase, frame in phase_offsets(len(traj), args.per_phase):
            point = extract_step_data(traj, frame, state_only, tag)
            obs: dict[str, Any] = {}
            obs.update({f"state.{k}": v for k, v in point.states.items()})
            obs.update({f"video.{k}": np.asarray(v) for k, v in point.images.items()})
            for key in modality["language"].modality_keys:
                obs[key] = point.text
            pred_dict, _ = policy.get_action(parse_observation_gr00t(obs, modality))
            pred = np.concatenate([np.asarray(pred_dict[k][0, 0]).reshape(-1) for k in action_keys])
            target = np.concatenate([np.asarray(traj.iloc[frame][f"action.{k}"]).reshape(-1) for k in action_keys])
            prev = max(frame - 1, 0)
            hold = np.concatenate([np.asarray(traj.iloc[prev][f"action.{k}"]).reshape(-1) for k in action_keys])
            rows.append({"episode": int(episode_id), "frame": int(frame), "phase": phase,
                         "target": target.tolist(), "model": pred.tolist(), "hold": hold.tolist(),
                         "mean": mean_action.tolist()})

    dims = [np.asarray(traj.iloc[0][f"action.{k}"]).size for k in action_keys]
    edges = np.cumsum([0] + dims)
    groups = [(key, slice(int(edges[i]), int(edges[i + 1]))) for i, key in enumerate(action_keys)]
    return {"model": "groot", "checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
            "split": "train (GR00T run configured no val_dataset_path)", "task_index": 0,
            "camera": "observation.images.ego_view, recorded head camera, RGB 640x480 at 50 Hz",
            "alignment": "prediction horizon row 0 vs action at the same anchor frame (delta index 0)",
            "normalization": "checkpoint processor_config/statistics; Gr00tPolicy decodes physical action units",
            "episodes": selected, "rows": rows, "summary": summarize(rows, groups)}


def load_psi_frames(args: argparse.Namespace):
    from transformers import AutoProcessor
    from psi.config.data_lerobot import LerobotDataConfig
    from psi.models.psi0 import QWEN3VL_VARIANT
    from psi.utils import apply_legacy_model_config_defaults, parse_args_to_tyro_config
    from psi.config.config import LaunchConfig

    cfg: LaunchConfig = parse_args_to_tyro_config(args.run_dir / "argv.txt")  # type: ignore
    cfg = cfg.model_validate(apply_legacy_model_config_defaults(json.loads((args.run_dir / "run_config.json").read_text())))
    data_cfg: LerobotDataConfig = cfg.data  # type: ignore
    processor = AutoProcessor.from_pretrained(QWEN3VL_VARIANT, local_files_only=True)
    dataset = data_cfg(split=args.split, transform_kwargs={"vlm_processor": processor, "no_aug": True})
    indices = dataset.raw_dataset.base_dataset.episode_data_index
    meta_path = args.dataset_root / args.split / "meta/episodes.jsonl"
    metadata = [json.loads(x) for x in meta_path.read_text().splitlines()]
    candidates = [e["episode_index"] for e in metadata if e["tasks"] == [PROMPT]]
    selected = [candidates[i] for i in np.linspace(0, len(candidates) - 1, args.episodes, dtype=int)]
    return cfg, dataset, indices, selected


async def psi_predict(args: argparse.Namespace, frames: list[tuple[dict, str, int, int]], cfg) -> list[dict[str, Any]]:
    import websockets
    from psi.deploy.mock_psi0_client_rtc import build_request, _parse_version

    field = cfg.data.transform.field
    image_keys = cfg.data.transform.repack.image_keys
    rows = []
    async with websockets.connect(f"ws://{args.host}:{args.port}/ws", max_size=16 * 1024 * 1024) as ws:
        version = 0
        for frame, phase, episode, offset in frames:
            payload, _ = build_request(frame, image_keys, field, frame.get("dataset_name"))
            await ws.send(payload)
            while True:
                text = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
                new_version, pred = _parse_version(text, version + 1)
                if new_version > version:
                    version = new_version
                    break
            target = np.asarray(frame["raw_actions"], np.float32)[0]
            rows.append({"episode": episode, "frame": offset, "phase": phase,
                         "target": target.tolist(), "model": pred.tolist()})
    return rows


def run_psi(args: argparse.Namespace) -> dict[str, Any]:
    from psi.deploy.mock_psi0_client_rtc import _slim_frame

    cfg, dataset, episode_index, selected = load_psi_frames(args)
    frames: list[tuple[dict, str, int, int]] = []
    episode_cache: dict[int, list[dict]] = {}
    # Compute the held-out task mean directly from parquet action columns. This avoids
    # running image/VLM preprocessing over every validation frame for a scalar baseline.
    import pyarrow.parquet as pq
    metadata = [json.loads(x) for x in (args.dataset_root / args.split / "meta/episodes.jsonl").read_text().splitlines()]
    sums = np.zeros(78, dtype=np.float64)
    count = 0
    for entry in metadata:
        if entry["tasks"] != [PROMPT]:
            continue
        episode = int(entry["episode_index"])
        path = args.dataset_root / args.split / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
        table = pq.read_table(path, columns=["action.body_token_v1_1", "action"])
        token = np.stack(table["action.body_token_v1_1"].to_pylist()).astype(np.float32)
        hand = np.stack(table["action"].to_pylist()).astype(np.float32)
        action = np.concatenate([token, hand], axis=1)
        sums += action.sum(axis=0, dtype=np.float64)
        count += len(action)
    mean_action = sums / count

    for episode in selected:
        start, end = int(episode_index["from"][episode]), int(episode_index["to"][episode])
        chosen = phase_offsets(end - start, args.per_phase)
        slim = {}
        for phase, offset in chosen:
            slim[offset] = _slim_frame(dataset[start + offset])
            frames.append((slim[offset], phase, int(episode), int(offset)))
        episode_cache[episode] = slim
    rows = asyncio.run(psi_predict(args, frames, cfg))
    for row in rows:
        start = int(episode_index["from"][row["episode"]])
        prev = max(row["frame"] - 1, 0)
        row["hold"] = np.asarray(dataset[start + prev]["raw_actions"], np.float32)[0].tolist()
        row["mean"] = mean_action.tolist()

    groups = [("body_token", slice(0, 64)), ("hands", slice(64, 78))]
    return {"model": "psi0", "checkpoint": str(args.run_dir / "checkpoints/ckpt_40000"),
            "dataset": str(args.dataset_root / args.split), "split": args.split,
            "task_index": 0, "camera": "observation.images.egocentric, recorded head camera, RGB 640x480 at 30 Hz",
            "alignment": "returned action row 0 vs raw_actions row 0 at the sent anchor frame",
            "normalization": "run_config bounds; server normalizes state and denormalizes 80D action; metrics use token[0:64] and hand[64:78]",
            "episodes": selected, "rows": rows, "summary": summarize(rows, groups)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("groot", "psi"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--per-phase", type=int, default=2)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8014)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()
    result = run_groot(args) if args.mode == "groot" else run_psi(args)
    result["generated_at_unix"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
