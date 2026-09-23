#!/usr/bin/env python3
"""Prepare and run a bounded captured-input GR00T conditioning audit."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = REPO / "data/outputs/blockstacking-debug/exp15-groot-contract-ab/corrected-sim-model-independent-rerun2-20260922T132233Z/groot-capture"
DEFAULT_ANCHORS = REPO / "data/outputs/blockstacking-debug/experiments/11-offline-demo-fit/groot-full-horizon-train.json"
DEFAULT_DATASET = REPO / "data/datasets/groot/unitree-dex3-sonic-v1/train"
DEFAULT_CHECKPOINT = REPO / "data/outputs/groot-unitree-dex3-sonic-v1/finetune/unitree-dex3-sonic-v1/full-20260918T232026Z/checkpoint-40000"
DEFAULT_OUT = REPO / "data/outputs/blockstacking-debug/exp16-conditioning-audit"
STATE_KEYS = ("left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand")
SEEDS = (292285, 292286, 292287)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_capture(path: Path) -> dict[str, Any]:
    from humanoid_lab.groot_inference_capture import load_capture as load
    return load(path)


def state_vector(observation: dict[str, Any]) -> np.ndarray:
    state = observation["state"]
    missing = [key for key in STATE_KEYS if key not in state]
    if missing:
        raise ValueError(f"missing state arrays: {missing}")
    return np.concatenate([np.asarray(state[key]).reshape(-1) for key in STATE_KEYS])


def replace_state_jointly(base: dict[str, Any], donor: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    if set(base["state"]) != set(donor["state"]):
        raise ValueError("state key sets differ; refusing partial state substitution")
    out["state"] = {key: np.array(value, copy=True) for key, value in donor["state"].items()}
    out["q"] = np.array(donor["q"], copy=True)
    return out


def variants(sim: dict[str, Any], demo: dict[str, Any]) -> dict[str, dict[str, Any]]:
    image = copy.deepcopy(sim)
    image["video"] = {key: np.array(value, copy=True) for key, value in demo["video"].items()}
    state = replace_state_jointly(sim, demo)
    full = copy.deepcopy(demo)
    return {"sim_full": copy.deepcopy(sim), "image_only_swap": image, "state_only_swap": state, "full_demo": full}


def capture_dirs(root: Path) -> list[Path]:
    candidates = []
    for path in root.glob("request-*/metadata.json"):
        meta = json.loads(path.read_text())
        if meta.get("checkpoint", {}).get("path", "").endswith("full-20260918T232026Z/checkpoint-40000"):
            candidates.append((int(meta["send_time_ns"]), path.parent))
    newest_by_sequence: dict[int, tuple[int, Path]] = {}
    for stamp, path in candidates:
        sequence = int(path.name.split("-")[1])
        if sequence < 32 and (sequence not in newest_by_sequence or stamp > newest_by_sequence[sequence][0]):
            newest_by_sequence[sequence] = (stamp, path)
    if set(newest_by_sequence) != set(range(32)):
        raise ValueError(f"expected request sequences 0..31, got {sorted(newest_by_sequence)}")
    return [newest_by_sequence[index][1] for index in range(32)]


def demo_observation(loader: Any, modality: dict[str, Any], tag: Any, episode: int, frame: int) -> tuple[dict[str, Any], np.ndarray]:
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    trajectory = loader[episode]
    observation_modality = dict(modality)
    observation_modality.pop("action")
    point = extract_step_data(trajectory, frame, observation_modality, tag)
    raw = {
        **{f"state.{key}": value for key, value in point.states.items()},
        **{f"video.{key}": np.asarray(value) for key, value in point.images.items()},
    }
    for key in modality["language"].modality_keys:
        raw[key] = point.text
    parsed = parse_observation_gr00t(raw, modality)
    parsed["q"] = state_vector(parsed).reshape(1, 1, 43).astype(np.float32)
    action_keys = modality["action"].modality_keys
    target = np.concatenate([np.vstack(trajectory[f"action.{key}"]) for key in action_keys], axis=1)
    return parsed, target[frame:frame + 40]


def prepare(args: argparse.Namespace) -> None:
    import gr00t.model  # noqa: F401
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from transformers import AutoProcessor

    args.output.mkdir(parents=True, exist_ok=True)
    tag = EmbodimentTag.resolve("unitree_g1_sonic")
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    modality = processor.get_modality_configs()[tag.value]
    del processor
    loader = LeRobotEpisodeLoader(dataset_path=str(args.dataset), modality_configs=modality)
    anchor_rows = json.loads(args.anchors.read_text())["rows"]
    anchors = {(row["episode"], row["frame"], row["phase"]) for row in anchor_rows}
    demos = []
    for episode, frame, phase in sorted(anchors):
        observation, target = demo_observation(loader, modality, tag, episode, frame)
        demos.append({"episode": episode, "frame": frame, "phase": phase, "observation": observation, "target": target})

    rows = []
    payload: dict[str, np.ndarray] = {}
    for sequence, directory in enumerate(capture_dirs(args.capture)):
        captured = load_capture(directory)
        sim = captured["observation"]
        sim_vector = state_vector(sim)
        distances = [float(np.linalg.norm(sim_vector - state_vector(demo["observation"]))) for demo in demos]
        match_index = int(np.argmin(distances))
        demo = demos[match_index]
        named = variants(sim, demo["observation"])
        prefix = f"request_{sequence:02d}"
        for variant, observation in named.items():
            for key, value in observation["state"].items():
                payload[f"{prefix}.{variant}.state.{key}"] = np.asarray(value)
            payload[f"{prefix}.{variant}.q"] = np.asarray(observation["q"])
            payload[f"{prefix}.{variant}.video.ego_view"] = np.asarray(observation["video"]["ego_view"])
        payload[f"{prefix}.demo_target"] = np.asarray(demo["target"])
        rows.append({
            "sequence": sequence,
            "capture": directory.name,
            "request_id": captured["metadata"]["request_id"],
            "demo_episode": demo["episode"],
            "demo_frame": demo["frame"],
            "phase": demo["phase"],
            "arm_hand_state_l2": distances[match_index],
            "variants": list(named),
        })
    np.savez(args.output / "conditioning-inputs.npz", **payload)
    manifest = {
        "schema_version": 1,
        "mode": "cpu_prepare_only_no_model_load",
        "capture_root": str(args.capture),
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
        "seeds": list(SEEDS),
        "requests": rows,
        "substitutions": {
            "image_only_swap": "demo ego_view only; every captured state array, q, projected gravity, language, and timestamp retained",
            "state_only_swap": "all demo named state arrays including projected_gravity plus q replaced jointly; captured image retained",
            "full_demo": "matched demo image, named state arrays, q, language, and target",
        },
        "interpretation_limits": [
            "hybrid inputs are interventions, not claimed natural observations",
            "matching is limited to the existing exp11 phase-labelled anchors and arm/hand state L2",
            "no object identity or correspondence is assumed",
            "motion-token distances are not decoded arm-target metrics; recurrent decoder history is unavailable in the capture",
            "distance to the matched demo target is descriptive and strongest for full_demo, not proof that hybrid inputs are natural",
        ],
        "npz": "conditioning-inputs.npz",
    }
    (args.output / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "requests": len(rows), "arrays": len(payload)}, indent=2))


def run(args: argparse.Namespace) -> None:
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    plan_path = args.plan
    prepared = json.loads(plan_path.read_text())
    stored = np.load(args.output / prepared["npz"], allow_pickle=False)
    tag = EmbodimentTag.resolve("unitree_g1_sonic")
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(args.checkpoint), device="cuda")
    policy.model.action_head.num_inference_timesteps = 4
    action_keys = policy.get_modality_config()["action"].modality_keys
    results = []
    for request in prepared["requests"]:
        prefix = f"request_{request['sequence']:02d}"
        target = stored[f"{prefix}.demo_target"]
        for variant in request["variants"]:
            observation = {"state": {}, "video": {"ego_view": stored[f"{prefix}.{variant}.video.ego_view"]}}
            for key in set(name.split(".state.", 1)[1] for name in stored.files if name.startswith(f"{prefix}.{variant}.state.")):
                observation["state"][key] = stored[f"{prefix}.{variant}.state.{key}"]
            observation["q"] = stored[f"{prefix}.{variant}.q"]
            observation["language"] = {"annotation.human.task_description": [["Stack the three cubic blocks on the black tape in the order red, yellow, blue."]]}
            for seed in prepared["seeds"]:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                action, _ = policy.get_action(observation)
                chunk = np.concatenate([action[key][0] for key in action_keys], axis=1)[:len(target)]
                row = {
                    **{key: request[key] for key in ("sequence", "request_id", "demo_episode", "demo_frame", "phase")},
                    "variant": variant,
                    "seed": seed,
                    "chunk": chunk.tolist(),
                }
                if variant == "full_demo":
                    delta = chunk - target
                    row["limited_full_demo_target_mae_78d"] = float(np.abs(delta).mean())
                    row["limited_full_demo_target_mae_hands"] = float(np.abs(delta[:, 64:]).mean())
                results.append(row)

    def distance(one: np.ndarray, two: np.ndarray) -> dict[str, float]:
        delta = one - two
        return {
            "mae_78d": float(np.abs(delta).mean()),
            "mae_motion_token_raw": float(np.abs(delta[:, :64]).mean()),
            "mae_hands": float(np.abs(delta[:, 64:]).mean()),
        }

    indexed = {(row["sequence"], row["seed"], row["variant"]): np.asarray(row["chunk"]) for row in results}
    paired = []
    for request in prepared["requests"]:
        sequence = request["sequence"]
        for seed in prepared["seeds"]:
            sim = indexed[sequence, seed, "sim_full"]
            image = indexed[sequence, seed, "image_only_swap"]
            state = indexed[sequence, seed, "state_only_swap"]
            demo = indexed[sequence, seed, "full_demo"]
            paired.append({
                "sequence": sequence,
                "seed": seed,
                "image_effect_at_captured_state": distance(image, sim),
                "state_effect_at_captured_image": distance(state, sim),
                "image_effect_at_same_demo_state": distance(demo, state),
            })
    stochastic = []
    for request in prepared["requests"]:
        for variant in request["variants"]:
            chunks = [indexed[request["sequence"], seed, variant] for seed in prepared["seeds"]]
            pair_distances = [distance(chunks[i], chunks[j]) for i in range(len(chunks)) for j in range(i + 1, len(chunks))]
            stochastic.append({"sequence": request["sequence"], "variant": variant, "seed_pair_distances": pair_distances})
    first = prepared["requests"][0]
    repeat_seed = prepared["seeds"][0]
    repeat_prefix = f"request_{first['sequence']:02d}"
    repeat_observation = {"state": {}, "video": {"ego_view": stored[f"{repeat_prefix}.sim_full.video.ego_view"]}}
    for key in set(name.split(".state.", 1)[1] for name in stored.files if name.startswith(f"{repeat_prefix}.sim_full.state.")):
        repeat_observation["state"][key] = stored[f"{repeat_prefix}.sim_full.state.{key}"]
    repeat_observation["language"] = {"annotation.human.task_description": [["Stack the three cubic blocks on the black tape in the order red, yellow, blue."]]}
    torch.manual_seed(repeat_seed)
    torch.cuda.manual_seed_all(repeat_seed)
    repeat_action, _ = policy.get_action(repeat_observation)
    repeat_chunk = np.concatenate([repeat_action[key][0] for key in action_keys], axis=1)
    original_chunk = indexed[first["sequence"], repeat_seed, "sim_full"]
    determinism = {
        "sequence": first["sequence"],
        "seed": repeat_seed,
        "array_equal": bool(np.array_equal(repeat_chunk, original_chunk)),
        "max_abs_delta": float(np.max(np.abs(repeat_chunk - original_chunk))),
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "plan": str(plan_path),
        "seeds": prepared["seeds"],
        "determinism_repeat": determinism,
        "rows": results,
        "paired_sensitivity": paired,
        "paired_stochastic_variance": stochastic,
        "interpretation": prepared["primary_analysis"],
        "non_primary": prepared["non_primary"],
    }
    (args.output / "gpu-results.json").write_text(json.dumps(result, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_OUT / "sampled-preparation.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare(arguments) if arguments.mode == "prepare" else run(arguments)
