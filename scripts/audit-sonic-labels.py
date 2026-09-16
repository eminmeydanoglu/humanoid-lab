#!/usr/bin/env python3
"""Audit SONIC training-label timing, tail validity and latent statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.datasets.sonic.encoder_runner import G1EncoderRunner  # noqa: E402


def audit(pilot: Path, model_dir: Path) -> tuple[dict, np.ndarray]:
    with np.load(pilot / "action_tokens.npz") as token_file, np.load(pilot / "encoder_observation.npz") as encoder_file, np.load(pilot / "reference.npz") as reference:
        token = np.asarray(token_file["motion_token"], dtype=np.float32)
        token_time = np.asarray(token_file["timestamp"], dtype=np.float64)
        frame = np.asarray(token_file["frame_index"], dtype=np.int64)
        observation = np.asarray(encoder_file["observation"], dtype=np.float32)
        observation_time = np.asarray(encoder_file["timestamps"], dtype=np.float64)
        clamp = np.asarray(encoder_file["future_clamp_fraction"], dtype=np.float32)
        reference_time = np.asarray(reference["timestamps"], dtype=np.float64)
        has_hands = "left_hand_joints" in token_file.files
        stored_mask = np.asarray(token_file["training_valid_mask"], dtype=bool) if "training_valid_mask" in token_file.files else None
    runner = G1EncoderRunner(model_dir / "model_encoder.onnx")
    repeated = runner.encode_frames(observation)
    perturbation = observation.copy()
    perturbation[:, 4:294] += 1e-4
    perturbed = runner.encode_frames(perturbation)
    norms = np.linalg.norm(token, axis=1)
    steps = np.linalg.norm(np.diff(token, axis=0), axis=1)
    valid = clamp == 0.0
    result = {
        "pilot": str(pilot),
        "frames": len(token),
        "finite": bool(np.isfinite(token).all()),
        "frame_index_exact": bool(np.array_equal(frame, np.arange(len(token)))),
        "timestamps_exact": bool(np.array_equal(token_time, observation_time) and np.array_equal(token_time, reference_time)),
        "fps_50hz": bool(len(token_time) < 2 or np.allclose(np.diff(token_time), 0.02, atol=2e-6, rtol=0.0)),
        "repeat_encode_bitwise": bool(np.array_equal(token, repeated)),
        "perturbation_joint_position_eps": 1e-4,
        "perturbation_token_l2_p50": float(np.percentile(np.linalg.norm(perturbed - token, axis=1), 50)),
        "perturbation_token_l2_max": float(np.linalg.norm(perturbed - token, axis=1).max()),
        "token_norm": {"p50": float(np.percentile(norms, 50)), "p95": float(np.percentile(norms, 95)), "max": float(norms.max())},
        "adjacent_token_l2": {"p50": float(np.percentile(steps, 50)), "p95": float(np.percentile(steps, 95)), "max": float(steps.max())},
        "tail": {"valid_frames": int(valid.sum()), "excluded_frames": int((~valid).sum()),
                 "excluded_fraction": float(np.mean(~valid)), "first_excluded_frame": int(np.flatnonzero(~valid)[0])},
        "stored_training_mask_matches": None if stored_mask is None else bool(np.array_equal(stored_mask, valid)),
        "hand_label_present": has_hands,
    }
    return result, token[valid]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilots", type=Path, nargs="+")
    parser.add_argument("--model-dir", type=Path, default=Path("/data/models/sonic-isaac/sonic_v1_1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    episodes, valid_tokens = zip(*(audit(path.resolve(), args.model_dir) for path in args.pilots))
    domains = {}
    for item, values in zip(episodes, valid_tokens):
        domains[Path(item["pilot"]).name] = {"valid_frames": len(values), "mean": values.mean(axis=0).tolist(),
                                               "std_mean": float(values.std(axis=0).mean())}
    pairwise = []
    names = list(domains)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            distance = float(np.linalg.norm(np.asarray(domains[first]["mean"]) - np.asarray(domains[second]["mean"])))
            pairwise.append({"first": first, "second": second, "latent_mean_l2": distance})
    report = {"semantics": "row t is observation_t -> encode(reference future window starting at t)",
              "episodes": episodes, "domains": domains, "pairwise_domains": pairwise}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
