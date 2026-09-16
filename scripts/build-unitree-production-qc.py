#!/usr/bin/env python3
"""Build the five-episode Unitree production QC manifest and HTML dashboard."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENCODER_SHA = "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"
CONFIG_SHA = "4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilots", nargs=5, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    episodes = []
    all_norms = []
    total_frames = valid_frames = 0
    for pilot in args.pilots:
        pilot = pilot.resolve()
        fidelity = json.loads((pilot / "sonic_fidelity.json").read_text())
        encoder = json.loads((pilot / "encoder_manifest.json").read_text())
        run = json.loads((pilot / "run_manifest.json").read_text())
        with np.load(pilot / "action_tokens.npz") as payload:
            if "training_valid_mask" not in payload.files:
                raise SystemExit(f"legacy artifact has no training_valid_mask: {pilot}")
            token = np.asarray(payload["motion_token"], dtype=np.float32)
            mask = np.asarray(payload["training_valid_mask"], dtype=bool)
        if encoder["encoder"]["sha256"] != ENCODER_SHA or encoder["observation_config_sha256"] != CONFIG_SHA:
            raise SystemExit(f"wrong SONIC family: {pilot}")
        expected = np.ones(len(token), dtype=bool); expected[-45:] = False
        if not np.array_equal(mask, expected):
            raise SystemExit(f"invalid future-window mask: {pilot}")
        norms = np.linalg.norm(token[mask], axis=1)
        continuity = np.linalg.norm(np.diff(token[mask], axis=0), axis=1)
        all_norms.append(norms)
        total_frames += len(token); valid_frames += int(mask.sum())
        fig, axes = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
        axes[0].plot(norms, linewidth=0.8); axes[0].set_ylabel("latent L2"); axes[0].grid(alpha=.2)
        axes[1].plot(mask.astype(int), linewidth=1); axes[1].set_ylabel("valid"); axes[1].set_ylim(-.1, 1.1)
        axes[1].set_xlabel("frame"); axes[1].grid(alpha=.2)
        fig.tight_layout(); fig.savefig(pilot / "sonic_token_mask.png", dpi=130); plt.close(fig)
        pair = fidelity["pairs"]["reference_to_response"]
        episodes.append({
            "dataset": run["dataset_name"], "episode_index": run["episode_index"], "pilot_dir": str(pilot),
            "result": fidelity["result"], "frames": len(token), "valid_frames": int(mask.sum()),
            "body_mae_rad": pair["body_mae_rad"], "body_p50_rad": pair["body_p50_rad"],
            "body_p95_rad": pair["body_p95_rad"], "body_max_rad": pair["body_max_rad"],
            "wrist_path_mae_m": fidelity["left_wrist"]["reference_to_response"]["path_mae_m"],
            "wrist_orientation": fidelity["left_wrist_orientation"]["reference_to_response"],
            "lag": fidelity["temporal_alignment"], "catastrophic": fidelity["catastrophic_windows"],
            "token_norm": {"p50": float(np.percentile(norms, 50)), "p95": float(np.percentile(norms, 95)), "max": float(norms.max())},
            "continuity": {"p50": float(np.percentile(continuity, 50)), "p95": float(np.percentile(continuity, 95)), "max": float(continuity.max())},
            "failures": [item["metric"] for item in fidelity["decisions"] if item["result"] == "FAIL"],
        })
    combined = np.concatenate(all_norms)
    status = "PASS" if all(item["result"] == "PASS" for item in episodes) else "FAIL"
    report = {
        "status": status, "purpose": "gate full Unitree SONIC v1.1 78D production conversion",
        "episodes_processed": len(episodes), "episodes_passed": sum(item["result"] == "PASS" for item in episodes),
        "episodes_failed": sum(item["result"] != "PASS" for item in episodes),
        "total_frames": total_frames, "valid_training_frames": valid_frames,
        "excluded_tail_frames": total_frames - valid_frames,
        "tail_mask_fraction": (total_frames - valid_frames) / total_frames,
        "token_norm": {"p50": float(np.percentile(combined, 50)), "p95": float(np.percentile(combined, 95)), "max": float(combined.max())},
        "converter": {"commit": commit, "schema_version": 1},
        "sonic": {"encoder_sha256": ENCODER_SHA, "observation_config_sha256": CONFIG_SHA},
        "episodes": episodes,
    }
    manifest = output / "unitree-production-qc.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    cards = []
    for item in episodes:
        pilot = Path(item["pilot_dir"])
        def link(name: str, label: str) -> str:
            path = pilot / name
            relative = os.path.relpath(path, output)
            return f'<a href="{html.escape(relative)}">{label}</a>' if path.exists() else f"{label} unavailable"
        reason = ", ".join(item["failures"]) or "all gates passed"
        cards.append(f'''<section><h2>{html.escape(item["dataset"])} · ep{item["episode_index"]:03d} <b class="{item["result"].lower()}">{item["result"]}</b></h2>
<p>{reason}; frames {item["frames"]}, valid {item["valid_frames"]}; body MAE/P95/max {item["body_mae_rad"]:.3f}/{item["body_p95_rad"]:.3f}/{item["body_max_rad"]:.3f} rad; wrist {item["wrist_path_mae_m"]*100:.2f} cm; orientation P95 {item["wrist_orientation"]["p95_rad"]:.3f} rad; lag {item["lag"]["best_lag_s"]*1000:.0f} ms, corr {item["lag"]["mean_correlation"]:.3f}.</p>
<p>Token norm P50/P95/max {item["token_norm"]["p50"]:.3f}/{item["token_norm"]["p95"]:.3f}/{item["token_norm"]["max"]:.3f}; continuity P95/max {item["continuity"]["p95"]:.3f}/{item["continuity"]["max"]:.3f}; catastrophic windows {len(item["catastrophic"]["windows"])}.</p>
<div class="links">{link("source.mp4", "source video")} · {link("completed_reference_kinematic.mp4", "A reference")} · {link("sonic_latent_motion.mp4", "C rollout")} · {link("sonic_fidelity.json", "metrics")}</div>
<div class="plots">{link("sonic_left_arm_abc.png", "arm plot")} {link("sonic_left_wrist_abc.png", "wrist plot")} {link("sonic_token_mask.png", "token/mask plot")}</div></section>''')
    page = f'''<!doctype html><meta charset="utf-8"><title>Unitree SONIC production QC</title><style>
body{{font:15px system-ui;max-width:1200px;margin:32px auto;padding:0 18px;background:#0d1117;color:#d8dee9}}section{{border:1px solid #30363d;border-radius:10px;padding:18px;margin:18px 0;background:#161b22}}a{{color:#79c0ff}}.pass{{color:#3fb950}}.fail{{color:#f85149}}code{{font-size:12px}}.plots a{{display:inline-block;margin:12px 18px 0 0}}</style>
<h1>Unitree/G1 → SONIC v1.1 → 78D VLA production QC</h1><section><h2>Dataset-level summary <b class="{status.lower()}">{status}</b></h2>
<p>Processed {len(episodes)}; passed {report["episodes_passed"]}; failed {report["episodes_failed"]}; total {total_frames} frames; valid {valid_frames}; excluded tail {total_frames-valid_frames} ({report["tail_mask_fraction"]:.2%}).</p>
<p>Token norm P50/P95/max {report["token_norm"]["p50"]:.3f}/{report["token_norm"]["p95"]:.3f}/{report["token_norm"]["max"]:.3f}.</p>
<p>Converter <code>{commit}</code>; encoder <code>{ENCODER_SHA}</code>; config <code>{CONFIG_SHA}</code>.</p></section>{''.join(cards)}'''
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"status": status, "manifest": str(manifest), "html": str(output / "index.html")}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
