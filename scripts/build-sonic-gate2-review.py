#!/usr/bin/env python3
"""Build plots, comparison videos, and one local Human Gate 2 review page."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def prepare(run: Path, *, repair_support_flag: bool) -> dict[str, object]:
    metrics = json.loads((run / "sonic_metrics.json").read_text())
    rows = pq.read_table(run / "sonic_tracking.parquet").to_pylist()
    release_tick = metrics["controller"]["support"]["release_tick"]
    if repair_support_flag:
        for row in rows:
            row["support_active"] = int(row["tick"]) < int(release_tick)
        pq.write_table(pa.Table.from_pylist(rows), run / "sonic_tracking.parquet")
    free_all = [row for row in rows if not row["support_active"]]
    if metrics["controller"]["support"]["release_reason"] != "controller_motion_after_hold":
        raise RuntimeError(f"{run}: invalid support release")
    if not free_all or metrics["controller"]["gap"] is not None or metrics["controller"]["rejected_commands"]:
        raise RuntimeError(f"{run}: incomplete free-control evidence")
    duration_s = float(json.loads((run / "run_manifest.json").read_text())["duration_s"])
    release_s = release_tick * 0.005
    free = [row for row in free_all if float(row["sim_s"]) <= release_s + duration_s + 0.02]
    if not free:
        raise RuntimeError(f"{run}: no SONIC tracking rows overlap the demonstration")

    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    time = np.asarray([row["sim_s"] for row in rows])
    body_target = np.asarray([row["body_target"] for row in rows])
    body_measured = np.asarray([row["body_measured"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time, np.mean(np.abs(body_target[:, :15] - body_measured[:, :15]), axis=1))
    axes[0].set_ylabel("Leg/waist MAE (rad)")
    axes[1].plot(time, np.mean(np.abs(body_target[:, 15:] - body_measured[:, 15:]), axis=1))
    axes[1].set_ylabel("Arm MAE (rad)"); axes[1].set_xlabel("Simulation time (s)")
    for axis in axes: axis.axvline(release_tick * 0.005, color="tab:red", linestyle="--", label="support release"); axis.legend()
    fig.tight_layout(); fig.savefig(plots / "body_tracking.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis, side in zip(axes, ("left", "right")):
        target = np.asarray([row[f"{side}_hand_target"] for row in rows])
        measured = np.asarray([row[f"{side}_hand_measured"] for row in rows])
        axis.plot(time, np.mean(np.abs(target - measured), axis=1))
        axis.set_ylabel(f"{side.title()} hand MAE (rad)")
    axes[-1].set_xlabel("Simulation time (s)"); fig.tight_layout()
    fig.savefig(plots / "hand_tracking.png", dpi=150); plt.close(fig)

    def tilt_deg(quaternion: list[float]) -> float:
        _, x, y, _ = quaternion
        up_z = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
        return float(np.degrees(np.arccos(up_z)))

    free_tilt = np.asarray([tilt_deg(row["root_quaternion_wxyz"]) for row in free])
    free_torso_tilt = np.asarray([tilt_deg(row["torso_quaternion_wxyz"]) for row in free])

    # The source clip starts at demonstration t=0. The simulator video also
    # contains model startup and supported IDLE, so trim it at support release.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(run / "source.mp4"),
                    "-ss", "1.000", "-i", str(run / "direct_fixed.mp4"),
                    "-ss", f"{release_s:.3f}", "-i", str(run / "sonic_free.mp4"),
                    "-filter_complex",
                    "[0:v]scale=480:-2,pad=480:480:(ow-iw)/2:(oh-ih)/2[a];"
                    "[1:v]scale=480:-2,pad=480:480:(ow-iw)/2:(oh-ih)/2[b];"
                    "[2:v]scale=480:-2,pad=480:480:(ow-iw)/2:(oh-ih)/2[c];"
                    "[a][b][c]hstack=inputs=3:shortest=1[v]",
                    "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-shortest", str(run / "triple_comparison.mp4")], check=True)
    return {"run": run, "free_frames": len(free), "min_free_z": min(row["root_position"][2] for row in free),
            "final_free_z": free[-1]["root_position"][2], "mean_error": metrics["controller"]["body_tracking_error_rad"]["mean"],
            "root_tilt_p95": float(np.percentile(free_tilt, 95)), "root_tilt_max": float(free_tilt.max()),
            "torso_tilt_p95": float(np.percentile(free_torso_tilt, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("output", type=Path); parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--repair-support-flag", type=int, action="append", default=[])
    args = parser.parse_args()
    results = [prepare(run.resolve(), repair_support_flag=index in args.repair_support_flag) for index, run in enumerate(args.runs)]
    cards = []
    for result in results:
        run = result["run"]; rel = Path(os.path.relpath(run, args.output.parent.resolve()))
        cards.append(f'''<section><h2>{html.escape(run.parent.parent.name)} / {html.escape(run.parent.name)}</h2>
<p>Serbest frame: {result['free_frames']} · min/final root z: {result['min_free_z']:.3f}/{result['final_free_z']:.3f} m · body MAE: {result['mean_error']:.4f} rad<br>
Root tilt p95/max: {result['root_tilt_p95']:.1f}°/{result['root_tilt_max']:.1f}° · torso tilt p95: {result['torso_tilt_p95']:.1f}°</p>
<div class="videos"><div><h3>1. Orijinal</h3><video controls preload="metadata" src="{rel / 'source.mp4'}"></video></div>
<div><h3>2. Askıda doğrudan joint replay</h3><video controls preload="metadata" src="{rel / 'direct_fixed.mp4'}"></video></div>
<div><h3>3. Serbest SONIC</h3><video controls preload="metadata" src="{rel / 'sonic_free.mp4'}"></video></div></div>
<p><a href="{rel / 'triple_comparison.mp4'}">Senkron üçlü video</a> · <a href="{rel / 'reference.parquet'}">Reference</a> · <a href="{rel / 'direct_tracking.parquet'}">Doğrudan tracking</a> · <a href="{rel / 'sonic_tracking.parquet'}">SONIC tracking</a> · <a href="{rel / 'sonic_metrics.json'}">SONIC metrics</a></p>
<div class="plots"><img src="{rel / 'plots/body_tracking.png'}"><img src="{rel / 'plots/hand_tracking.png'}"></div></section>''')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('''<!doctype html><html lang="tr"><meta charset="utf-8"><title>Human Gate 2</title><style>
body{font:16px system-ui;background:#111;color:#eee;max-width:1600px;margin:auto;padding:24px}section{background:#1b1b1b;padding:20px;margin:24px 0;border-radius:14px}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:16px}video,img{width:100%}a{color:#7dcfff}</style><h1>Human Gate 2 — Joint-order düzeltmesi sonrası üçlü doğrulama</h1><p>Her kart: kaynak kamera → SONIC bypass edilmiş sabit tabanlı joint oracle → serbest SONIC. Gate hâlâ insan onayı bekler.</p>''' + "".join(cards) + "</html>\n", encoding="utf-8")


if __name__ == "__main__": main()
