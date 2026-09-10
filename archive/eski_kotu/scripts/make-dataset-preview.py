#!/usr/bin/env python3
"""Build a self-contained HTML preview of a LeRobot-v2-style robotics dataset.

Extracts a few frames from episode videos together with the corresponding
state/action rows from the parquet files, plus the task text, and embeds
everything into a single HTML file (base64 images, inline SVG sparklines).

Usage:
  make-dataset-preview.py --root <dataset_root> --out <out.html> \
      [--episodes 0 1 2] [--frames 5]
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import os
import sys

import cv2  # pip: opencv-python-headless
import numpy as np
import pyarrow.parquet as pq


def load_tasks(root: str) -> list[str]:
    path = os.path.join(root, "meta", "tasks.jsonl")
    if not os.path.exists(path):
        return []
    tasks: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tasks[int(rec["task_index"])] = rec["task"]
    return [tasks[i] for i in sorted(tasks)]


def frame_to_jpeg(frame: np.ndarray, quality: int = 70) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def sparkline(values: np.ndarray, width: int = 220, height: int = 44) -> str:
    """Render a 1-D series as an inline SVG polyline."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return ""
    if vals.size == 1:
        vals = np.concatenate([vals, vals])
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = (vmax - vmin) or 1.0
    pad = span * 0.05
    lo, hi = vmin - pad, vmax + pad
    xs = np.linspace(0, width, vals.size)
    ys = height - (vals - lo) / (hi - lo) * height
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="background:#f6f6f6;border:1px solid #ddd">'
        f'<polyline points="{pts}" fill="none" stroke="#1a73e8" stroke-width="1.4"/>'
        f"</svg>"
    )


def column_from(table, name: str):
    """Return a column as a list-of-arrays, tolerating nested/list storage."""
    if name not in table.column_names:
        return None
    col = table.column(name).combine_chunks()
    return col.to_pylist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--max-state-dims", type=int, default=12)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    parquet_files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "*.parquet")))
    video_files = sorted(
        glob.glob(os.path.join(root, "videos", "chunk-*", "observation.images.*", "*.mp4"))
    )
    # Also match paths where videos sit directly under videos/chunk-*/…
    if not video_files:
        video_files = sorted(glob.glob(os.path.join(root, "videos", "chunk-*", "*.mp4")))
    if not parquet_files or not video_files:
        print(f"no data found under {root}", file=sys.stderr)
        return 2

    tasks = load_tasks(root)
    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    fps = info.get("fps", 50)
    features = info.get("features", {})
    state_names = []
    for key in features:
        if key == "observation.state":
            state_names = features[key].get("names", [])
    action_key = "action" if any("action" in k for k in features) else None

    max_ep = len(parquet_files)
    ep_indices = [e for e in args.episodes if 0 <= e < max_ep]
    if not ep_indices:
        ep_indices = [0]

    cards = []
    for ep in ep_indices:
        pq_path = parquet_files[ep]
        # video path mirrors parquet chunk dir naming
        ep_label = f"episode_{ep:06d}"
        vid_path = next((v for v in video_files if ep_label in v), None)
        if vid_path is None:
            print(f"no video for {ep_label}", file=sys.stderr)
            continue
        table = pq.read_table(pq_path)
        n = table.num_rows
        state_col = column_from(table, "observation.state")
        action_col = None
        for cand in ("action", "action.wbc"):
            if cand in table.column_names:
                action_col = column_from(table, cand)
                action_key = cand
                break
        task_idx = table.column("task_index").to_pylist()[0] if "task_index" in table.column_names else 0
        task = tasks[task_idx] if task_idx < len(tasks) else f"task {task_idx}"

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"cannot open {vid_path}", file=sys.stderr)
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # pick spread indices across the episode
        idxs = np.linspace(0, max(n - 1, 1), args.frames).astype(int)
        idxs = np.unique(idxs)

        imgs_html = []
        for fi in idxs:
            cap = cv2.VideoCapture(vid_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            t_sec = fi / fps if fps else fi / 50
            state_txt = ""
            if state_col is not None:
                row = state_col[fi]
                dims = row if isinstance(row, (list, np.ndarray)) else []
                if state_names and dims:
                    shown = state_names[: args.max_state_dims]
                    parts = [f"{name}={float(dims[i]):.3f}" for i, name in enumerate(shown)]
                    state_txt = ", ".join(parts)
            b64 = frame_to_jpeg(frame)
            imgs_html.append(
                f'<figure style="margin:6px">'
                f'<img src="data:image/jpeg;base64,{b64}" style="width:320px;height:240px;object-fit:cover;'
                f'border:1px solid #ccc;border-radius:4px"/>'
                f"<figcaption style='font-size:11px;color:#555'>t={t_sec:.1f}s (frame {fi})<br/>{html.escape(state_txt)}</figcaption>"
                f"</figure>"
            )

        # state/action summary sparklines: downsample columns
        def down(col):
            if col is None:
                return None
            arr = np.asarray(col, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr

        svgs = []
        for label, col in (("state", state_col), ("action", action_col)):
            arr = down(col)
            if arr is None or arr.size == 0:
                continue
            n_dims = min(arr.shape[1], 6)
            for d in range(n_dims):
                series = arr[:, d]
                svgs.append(f"<div>{label}[{d}]</div>" + sparkline(series))

        # task + frames-per-second badge
        cards.append(
            f'<section style="border:1px solid #ddd;border-radius:8px;padding:12px;margin:16px 0;background:#fff">'
            f'<h3 style="margin:0 0 4px">Episode {ep:06d} · {n} frames @ {fps} fps · {n / fps:.1f}s</h3>'
            f'<p style="margin:2px 0 10px;font-size:13px"><b>Task:</b> {html.escape(task)}</p>'
            f'<div style="display:flex;flex-wrap:wrap;gap:4px">{ "".join(imgs_html) }</div>'
            f'<details style="margin-top:10px"><summary style="cursor:pointer;font-size:13px">state/action signal preview</summary>'
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px;margin-top:8px">{ "".join(svgs) }</div>'
            f"</details></section>"
        )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset preview · {os.path.basename(root)}</title>
<style>body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:24px;background:#fafafa;color:#222}}
h1{{font-size:20px}} h2{{font-size:16px;color:#444}}</style></head>
<body>
<h1>Dataset: {html.escape(os.path.basename(root))}</h1>
<p style="font-size:13px;color:#555">{html.escape(root)} · {len(parquet_files)} episodes · task(s): {html.escape('; '.join(tasks) or '-')}</p>
{ "".join(cards) }
<p style="font-size:11px;color:#999;margin-top:24px">generated by make-dataset-preview.py</p>
</body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
