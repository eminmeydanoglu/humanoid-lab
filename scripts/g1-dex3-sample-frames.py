#!/usr/bin/env python3
"""Extract representative camera frames from the Unitree G1 Dex3 datasets.

Datasets live in a LeRobot v3 tree (meta/info.json + meta/episodes/*.parquet),
with long MP4 files that pack many episodes back to back. Episode metadata
gives each episode's (file_index, from_timestamp, to_timestamp) per camera, so
single frames can be grabbed with ffmpeg either straight from the Hugging Face
URLs (range reads, no full download) or from a local mirror.

Outputs JPEG frames plus a manifest JSON that the preview builder consumes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_TEMPLATE = "https://huggingface.co/datasets/{repo}/resolve/main/videos/{cam}/chunk-{chunk:03d}/file-{file:03d}.mp4"


def slugify(dataset_dir: Path) -> str:
    """G1_Dex3_PickApple_Dataset -> g1-dex3-pick-apple."""
    name = dataset_dir.name
    prefix = "G1_Dex3_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith("_Dataset"):
        name = name[: -len("_Dataset")]
    out = []
    for ch in name:
        if ch.isupper() and out and out[-1] != "-":
            out.append("-")
        out.append(ch.lower())
    return "g1-dex3-" + "".join(out).replace("_", "-")


def load_episodes(dataset_dir: Path):
    files = sorted(dataset_dir.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {dataset_dir}/meta/episodes")
    table = pq.read_table(files)
    return table.to_pandas()


def load_info(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "meta/info.json").read_text())


def pick_episodes(df, count: int) -> list[int]:
    """Spread picks across the episode index range: e.g. 10%, 50%, 90%."""
    ordered = sorted(int(e) for e in df["episode_index"])
    if count >= len(ordered):
        return ordered
    return [ordered[round(f * (len(ordered) - 1))] for f in (0.1, 0.5, 0.9)][: count or None]


def camera_span(row, cam: str) -> tuple[int, int, float, float]:
    chunk = int(row[f"videos/observation.images.{cam}/chunk_index"])
    file_index = int(row[f"videos/observation.images.{cam}/file_index"])
    return chunk, file_index, float(row[f"videos/observation.images.{cam}/from_timestamp"]), float(
        row[f"videos/observation.images.{cam}/to_timestamp"]
    )


def video_url(repo: str, cam: str, chunk: int, file_index: int) -> str:
    return REPO_TEMPLATE.format(repo=repo, cam=f"observation.images.{cam}", chunk=chunk, file=file_index)


def grab_frame(source: str, timestamp: float, out_path: Path, retries: int = 3) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(retries):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            source,
            "-frames:v",
            "1",
            "-q:v",
            "3",
            "-y",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
            return
        last_error = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit {proc.returncode}"
    raise RuntimeError(f"frame grab failed for {source} @ {timestamp:.2f}s: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filters", nargs="*", help="only datasets whose slug contains one of these")
    parser.add_argument("--datasets-root", default="data/datasets/first_tur_ham/unitree-g1-dex3")
    parser.add_argument("--out-root", default="data/outputs/dataset-previews/g1-dex3")
    parser.add_argument("--episodes-per-dataset", type=int, default=3)
    parser.add_argument("--frames-per-episode", type=int, default=5)
    parser.add_argument("--fractions", default="0.05,0.3,0.5,0.7,0.95")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--source", choices=["remote", "local"], default="remote")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fractions = [float(f) for f in args.fractions.split(",")]
    datasets_root = Path(args.datasets_root)
    out_root = Path(args.out_root)
    data_root = datasets_root.parent
    sources_path = datasets_root / "sources.json"
    sources = json.loads(sources_path.read_text()) if sources_path.exists() else {}
    dataset_dirs = sorted(p for p in datasets_root.iterdir() if (p / "meta/info.json").exists())
    if args.filters:
        dataset_dirs = [p for p in dataset_dirs if any(f in slugify(p) for f in args.filters)]
    if not dataset_dirs:
        print("no datasets matched", file=sys.stderr)
        return 2

    tasks = []
    for dataset_dir in dataset_dirs:
        info = load_info(dataset_dir)
        df = load_episodes(dataset_dir)
        repo = sources.get(dataset_dir.name, dataset_dir.name)
        slug = slugify(dataset_dir)
        cams = sorted(k.split("observation.images.")[1] for k in info["features"] if k.startswith("observation.images"))
        cams = [c for c in cams if f"videos/observation.images.{c}/from_timestamp" in df.columns]
        episodes = pick_episodes(df, args.episodes_per_dataset)
        for ep in episodes:
            row = df[df["episode_index"] == ep].iloc[0]
            for cam in cams:
                chunk, file_index, from_t, to_t = camera_span(row, cam)
                duration = to_t - from_t
                for k, frac in enumerate(fractions[: args.frames_per_episode]):
                    timestamp = from_t + frac * duration
                    out_path = out_root / "frames" / slug / f"ep{ep:06d}" / cam / f"{k:02d}_t{timestamp:08.2f}.jpg"
                    if args.source == "remote":
                        source = video_url(repo, cam, chunk, file_index)
                    else:
                        source = str(
                            dataset_dir / "videos" / f"observation.images.{cam}" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
                        )
                    tasks.append(
                        {
                            "dataset": dataset_dir.name,
                            "slug": slug,
                            "repo": repo,
                            "episode": int(ep),
                            "task": row["tasks"][0] if len(row["tasks"]) else "",
                            "length": int(row["length"]),
                            "fps": info["fps"],
                            "cam": cam,
                            "frame_slot": k,
                            "fraction": frac,
                            "timestamp": round(timestamp, 3),
                            "episode_from": round(from_t, 3),
                            "episode_to": round(to_t, 3),
                            "source": source,
                            "out": str(out_path),
                            "chunk": chunk,
                            "file_index": file_index,
                        }
                    )

    pending = [t for t in tasks if args.force or not Path(t["out"]).exists()]
    print(
        f"{len(tasks)} frames across {len(dataset_dirs)} datasets, {len(pending)} still to grab",
        flush=True,
    )
    failed_keys = set()

    def task_key(task):
        return (task["dataset"], task["episode"], task["cam"], task["frame_slot"])

    def run(task):
        if not args.force and Path(task["out"]).exists() and Path(task["out"]).stat().st_size > 1024:
            return None
        try:
            grab_frame(task["source"], task["timestamp"], Path(task["out"]))
            print(f"ok   {task['slug']} ep{task['episode']:06d} {task['cam']} t={task['timestamp']:.2f}", flush=True)
            return None
        except Exception as exc:  # noqa: BLE001 - report per-frame failure, keep going
            print(f"FAIL {task['slug']} ep{task['episode']:06d} {task['cam']} t={task['timestamp']:.2f}: {exc}", flush=True)
            return task_key(task)

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        for key in ex.map(run, tasks):
            if key:
                failed_keys.add(key)

    manifest = {
        "datasets_root": str(datasets_root),
        "data_root": str(data_root),
        "frames_root": str(out_root / "frames"),
        "episodes_per_dataset": args.episodes_per_dataset,
        "frames_per_episode": args.frames_per_episode,
        "fractions": fractions,
        "frames": [t for t in tasks if task_key(t) not in failed_keys and Path(t["out"]).exists()],
        "failed": [t for t in tasks if task_key(t) in failed_keys],
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {manifest_path}: {len(manifest['frames'])} frames, {len(manifest['failed'])} failures",
        flush=True,
    )
    return 1 if failed_keys else 0


if __name__ == "__main__":
    sys.exit(main())
