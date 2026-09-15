#!/usr/bin/env python3
"""Export representative G1 Dex3 episodes as standalone video clips.

Reads the episode picks recorded by g1-dex3-sample-frames.py (manifest.json),
cuts each picked episode's camera segment out of the packed LeRobot v3 MP4 and
re-encodes it to H.264 so the clip plays in any player. Clips are cut from the
local mirror when present and read straight from the Hugging Face URL otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

URL_TEMPLATE = (
    "https://huggingface.co/datasets/{repo}/resolve/main/"
    "videos/observation.images.{cam}/chunk-{chunk:03d}/file-{file:03d}.mp4"
)


def local_path(datasets_root: Path, dataset: str, cam: str, chunk: int, file_index: int) -> Path:
    return (
        datasets_root
        / dataset
        / "videos"
        / f"observation.images.{cam}"
        / f"chunk-{chunk:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def run_ffmpeg(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], capture_output=True, text=True, timeout=timeout)


def export_clip(source: str, start: float, duration: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "-ss", f"{start:.3f}",
        "-i", source,
        "-t", f"{duration:.3f}",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", str(out),
    ]
    try:
        proc = run_ffmpeg(cmd, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timeout for {source}") from exc
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit {proc.returncode}"
        raise RuntimeError(detail)


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return float(proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filters", nargs="*", help="only datasets whose slug contains one of these")
    parser.add_argument("--manifest", default="data/outputs/dataset-previews/g1-dex3/manifest.json")
    parser.add_argument("--datasets-root", default="data/datasets/first_tur_ham/unitree-g1-dex3")
    parser.add_argument("--out-root", default="data/outputs/dataset-previews/g1-dex3/videos")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    datasets_root = Path(args.datasets_root)
    out_root = Path(args.out_root)

    picks: dict[tuple, dict] = {}
    for frame in manifest["frames"]:
        key = (frame["slug"], frame["episode"], frame["cam"])
        picks[key] = frame

    tasks = []
    for (slug, episode, cam), frame in sorted(picks.items()):
        if args.filters and not any(f in slug for f in args.filters):
            continue
        out = out_root / slug / f"ep{episode:06d}_{cam}.mp4"
        local = local_path(datasets_root, frame["dataset"], cam, frame["chunk"], frame["file_index"])
        source = str(local) if local.exists() and local.stat().st_size > 0 else URL_TEMPLATE.format(
            repo=frame["repo"], cam=cam, chunk=frame["chunk"], file=frame["file_index"]
        )
        tasks.append(
            {
                "slug": slug,
                "dataset": frame["dataset"],
                "repo": frame["repo"],
                "episode": episode,
                "cam": cam,
                "task": frame["task"],
                "length": frame["length"],
                "fps": frame["fps"],
                "start": frame["episode_from"],
                "duration": round(frame["episode_to"] - frame["episode_from"], 3),
                "source": source,
                "origin": "local" if source == str(local) else "remote",
                "out": str(out),
            }
        )

    if not tasks:
        print("nothing to export", file=sys.stderr)
        return 2
    total_seconds = sum(t["duration"] for t in tasks)
    print(f"{len(tasks)} clips to export ({total_seconds / 60:.1f} min of video)", flush=True)

    results = []
    failures = []

    def work(task):
        out = Path(task["out"])
        if out.exists() and not args.force:
            got = probe_duration(out)
            return {**task, "written": False, "duration_out": round(got, 3)}
        try:
            export_clip(task["source"], task["start"], task["duration"], out)
            got = probe_duration(out)
            delta = abs(got - task["duration"])
            note = "" if delta <= 0.5 else f" (duration off by {delta:.2f}s)"
            print(f"ok   {task['slug']} ep{task['episode']:06d} {task['cam']} {got:.1f}s{note}", flush=True)
            return {**task, "written": True, "duration_out": round(got, 3)}
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            print(f"FAIL {task['slug']} ep{task['episode']:06d} {task['cam']}: {exc}", flush=True)
            failures.append({**task, "error": str(exc)})
            return None

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for result in pool.map(work, tasks):
            if result:
                results.append(result)

    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "videos_root": str(out_root),
        "clips": results,
        "failed": failures,
    }
    (out_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_root / 'manifest.json'}: {len(results)} clips, {len(failures)} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
