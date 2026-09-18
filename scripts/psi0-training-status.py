#!/usr/bin/env python3
"""Render Psi0 tqdm output as stable, line-oriented training status."""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_RE = re.compile(
    r"Training steps:.*?\|\s*(?P<step>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[^]<]+)<(?P<eta>[^],]+),\s*"
    r"(?P<rate>[^,\]]+)"
    r"(?:,\s*loss=(?P<loss>[^,\]]+),\s*lr=(?P<lr>[^\]]+))?\]"
)
EVAL_RE = re.compile(r"Eval at global step (?P<step>\d+):")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def clean(text: str) -> str:
    return ANSI_RE.sub("", text).strip()


def parse_rate_seconds(rate: str) -> float | None:
    match = re.fullmatch(r"\s*([0-9.]+)s/it\s*", rate)
    if match:
        return float(match.group(1))
    match = re.fullmatch(r"\s*([0-9.]+)it/s\s*", rate)
    if match and float(match.group(1)) > 0:
        return 1.0 / float(match.group(1))
    return None


def compact_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    return f"{hours:02d}h{minutes:02d}m"


def format_progress(match: re.Match[str]) -> str:
    step = int(match.group("step"))
    total = int(match.group("total"))
    rate_text = match.group("rate").strip()
    seconds_per_step = parse_rate_seconds(rate_text)
    if seconds_per_step is not None:
        remaining_seconds = (total - step) * seconds_per_step
        eta = compact_duration(remaining_seconds)
    else:
        eta = match.group("eta").strip()

    loss = (match.group("loss") or "?").strip()
    percent = 100.0 * step / total if total else 0.0
    timestamp = datetime.now().strftime("%H:%M:%S")
    return (
        f"{timestamp} | {step:>5}/{total:<5} | {percent:>6.2f}% | "
        f"loss {loss:>7} | {rate_text:>8} | ETA {eta}"
    )


def records(text: str) -> list[str]:
    return [clean(record) for record in re.split(r"[\r\n]+", text) if clean(record)]


def newest_progress(text: str) -> re.Match[str] | None:
    latest = None
    for record in records(text):
        match = PROGRESS_RE.search(record)
        if match and match.group("loss") is not None:
            latest = match
    return latest


def checkpoint_steps(root: Path | None) -> set[int]:
    if root is None or not root.exists():
        return set()
    found = set()
    for path in root.glob("ckpt_*"):
        suffix = path.name.removeprefix("ckpt_")
        if path.is_dir() and suffix.isdigit():
            found.add(int(suffix))
    return found


def main() -> int:
    args = parse_args()
    while not args.log.exists():
        if args.once:
            raise SystemExit(f"log not found: {args.log}")
        time.sleep(args.poll_seconds)

    with args.log.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        stream.seek(max(0, end - 4 * 1024 * 1024))
        initial = stream.read(end - stream.tell())
        stream.seek(end)

        print("Psi0 training — clean live status", flush=True)
        print("time     | step        | done    | training loss | speed    | remaining", flush=True)
        print("-" * 78, flush=True)

        latest = newest_progress(initial)
        last_step = -1
        pending: re.Match[str] | None = latest
        pending_updated = time.monotonic()
        if latest:
            print(format_progress(latest), flush=True)
            last_step = int(latest.group("step"))

        seen_evals = {int(match.group("step")) for match in EVAL_RE.finditer(clean(initial))}
        seen_checkpoints = checkpoint_steps(args.checkpoints)
        if seen_checkpoints:
            print(f"[checkpoint] existing: {', '.join(map(str, sorted(seen_checkpoints)))}", flush=True)
        if args.once:
            return 0

        buffer = ""
        while True:
            chunk = stream.read()
            if chunk:
                buffer += chunk
                parts = re.split(r"([\r\n]+)", buffer)
                buffer = parts[-1]
                for index in range(0, len(parts) - 1, 2):
                    record = clean(parts[index])
                    if not record:
                        continue
                    progress = PROGRESS_RE.search(record)
                    if progress and progress.group("loss") is not None:
                        step = int(progress.group("step"))
                        if step >= last_step:
                            pending = progress
                            pending_updated = time.monotonic()
                    eval_match = EVAL_RE.search(record)
                    if eval_match:
                        eval_step = int(eval_match.group("step"))
                        if eval_step not in seen_evals:
                            seen_evals.add(eval_step)
                            print(f"[validation] started at step {eval_step}", flush=True)

            if pending is not None:
                step = int(pending.group("step"))
                if step > last_step and time.monotonic() - pending_updated >= 0.6:
                    print(format_progress(pending), flush=True)
                    last_step = step
                    pending = None

            current_checkpoints = checkpoint_steps(args.checkpoints)
            for step in sorted(current_checkpoints - seen_checkpoints):
                print(f"[checkpoint] ckpt_{step} saved", flush=True)
            seen_checkpoints = current_checkpoints
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
