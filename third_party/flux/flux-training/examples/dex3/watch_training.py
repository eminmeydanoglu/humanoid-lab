"""Live terminal view of a ``peft_smoke.py --mode pilot`` run, reading its stdout log.

Renders the newest optimizer steps, loss/grad sparklines, the validation points and the last
checkpoint, and derives an ETA from the observed step rate. Safe to run at any time; it only reads.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

SPARK = "▁▂▃▄▅▆▇█"
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
CYAN, GREEN, YELLOW, RED, MAGENTA = "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[35m"


def parse_log(path: Path) -> dict:
    """Collect the newest state from the training log (missing file is an empty state)."""
    state = {"steps": [], "validations": [], "checkpoint": None, "finished": False, "error": None}
    if not path.exists():
        return state
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("OPTSTEP "):
            state["steps"].append(json.loads(line[len("OPTSTEP ") :]))
        elif line.startswith("VAL "):
            state["validations"].append(json.loads(line[len("VAL ") :]))
        elif line.startswith("CHECKPOINT "):
            state["checkpoint"] = json.loads(line[len("CHECKPOINT ") :])
        elif line.startswith(("PILOT ", "Traceback", "RuntimeError", "torch.OutOfMemoryError")):
            state["finished"] = line.startswith("PILOT ")
            state["error"] = None if state["finished"] else line.strip()
    return state


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    span = high - low or 1.0
    return "".join(
        SPARK[min(len(SPARK) - 1, int((value - low) / span * (len(SPARK) - 1)))] for value in values
    )


def human_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, minutes = divmod(seconds // 60, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds % 60:02d}s"


def grad_color(grad: float) -> str:
    if grad > 120:
        return RED
    if grad > 60:
        return YELLOW
    return GREEN


def render(state: dict, args, session: dict) -> str:
    width = max(78, min(shutil.get_terminal_size(fallback=(110, 40)).columns, 140))
    rule = f"{DIM}{'─' * (width - 1)}{RESET}"
    spark_len = max(16, min(args.spark, width - 14))
    steps = state["steps"]
    out = [f"{BOLD}{CYAN} G1 Dex3 PEFT/LoRA — LeRobot FLUX3 action policy{RESET}"]
    out.append(rule)
    if state["error"]:
        out.append(f"{RED}{BOLD} TRAINING ERROR{RESET}  {state['error'][:90]}")
    elif state["finished"]:
        out.append(f"{BOLD}{GREEN} TRAINING FINISHED{RESET}")
    elif not steps:
        out.append(f"{YELLOW} waiting for the first optimizer step (model load takes ~25 s)…{RESET}")
    total = args.steps
    last = steps[-1]["step"] if steps else 0
    if steps and session.get("first") is None:
        session["first"], session["t0"] = last, time.time()
    elapsed = time.time() - session["t0"] if session.get("t0") else 0.0
    done = last - session["first"] + 1 if session.get("first") is not None else 0
    rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
    eta = (total - last) / rate if rate else 0.0
    age = f"{time.time() - args.log.stat().st_mtime:.0f}s" if args.log.exists() else "n/a"
    out.append(
        f" step {BOLD}{last:>5}{RESET} / {total}   {100 * last / total:5.1f}%   "
        f"session {human_time(elapsed)}   {rate:.2f} step/s   ETA {human_time(eta)}   log age {age}"
    )
    if steps:
        newest = steps[-1]
        out.append(
            f" loss {BOLD}{newest['loss']:7.3f}{RESET}   action {newest['action_mse']:6.3f}   "
            f"video {newest['video_mse']:6.4f}   grad {grad_color(newest['grad_norm'])}"
            f"{newest['grad_norm']:7.2f}{RESET}   lr {newest['lrs'][0]:g}/{newest['lrs'][-1]:g}   "
            f"{newest['step_time_s']:5.2f} s/step   rss {newest['host_rss_mib'] / 1024:4.2f} GiB"
        )
        window = steps[-spark_len:]
        out.append(f" {DIM}loss{RESET} {CYAN}{sparkline([s['loss'] for s in window])}{RESET}")
        out.append(f" {DIM}grad{RESET} {YELLOW}{sparkline([s['grad_norm'] for s in window])}{RESET}")
    out.append(rule)
    out.append(f"{DIM}   step      loss    action     video      grad    s/step    rss GiB   vram GiB{RESET}")
    for entry in steps[-args.rows :]:
        out.append(
            f"  {entry['step']:>5}  {entry['loss']:8.3f}  {entry['action_mse']:8.4f}  "
            f"{entry['video_mse']:8.4f}  {grad_color(entry['grad_norm'])}{entry['grad_norm']:7.2f}{RESET}  "
            f"{entry['step_time_s']:7.2f}  {entry['host_rss_mib'] / 1024:8.2f}  "
            f"{entry['cuda_reserved_mib'] / 1024:8.2f}"
        )
    out.append(rule)
    out.append(f" {BOLD}validation{RESET} {DIM}(32 fixed windows, raw | EMA){RESET}")
    if state["validations"]:
        for point in state["validations"][-6:]:
            raw = point.get("raw")
            ema = point.get("ema")
            line = (
                f"   step {point['step']:>5}   raw loss {raw['loss']:7.3f}  action {raw['action_mse']:6.3f}"
                f"  video {raw['video_mse']:6.4f}"
            )
            if ema:
                line += f"   {MAGENTA}ema loss {ema['loss']:7.3f}{RESET}"
            out.append(line)
    else:
        out.append(f"{DIM}   none yet{RESET}")
    checkpoint = state["checkpoint"]
    if checkpoint:
        pruned = (
            f"  {DIM}(pruned: {', '.join(checkpoint['pruned'])}){RESET}" if checkpoint.get("pruned") else ""
        )
        out.append(rule)
        directory = str(checkpoint["dir"])
        if len(directory) > width - 40:
            directory = "…" + directory[-(width - 41) :]
        out.append(f" {BOLD}last checkpoint{RESET} step {checkpoint['step']}  {directory}{pruned}")
    out.append(rule)
    out.append(f" {DIM}TensorBoard: {args.url}   refresh {args.interval}s   quit with Ctrl-C{RESET}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("outputs/dex3/peft-train/train.log"))
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--spark", type=int, default=96)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--url", default="http://aksoy.tail2e36f3.ts.net:6006")
    args = parser.parse_args()
    session: dict = {"first": None, "t0": None}
    try:
        while True:
            frame = render(parse_log(args.log), args, session)
            print("\033[2J\033[H" + frame, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped watching (training keeps running)")


if __name__ == "__main__":
    main()
