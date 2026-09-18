#!/usr/bin/env python3
"""Measure the interactive Isaac paths under one repeatable procedure.

Every case runs the canonical launcher for a path (``./dev.sh ...``), skips a
warm-up window, and aggregates the steady-state samples the path itself prints:

* ``./dev.sh isaac-g1-sonic[-rough] dex3`` prints one ``[isaac-g1]`` line per
  wall-clock second.  Each line is already a windowed sample of that second
  (physics rate, render rate, RTF, mean step time, mean render-call time).
* ``./dev.sh instinct-parkour`` prints a cumulative ``[perf]`` line every
  ``--report_period`` seconds; a windowed rate is the delta between two lines.

Runs are strictly sequential: one Isaac process at a time, so two runs never
share the GPU.  GPU utilization and host CPU busy time are sampled alongside,
because CPU physics and GPU rendering contend for different resources and the
result is only interpretable with both.

The output is one JSON artifact (default under ``.generated/benchmarks``) with
the per-case command, raw samples, aggregate statistics and the run summary
each path wrote, plus the log file for each case.

    scripts/benchmark-sim.py --list
    scripts/benchmark-sim.py --case sonic-flat-paced --case sonic-rough-paced
    scripts/benchmark-sim.py --warmup 10 --window 20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / ".generated" / "benchmarks"
LOG_DIR = DEFAULT_OUTPUT_DIR / "logs"

# --------------------------------------------------------------------- cases

#: ``pacing`` selects how the case relates to wall clock.  A paced case is the
#: acceptance condition (the path is expected to hold real time); a free case
#: removes the pacing sleep and measures raw headroom.
PACING_ARGS = {
    "sonic": {"paced": [], "free": ["--controller", "none"]},
    "instinct": {"paced": [], "free": ["--no_realtime"]},
}

#: ``ui`` selects what draws the scene.  ``stream`` is the canonical default of
#: both paths: no local window, the full UI served over WebRTC.
UI_ARGS = {
    "stream": [],
    "gui": ["--gui"],
    "headless": ["--headless"],
}


@dataclass(frozen=True)
class Case:
    name: str
    kind: str  # "sonic" | "instinct"
    argv: list[str]
    #: Wall-clock seconds of samples to discard before measuring.
    warmup_s: float
    #: Wall-clock seconds of steady-state samples to keep.
    window_s: float
    #: Hard wall-clock ceiling for the whole process, including Isaac start-up.
    timeout_s: float
    notes: str = ""


def sonic_case(name: str, ui: str, pacing: str, *, extra: list[str], notes: str = "") -> Case:
    assert ui in UI_ARGS and pacing in PACING_ARGS["sonic"]
    if "-rough" in name:
        launcher = ["./dev.sh", "isaac-g1-sonic-rough", "dex3"]
    else:
        launcher = ["./dev.sh", "isaac-g1-sonic", "dex3"]
    return Case(
        name=name,
        kind="sonic",
        argv=[*launcher, *UI_ARGS[ui], *PACING_ARGS["sonic"][pacing], *extra],
        warmup_s=10.0,
        window_s=20.0,
        timeout_s=420.0,
        notes=notes or f"{pacing} physics loop, {ui} UI",
    )


def instinct_case(name: str, ui: str, pacing: str, *, extra: list[str], notes: str = "") -> Case:
    assert ui in UI_ARGS and pacing in PACING_ARGS["instinct"]
    return Case(
        name=name,
        kind="instinct",
        argv=["./dev.sh", "instinct-parkour", *UI_ARGS[ui], *PACING_ARGS["instinct"][pacing], *extra],
        warmup_s=10.0,
        window_s=20.0,
        timeout_s=600.0,
        notes=notes or f"{pacing} policy loop, {ui} UI",
    )


def default_cases() -> list[Case]:
    """The acceptance matrix: three paths, each paced and free, UI streamed."""
    return [
        sonic_case("sonic-flat-paced", "stream", "paced", extra=[]),
        sonic_case("sonic-flat-free", "stream", "free", extra=[]),
        sonic_case("sonic-rough-paced", "stream", "paced", extra=[]),
        sonic_case("sonic-rough-free", "stream", "free", extra=[]),
        instinct_case("instinct-parkour-paced", "stream", "paced", extra=[]),
        instinct_case("instinct-parkour-free", "stream", "free", extra=[]),
    ]


# ------------------------------------------------------------------ sampling


class ResourceSampler:
    """Sample GPU utilization and host CPU busy time once per second."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.gpu: list[dict[str, float]] = []
        self.cpu_busy_percent: list[float] = []
        self._cpu_previous: tuple[int, int] | None = None

    def __enter__(self) -> "ResourceSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _cpu_total_busy(self) -> tuple[int, int]:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    def _run(self) -> None:
        query = "utilization.gpu,memory.used,power.draw,clocks.sm"
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=False,
                ).stdout.strip()
                if output:
                    values = [float(part.strip()) for part in output.splitlines()[0].split(",")]
                    self.gpu.append(
                        dict(zip(("util_percent", "memory_mib", "power_w", "clock_mhz"), values))
                    )
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            try:
                total, idle = self._cpu_total_busy()
                if self._cpu_previous is not None:
                    delta_total = total - self._cpu_previous[0]
                    delta_idle = idle - self._cpu_previous[1]
                    if delta_total > 0:
                        self.cpu_busy_percent.append(100.0 * (delta_total - delta_idle) / delta_total)
                self._cpu_previous = (total, idle)
            except (OSError, ValueError):
                pass
            self._stop.wait(1.0)

    def summary(self) -> dict[str, object]:
        def stats(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"median": None, "max": None, "n": 0}
            return {
                "median": round(statistics.median(values), 2),
                "max": round(max(values), 2),
                "n": len(values),
            }

        return {
            "gpu_util_percent": stats([sample["util_percent"] for sample in self.gpu]),
            "gpu_memory_mib": stats([sample["memory_mib"] for sample in self.gpu]),
            "gpu_power_w": stats([sample["power_w"] for sample in self.gpu]),
            "gpu_clock_mhz": stats([sample["clock_mhz"] for sample in self.gpu]),
            "cpu_busy_percent": stats(self.cpu_busy_percent),
        }


# ------------------------------------------------------------------- parsing

SONIC_RE = re.compile(
    r"\[isaac-g1\] physics=(?P<physics_hz>[\d.]+) Hz\s+render=(?P<render_fps>[\d.]+) FPS\s+"
    r"RTF=(?P<rtf>[\d.]+)x\s+step=(?P<step_ms>[\d.]+) ms\s+render_call=(?P<render_call_ms>[\d.]+) ms\s+"
    r"device=(?P<device>\S+)"
)
#: A run whose cadence is not the default also states it in the line.
SONIC_INTERVAL_RE = re.compile(r"render_interval=(?P<render_interval>\d+)")
INSTINCT_RE = re.compile(
    r"\[perf\] t=\s*(?P<sim_s>[\d.]+)s sim \|\s*(?P<loops_per_s>[\d.]+) loops/s \|\s*"
    r"(?P<env_steps_per_s>[\d.]+) env-steps/s \| RTF\s*(?P<rtf>[\d.]+)x"
)
INSTINCT_TOTAL_RE = re.compile(
    r"\[perf\] TOTAL (?P<steps>\d+) policy steps \| (?P<wall_s>[\d.]+)s wall \| "
    r"(?P<loops_per_s>[\d.]+) loops/s \| (?P<rtf>[\d.]+)x RTF \| (?P<sim_s>[\d.]+)s sim"
)
#: The SONIC runner's own summary, printed by the CLI at the end of a run.
SUMMARY_KEYS = ("physics_ticks", "real_time_factor", "device", "render_interval", "physics_pacing",
                "pacing_overruns", "result")


@dataclass
class Sample:
    wall: float  # seconds since the case started
    values: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        return {"wall_s": round(self.wall, 3), **{k: round(v, 4) for k, v in self.values.items()}}


@dataclass
class CaseResult:
    case: Case
    command: str
    log_path: str
    #: The launcher arguments as actually executed, including the duration and
    #: summary flags the harness adds.  Reporting ``case.argv`` instead would
    #: describe a command the harness never ran.
    executed_argv: list[str] = field(default_factory=list)
    exit_code: int | None = None
    samples: list[Sample] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    summary: dict[str, object] | None = None
    resources: dict[str, object] = field(default_factory=dict)
    error: str | None = None


def replace_case(case: Case, **changes: object) -> Case:
    return Case(**{**dataclasses.asdict(case), **changes})  # type: ignore[arg-type]


#: The running case's launcher, so an interrupted harness cannot leave a
#: simulation behind: the launcher's own trap cleans up its container process.
ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def _stop_active_process(*_args: object) -> None:
    process = ACTIVE_PROCESS
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    raise SystemExit(130)


def run_case(case: Case, label: str) -> CaseResult:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # The label is part of the file names: the same case can legitimately run
    # twice in one campaign (say with and without a flag), and the second run
    # must not overwrite the first run's log or summary.
    log_path = LOG_DIR / f"{label}-{case.name}.log"
    command = " ".join(case.argv)
    result = CaseResult(case=case, command=command, log_path=str(log_path))

    # ``--duration`` bounds the measured part of the run: the warm-up samples
    # are still produced and thrown away here, which keeps the launcher's own
    # flags untouched.  SONIC counts wall seconds, the Instinct playback counts
    # simulated seconds; both are paced to wall clock in the paced cases, so the
    # same number means the same window there.
    duration = case.warmup_s + case.window_s
    argv = [*case.argv, "--duration", f"{duration:g}"]
    metrics_path: Path | None = None
    if case.kind == "sonic":
        summary_name = f"{label}-{case.name}.summary.json"
        metrics_path = DEFAULT_OUTPUT_DIR / summary_name
        # An earlier run of the same case leaves a summary behind.  Reading it
        # as this run's result would attribute one run's ticks and deadline
        # misses to another, so it is removed before the launcher starts.
        metrics_path.unlink(missing_ok=True)
        argv += ["--metrics-output", f"/workspace/humanoid-lab/.generated/benchmarks/{summary_name}"]
    else:
        # The Instinct playback reports every 2 s by default; one sample per
        # second keeps the window well sampled at either pacing.
        argv += ["--report_period", "1.0"]

    result.executed_argv = list(argv)
    result.command = " ".join(argv)
    print(f"[bench] {case.name}: {result.command}  ({case.notes})", flush=True)
    started = time.monotonic()
    samples: list[Sample] = []
    events: list[dict[str, object]] = []
    exit_code: int | None = None
    killed = False

    with ResourceSampler() as sampler:
        process = subprocess.Popen(
            argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        global ACTIVE_PROCESS
        ACTIVE_PROCESS = process
        log_handle = log_path.open("w", encoding="utf-8")

        def pump() -> None:
            # Keep draining until the process closes the pipe: a paused reader
            # fills the 64 KiB pipe buffer and stalls the simulation's logger.
            assert process.stdout is not None
            for line in process.stdout:
                try:
                    log_handle.write(line)
                    log_handle.flush()
                except ValueError:  # the run ended and the log was closed
                    return
                now = time.monotonic() - started
                matched = False
                if case.kind == "sonic":
                    match = SONIC_RE.search(line)
                    if match:
                        values = {key: float(value) for key, value in match.groupdict().items() if key != "device"}
                        interval = SONIC_INTERVAL_RE.search(line)
                        if interval:
                            values["render_interval"] = float(interval.group("render_interval"))
                        samples.append(Sample(now, values))
                        matched = True
                else:
                    match = INSTINCT_RE.search(line)
                    if match:
                        values = {
                            "sim_s": float(match.group("sim_s")),
                            "loops_per_s": float(match.group("loops_per_s")),
                            "env_steps_per_s": float(match.group("env_steps_per_s")),
                            "rtf": float(match.group("rtf")),
                        }
                        samples.append(Sample(now, values))
                        matched = True
                if matched:
                    continue
                if '"event"' in line or "[INFO] perf" in line or "[INFO] scene" in line:
                    text = line.strip()
                    if len(text) <= 240:
                        events.append({"wall_s": round(now, 3), "line": text})

        try:
            pump_thread = threading.Thread(target=pump, daemon=True)
            pump_thread.start()
            deadline = started + case.timeout_s
            while process.poll() is None:
                if time.monotonic() > deadline:
                    killed = True
                    os.killpg(process.pid, signal.SIGTERM)
                    time.sleep(10)
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.5)
            exit_code = process.wait()
            pump_thread.join(timeout=10.0)
        finally:
            log_handle.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            ACTIVE_PROCESS = None

    result.exit_code = exit_code
    result.samples = samples
    result.events = events
    result.resources = sampler.summary()
    if killed:
        result.error = f"killed after {case.timeout_s:g}s"
    if metrics_path is not None and metrics_path.exists():
        try:
            result.summary = json.loads(metrics_path.read_text())
        except json.JSONDecodeError as exc:
            result.error = f"unreadable summary: {exc}"
    elif case.kind == "sonic":
        # The run may have been killed before it could write its summary; the
        # tail of the log still carries the same JSON object.
        result.summary = _summary_from_log(log_path)

    elapsed = time.monotonic() - started
    print(
        f"[bench] {case.name}: exit={exit_code} samples={len(samples)} wall={elapsed:.1f}s"
        + (f" error={result.error}" if result.error else ""),
        flush=True,
    )
    return result


def _summary_from_log(log_path: Path) -> dict[str, object] | None:
    """Recover the run summary the CLI printed, if the summary file is absent."""
    text = log_path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    for start in [index for index, char in enumerate(text) if char == "{"][::-1]:
        try:
            decoded, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and any(key in decoded for key in SUMMARY_KEYS):
            return decoded
    return None


# ---------------------------------------------------------------- aggregation


def steady_samples(result: CaseResult) -> list[Sample]:
    """The measured window: samples after the warm-up and inside the window.

    The harness asks each path for ``warmup + window`` of run time, but the two
    paths count different clocks — the G1 runner bounds wall seconds, the
    Instinct playback bounds *simulated* seconds, so at RTF below 1 it keeps
    running far past the wall window — and a shutdown can emit a trailing
    sample.  Applying the window again here, on the wall timestamps the harness
    itself recorded, is what makes the two comparable.
    """
    if not result.samples:
        return []
    start = result.samples[0].wall + result.case.warmup_s
    stop = start + result.case.window_s
    return [sample for sample in result.samples if start <= sample.wall <= stop]


def aggregate(result: CaseResult) -> dict[str, object]:
    samples = steady_samples(result)
    entry: dict[str, object] = {
        "case": result.case.name,
        "kind": result.case.kind,
        "notes": result.case.notes,
        "command": result.command,
        "argv": result.executed_argv or result.case.argv,
        "exit_code": result.exit_code,
        "log": os.path.relpath(result.log_path, ROOT),
        "warmup_s": result.case.warmup_s,
        "window_s": result.case.window_s,
        "sample_count": len(result.samples),
        "steady_sample_count": len(samples),
        "error": result.error,
        "resources": result.resources,
    }
    if result.summary is not None:
        entry["run_summary"] = {
            key: result.summary.get(key)
            for key in ("result", "physics_ticks", "real_time_factor", "device", "render_interval",
                        "physics_pacing", "pacing_overruns", "profile_id", "step_ms_by_phase")
            if key in result.summary
        }
    if not samples:
        entry["metrics"] = {}
        entry["samples"] = []
        return entry

    def column(name: str) -> list[float]:
        return [sample.values[name] for sample in samples if name in sample.values]

    metrics: dict[str, object] = {}
    for name in ("physics_hz", "render_fps", "rtf", "step_ms", "render_call_ms",
                 "loops_per_s", "env_steps_per_s"):
        values = column(name)
        if not values:
            continue
        metrics[name] = {
            "median": round(statistics.median(values), 3),
            "mean": round(statistics.fmean(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "p95": round(sorted(values)[min(len(values) - 1, int(0.95 * len(values)))], 3),
            "n": len(values),
        }
    if result.case.kind == "instinct":
        # The playback's own numbers are cumulative; a windowed rate is the
        # delta between consecutive samples.
        windowed: list[float] = []
        for previous, current in zip(samples, samples[1:]):
            delta_wall = current.wall - previous.wall
            if delta_wall > 0:
                windowed.append((current.values["sim_s"] - previous.values["sim_s"]) / delta_wall)
        if windowed:
            metrics["windowed_rtf"] = {
                "median": round(statistics.median(windowed), 3),
                "mean": round(statistics.fmean(windowed), 3),
                "min": round(min(windowed), 3),
                "max": round(max(windowed), 3),
                "n": len(windowed),
            }
    if result.case.kind == "sonic":
        # A paced run misses its deadline whenever a second of simulated time
        # did not fit in a second of wall clock; the runner counts those.
        summary = result.summary or {}
        overruns = summary.get("pacing_overruns")
        metrics["deadline_misses"] = overruns if isinstance(overruns, int) else None
    entry["metrics"] = metrics
    entry["measured_window"] = {
        "warmup_s": result.case.warmup_s,
        "requested_window_s": result.case.window_s,
        "first_sample_s": round(samples[0].wall, 3),
        "last_sample_s": round(samples[-1].wall, 3),
        "covered_s": round(samples[-1].wall - samples[0].wall, 3),
        "samples": len(samples),
        "rate_per_s": round(len(samples) / (samples[-1].wall - samples[0].wall), 2)
        if samples[-1].wall > samples[0].wall
        else None,
    }
    entry["samples"] = [sample.as_dict() for sample in samples]
    return entry


def print_table(results: list[dict[str, object]]) -> None:
    header = f"{'case':<26} {'rate':>7} {'RTF':>6} {'FPS':>7} {'step ms':>8} {'render ms':>9} {'misses':>7} {'err':>4}"
    print("\n" + header)
    print("-" * len(header))
    for entry in results:
        metrics = entry.get("metrics") or {}
        if not metrics:
            print(f"{entry['case']:<26} {'-':>7} {'-':>6} {'-':>7} {'-':>8} {'-':>9} {'-':>7} {'!':>4}")
            continue
        physics = metrics.get("physics_hz", {}).get("median")
        loops = metrics.get("loops_per_s", {}).get("median")
        rtf = metrics.get("windowed_rtf", metrics.get("rtf", {})).get("median")
        render_fps = metrics.get("render_fps", {}).get("median")
        step = metrics.get("step_ms", {}).get("median")
        render_ms = metrics.get("render_call_ms", {}).get("median")
        misses = metrics.get("deadline_misses")
        rate = physics if physics is not None else loops
        print(
            f"{entry['case']:<26} {rate:>7.1f} {rtf:>6.2f} "
            f"{(render_fps if render_fps is not None else float('nan')):>7.1f} "
            f"{(step if step is not None else float('nan')):>8.2f} "
            f"{(render_ms if render_ms is not None else float('nan')):>9.2f} "
            f"{('-' if misses is None else misses):>7} "
            f"{('!' if entry.get('error') else ''):>4}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=[], help="case name to run (repeatable; default: all)")
    parser.add_argument("--list", action="store_true", help="list the case matrix and exit")
    parser.add_argument("--warmup", type=float, default=None, help="seconds of samples to discard")
    parser.add_argument("--window", type=float, default=None, help="seconds of steady-state samples to keep")
    parser.add_argument("--label", default=None, help="artifact label (default: UTC timestamp)")
    parser.add_argument("--output", type=Path, default=None, help="artifact path")
    parser.add_argument("--kinds", default=None, help="comma-separated subset of path kinds: sonic,instinct")
    parser.add_argument(
        "--perf-detail",
        action="store_true",
        help="ask each path for its per-phase cost breakdown (each path spells its own flag "
             "differently, so this is translated per case)",
    )
    parser.add_argument(
        "--extra", default=None,
        help="extra arguments appended to every selected case's launcher (for ad-hoc runs, "
             "e.g. --extra '--no-dlssg', or --extra '--render-interval 16' for the G1 paths)",
    )
    args = parser.parse_args()

    cases = default_cases()
    if args.kinds:
        wanted = {value.strip() for value in args.kinds.split(",") if value.strip()}
        cases = [case for case in cases if case.kind in wanted]
    if args.case:
        selected = set(args.case)
        unknown = selected - {case.name for case in cases}
        if unknown:
            print(f"unknown case(s): {sorted(unknown)}", file=sys.stderr)
            print("known: " + ", ".join(case.name for case in cases), file=sys.stderr)
            return 2
        cases = [case for case in cases if case.name in selected]
    if args.list:
        for case in cases:
            print(f"{case.name:<26} {case.kind:<9} warmup={case.warmup_s:g}s window={case.window_s:g}s  {case.notes}")
        return 0

    if args.perf_detail or args.extra:
        extras = shlex.split(args.extra) if args.extra else []
        for index, case in enumerate(cases):
            detail = []
            if args.perf_detail:
                detail = ["--perf-detail"] if case.kind == "sonic" else ["--perf_detail"]
            case_extras = [*detail, *extras]
            if case_extras:
                cases[index] = replace_case(case, argv=[*case.argv, *case_extras])

    if args.warmup is not None or args.window is not None:
        cases = [
            Case(
                name=case.name, kind=case.kind, argv=case.argv,
                warmup_s=args.warmup if args.warmup is not None else case.warmup_s,
                window_s=args.window if args.window is not None else case.window_s,
                timeout_s=case.timeout_s, notes=case.notes,
            )
            for case in cases
        ]

    signal.signal(signal.SIGINT, _stop_active_process)
    signal.signal(signal.SIGTERM, _stop_active_process)

    label = args.label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (DEFAULT_OUTPUT_DIR / f"{label}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    artifact: dict[str, object] = {
        "schema": "humanoid-lab.benchmark/1",
        "label": label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "results": [],
    }
    for case in cases:
        result = run_case(case, label)
        artifact["results"].append(aggregate(result))
        # Write after every case so an interrupted matrix still leaves evidence.
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_table(artifact["results"])  # type: ignore[arg-type]
    print("\nrate: physics Hz for the isaac-g1 cases, policy loops/s for instinct-parkour.")
    print("RTF: for instinct cases the windowed rate derived from consecutive samples.")
    print(f"\n[bench] artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
