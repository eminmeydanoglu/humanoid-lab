#!/usr/bin/env python3
"""Drive reproducible BlockStacking rollouts through the shared evaluation UI.

This is a **host** script: it runs ``./dev.sh psi0-isaac-eval`` (Isaac
BlockStacking + the SONIC Y controller + the bridge that owns the policy server
and the UI), waits for the stack to be ready, and then drives the same
``Start``/``Stop``/``Reset`` the operator clicks -- over the HTTP API -- so a
campaign is a command instead of a hand on a mouse.  Every rollout boundary is
recorded with the host clock, which the container shares, so the bridge
telemetry, the Isaac samples and the video frame map can be lined up afterwards.

Nothing is inferred here: the driver records what the API answered, the exact
commands it ran, and the artifact paths.  Onset detection, video cropping and
the per-rollout summary belong to ``scripts/analyze-blockstacking-rollout.py``.

    scripts/blockstacking-rollout.py \
        --psi-checkpoint-dir data/outputs/psi0-unitree-dex3-sonic-v1/finetune/<run> \
        --checkpoint-step 40000 \
        --groot-checkpoint-dir data/outputs/groot-unitree-dex3-sonic-v1/finetune/<run>/checkpoint-40000 \
        --model psi --rollouts 2 --rollout-seconds 45

The launcher writes its own logs under ``$HUMANOID_DATA_ROOT/outputs/psi0-isaac-eval/<tag>/``;
they are copied into ``<out-dir>/logs/`` when the session ends.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
SESSION_SCHEMA_VERSION = 1
DEFAULT_UI_PORT = 8015
READY_TIMEOUT_S = 900.0
SWITCH_TIMEOUT_S = 1200.0
SHUTDOWN_TIMEOUT_S = 180.0
#: The Isaac control endpoint inside the container (``--control-endpoint``).
ISAAC_CONTROL_ENDPOINT = "tcp://127.0.0.1:5559"
CONTAINER_NAME = "humanoid-lab-dev"
#: Time Isaac gets to finalize the video, the summary, the tracking Parquet and
#: the sample stream after a shutdown request.
ISAAC_FINALIZE_TIMEOUT_S = 180.0


class DriverError(RuntimeError):
    """The campaign cannot continue; the message is meant for the operator."""


# -- paths ----------------------------------------------------------------


def data_root() -> Path:
    configured = os.environ.get("HUMANOID_DATA_ROOT")
    return Path(configured).expanduser() if configured else DEFAULT_DATA_ROOT


def container_path(path: Path, root: Path | None = None) -> str:
    """The in-container spelling of a data-root path (the bind mounts)."""
    root = root or data_root()
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (ROOT / resolved).resolve()
    outputs = (root / "outputs").resolve()
    if resolved == outputs or outputs in resolved.parents:
        return "/outputs/" + str(resolved.relative_to(outputs))
    if resolved == root or root in resolved.parents:
        return "/data/" + str(resolved.relative_to(root))
    return str(resolved)


def inside_container_path(path: str, root: Path) -> Path:
    """Map a container spelling back to its host path for the checks that matter."""
    if path.startswith("/outputs/"):
        return root / "outputs" / path[len("/outputs/") :]
    if path.startswith("/data/"):
        return root / "data" / path[len("/data/") :]
    if path.startswith("/workspace/humanoid-lab/"):
        return ROOT / path[len("/workspace/humanoid-lab/") :]
    return Path(path)


# -- the bridge's HTTP surface --------------------------------------------


class Ui:
    """The evaluation UI's control API, with the failures it can answer."""

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self.base = f"http://{host}:{port}"

    def _call(self, path: str, *, method: str = "GET", body: dict | None = None, timeout: float = 10.0) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise DriverError(f"{method} {path} -> HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise DriverError(f"{method} {path} -> {type(exc).__name__}: {exc}") from None
        if not payload:
            return None
        return json.loads(payload.decode("utf-8"))

    def status(self) -> dict[str, Any]:
        return self._call("/api/status")

    def checkpoints(self) -> dict[str, Any]:
        return self._call("/api/checkpoints")

    def meta(self) -> dict[str, Any]:
        return self._call("/api/meta")

    def select_model(self, model_id: str) -> dict[str, Any]:
        return self._call("/api/checkpoint", method="POST", body={"id": model_id}, timeout=SHUTDOWN_TIMEOUT_S)

    def start(self, instruction: str) -> dict[str, Any]:
        return self._call("/api/start", method="POST", body={"instruction": instruction})

    def stop(self) -> dict[str, Any]:
        return self._call("/api/stop", method="POST")

    def reset(self) -> dict[str, Any]:
        # A reset on a GR00T selection stops and restarts NVIDIA's policy server
        # before it answers, so this call is minutes long by design.
        return self._call("/api/reset", method="POST", timeout=SWITCH_TIMEOUT_S)


def wait_for_ui(ui: Ui, launcher: subprocess.Popen, log_path: Path, timeout_s: float = READY_TIMEOUT_S) -> dict[str, Any]:
    """Wait until the bridge answers; a dead launcher fails here, not at the end."""
    deadline = time.monotonic() + timeout_s
    last = "no response yet"
    while time.monotonic() < deadline:
        if launcher.poll() is not None:
            raise DriverError(
                f"the launcher exited with {launcher.returncode} before the UI was ready; "
                f"see {log_path}"
            )
        try:
            meta = ui.meta()
            if isinstance(meta, dict) and meta.get("prompt"):
                return meta
            last = f"unexpected /api/meta payload: {meta!r}"
        except DriverError as exc:
            last = str(exc)
        time.sleep(2.0)
    raise DriverError(f"the UI did not become ready within {timeout_s:.0f}s ({last})")


def wait_for_model(ui: Ui, model_id: str, timeout_s: float = SWITCH_TIMEOUT_S) -> dict[str, Any]:
    """Wait until the selected backend is serving; a failed switch is reported verbatim."""
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = ui.checkpoints()
        selected = (last.get("selected") or {}).get("id")
        if selected == model_id and last.get("serving_selected") and not last.get("switching"):
            return last
        if last.get("switching"):
            time.sleep(2.0)
            continue
        time.sleep(2.0)
    raise DriverError(f"model {model_id!r} was not serving within {timeout_s:.0f}s; last: {json.dumps(last)[:400]}")


def wait_for_idle(ui: Ui, timeout_s: float = 60.0) -> dict[str, Any]:
    """The session must be restartable before a Start is attempted."""
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = ui.status()
        if last.get("state") in {"IDLE", "STOPPED"} and not last.get("starting"):
            return last
        time.sleep(1.0)
    raise DriverError(f"session did not come back to IDLE/STOPPED: {json.dumps(last)[:400]}")


def wait_for_running(ui: Ui, timeout_s: float = 60.0) -> dict[str, Any]:
    """Start returns before the first action lands; wait for the stream to be live."""
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = ui.status()
        if last.get("state") == "RUNNING":
            return last
        if last.get("state") == "ERROR":
            raise DriverError(f"session went to ERROR during Start: {last.get('error')!r}")
        time.sleep(0.5)
    raise DriverError(f"session did not reach RUNNING within {timeout_s:.0f}s: {json.dumps(last)[:400]}")


# -- launcher --------------------------------------------------------------


def build_launcher_command(args: argparse.Namespace, paths: dict[str, str]) -> list[str]:
    command = [
        str(ROOT / "dev.sh"), "psi0-isaac-eval",
        "--checkpoint-dir", str(args.psi_checkpoint_dir),
        "--checkpoint-step", str(args.checkpoint_step),
        "--groot-checkpoint-dir", str(args.groot_checkpoint_dir),
        "--telemetry-dir", paths["telemetry"],
        "--record-video", paths["video_raw"],
        "--video-timestamps-output", paths["video_timestamps"],
        "--samples-output", paths["isaac_samples"],
        "--tracking-output", paths["isaac_tracking"],
        "--metrics-output", paths["isaac_metrics"],
        "--policy-clock", args.policy_clock,
        "--policy-clock-timeout-s", f"{args.policy_clock_timeout_s:g}",
    ]
    if args.policy_clock == "simulation":
        command.extend(["--policy-clock-file", paths["policy_clock"]])
    elif paths.get("run_clock"):
        command.extend(["--replay-clock-output", paths["run_clock"]])
    if args.groot_capture_dir is not None:
        command.extend(["--groot-capture-dir", paths["groot_capture"]])
        command.extend(["--groot-capture-max-requests", str(args.groot_capture_max_requests)])
    command.extend(["--groot-left-hand-contract", args.groot_left_hand_contract])
    command.extend(["--psi0-neck-policy", args.psi0_neck_policy])
    if args.psi0_rtc_off:
        command.append("--psi0-rtc-off")
    if args.headless:
        command.append("--headless")
    elif args.gui:
        command.append("--gui")
    if args.reset_pose_file is not None:
        # The simulator reads the file on its own side of the mount; hand it the
        # in-container spelling rather than the host path.
        resolved = resolve_host_path(args.reset_pose_file, data_root())
        if str(resolved).startswith(str(ROOT)):
            inside = "/workspace/humanoid-lab/" + str(resolved.relative_to(ROOT))
        else:
            inside = container_path(resolved)
        command.extend(["--reset-pose-file", inside])
    if args.reset_pose_hold:
        command.append("--reset-pose-hold")
    if args.warmstart_tokens is not None:
        # The bridge reads the stream; same in-container spelling as above.
        resolved = resolve_host_path(args.warmstart_tokens, data_root())
        if str(resolved).startswith(str(ROOT)):
            inside = "/workspace/humanoid-lab/" + str(resolved.relative_to(ROOT))
        else:
            inside = container_path(resolved)
        command.extend(["--warmstart-tokens", inside])
        command.extend(["--warmstart-delay-s", f"{warmstart_delay_seconds(args):g}"])
        sim_clock = getattr(args, "warmstart_sim_clock", None)
        if sim_clock is not None:
            resolved_clock = resolve_host_path(sim_clock, data_root())
            clock_inside = container_path(resolved_clock)
            command.extend(["--replay-clock-output", clock_inside])
            command.extend(["--warmstart-sim-clock", clock_inside])
            command.extend(["--warmstart-clock-timeout-s", f"{float(getattr(args, 'warmstart_clock_timeout_s', 5.0)):g}"])
    if args.initial_pose_handshake:
        command.append("--initial-pose-handshake")
    if args.scene_profile is not None:
        # Isaac is launched with the checkout mounted at /workspace/humanoid-lab
        # and the profile path prefixed by it, so the variant is named by its
        # own path; dev.sh rewrites a host path inside the checkout itself.
        command.extend(["--scene-profile", str(args.scene_profile)])
    if args.gravity_feedforward:
        command.append("--gravity-feedforward")
    return command


def warmstart_delay_seconds(args: argparse.Namespace) -> float:
    """When the demo token stream starts, measured from the end of the Reset.

    The stream occupies the settle's own tail (``--warmstart-window-seconds``),
    so the canonical part of the settle is the same as the baseline's and only
    the window before Start differs between the cells.
    """
    delay = float(args.settle_seconds) - float(args.warmstart_window_seconds)
    if delay < 0:
        raise DriverError(
            f"--warmstart-window-seconds {args.warmstart_window_seconds:g} is longer than the "
            f"settle --settle-seconds {args.settle_seconds:g}; the stream would not fit inside it"
        )
    return delay


def container_name() -> str:
    completed = subprocess.run(
        ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=False,
    )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return names[0] if names else CONTAINER_NAME


def request_isaac_shutdown(endpoint: str = ISAAC_CONTROL_ENDPOINT, *, timeout_ms: int = 5000) -> str | None:
    """Ask Isaac to end its run through its own control endpoint.

    Killing the process group instead interrupts ffmpeg mid-trailer and leaves
    the run summary and the tracking Parquet unwritten, so the campaign asks
    Isaac to stop and then waits for it to finish its artifacts.  The request
    goes through the container because only that environment ships pyzmq.
    """
    client = (
        "import zmq; ctx=zmq.Context(); sock=ctx.socket(zmq.REQ); "
        "sock.setsockopt(zmq.LINGER, 0); sock.connect(" + repr(endpoint) + "); "
        "sock.send(b'shutdown'); "
        f"print(sock.recv().decode() if sock.poll({int(timeout_ms)}) else 'no reply'); "
        "sock.close(); ctx.term()"
    )
    command = (
        "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && exec python -c "
        + shlex.quote(client)
    )
    completed = subprocess.run(
        ["docker", "exec", container_name(), "bash", "-lc", command],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def wait_for_artifacts(paths: Sequence[Path], timeout_s: float) -> dict[str, bool]:
    """Wait for Isaac's end-of-run artifacts; report each one's existence."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(path.is_file() for path in paths):
            break
        time.sleep(1.0)
    return {str(path): path.is_file() for path in paths}


def terminate_launcher(launcher: subprocess.Popen, log_path: Path, *, grace_s: float = 0.0) -> int | None:
    """Let the launcher finish on its own, then Ctrl-C its group if needed."""
    if grace_s > 0:
        try:
            return launcher.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    if launcher.poll() is None:
        try:
            os.killpg(os.getpgid(launcher.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            launcher.send_signal(signal.SIGINT)
    try:
        return launcher.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"[rollout] launcher ignored SIGINT; killing it (see {log_path})", file=sys.stderr)
        try:
            os.killpg(os.getpgid(launcher.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            launcher.kill()
        try:
            launcher.wait(timeout=30)
        except subprocess.TimeoutExpired:
            return None
    return launcher.returncode


def read_sim_clock(path: Path) -> float | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw.endswith(b"\n"):
        return None
    try:
        value = float(raw.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError):
        return None
    return value if value >= 0.0 else None


def wait_for_sim_horizon(path: Path, seconds: float, *, timeout_s: float) -> dict[str, float]:
    deadline = time.monotonic() + timeout_s
    start = None
    last = None
    while time.monotonic() < deadline:
        sample = read_sim_clock(path)
        if sample is not None:
            if start is None:
                start = sample
            if last is not None and sample < last - 1e-9:
                raise DriverError(f"Isaac simulation clock moved backwards during rollout ({last:.6f} -> {sample:.6f})")
            last = sample
            if sample - start >= seconds:
                return {"start_sim_s": start, "stop_request_sim_s": sample, "elapsed_sim_s": sample - start}
        time.sleep(0.01)
    raise DriverError(
        f"Isaac simulation clock did not advance {seconds:g}s within {timeout_s:g}s; "
        f"start={start!r}, last={last!r}"
    )


def find_launcher_log_dir(log_path: Path) -> str | None:
    """The launcher prints where the container-side logs went; that is authoritative."""
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            marker = "[psi0-isaac-eval] logs       : "
            if line.startswith(marker):
                return line[len(marker) :].strip()
    except OSError:
        return None
    return None


# -- arguments -------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive BlockStacking rollouts through the eval UI")
    parser.add_argument("--psi-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--groot-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--policy-clock", choices=("wall", "simulation"), default="simulation",
                        help="both models use this session's Isaac replay clock by default; wall reproduces legacy runs")
    parser.add_argument("--policy-clock-file", type=Path, default=None,
                        help="simulation clock path; defaults to <out-dir>/raw/isaac-policy-clock.txt")
    parser.add_argument("--policy-clock-timeout-s", type=float, default=5.0,
                        help="wall watchdog for an unavailable or stalled simulation clock")
    parser.add_argument("--groot-capture-dir", type=Path, default=None,
                        help="opt-in lossless GR00T request/response capture directory")
    parser.add_argument("--groot-capture-max-requests", type=int, default=32)
    parser.add_argument(
        "--groot-left-hand-contract",
        choices=("compatibility", "model-independent", "model-coupled"),
        default="model-independent",
    )
    parser.add_argument("--psi0-neck-policy", choices=("error", "discard"), default="discard",
                        help="discard masked 80D neck padding (default); error restores strict rejection")
    parser.add_argument("--psi0-rtc-off", action="store_true",
                        help="run psi0 with independent unguided chunks instead of guided RTC")
    parser.add_argument("--model", choices=("psi", "groot"), default="psi",
                        help="backend this session rolls out; the launcher serves both")
    parser.add_argument("--model-id", default=None,
                        help="explicit allowlisted model id (for example fine-tuned); "
                             "defaults to fine-tuned for psi")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--rollout-seconds", type=float, default=45.0)
    parser.add_argument("--rollout-sim-seconds", type=float, default=None,
                        help="stop each active rollout after this many Isaac simulation seconds")
    parser.add_argument("--settle-seconds", type=float, default=12.0,
                        help="wait after a Reset before Start (support band re-armed, robot still)")
    parser.add_argument("--stop-tail-seconds", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--label", default=None, help="session tag; defaults to a timestamp")
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--reset-pose-file", type=Path, default=None,
                        help="opt-in: upper-limb reset pose (JSON) the simulator applies on "
                             "every reset; off by default")
    parser.add_argument("--reset-pose-hold", action="store_true",
                        help="opt-in: hold that pose until the controller's arm setpoint moves")
    parser.add_argument("--warmstart-tokens", type=Path, default=None,
                        help="opt-in: demonstration token stream (JSON) the bridge publishes to "
                             "SONIC's action port over the settle tail of every Reset; off by "
                             "default, so a default campaign takes the canonical path")
    parser.add_argument("--warmstart-window-seconds", type=float, default=6.0,
                        help="length of that settle tail (seconds before Start); the stream "
                             "starts --settle-seconds minus this after the Reset")
    parser.add_argument("--initial-pose-handshake", action="store_true",
                        help="opt-in: the bridge publishes the VLA client's own initial-pose "
                             "command to SONIC's action port from the start of every settle "
                             "until Start; off by default, so a default campaign takes the "
                             "canonical path")
    parser.add_argument("--scene-profile", type=Path, default=None,
                        help="opt-in: scene profile Isaac loads instead of the shipped "
                             "BlockStacking profile, for a scene variant that differs in one "
                             "declared value (for example one cube's rendered material); "
                             "off by default, so a default campaign runs the shipped scene")
    parser.add_argument("--gravity-feedforward", action="store_true",
                        help="opt-in: the simulator adds PhysX's own generalized gravity "
                             "compensation torque of the body joints to the commanded torque "
                             "(experiment 04's verified term, applied before the one effort "
                             "clamp); off by default, so a default campaign runs the "
                             "unchanged PD law")
    parser.add_argument("--leave-running", action="store_true",
                        help="do not stop the launcher at the end (manual inspection)")
    return parser.parse_args(argv)


def resolve_host_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def scene_profile_record(path: Path | None) -> dict[str, Any] | None:
    """The scene profile a session ran under, so no session reads as the shipped one.

    Absent means the shipped profile: the flag is opt-in and the default command
    is unchanged.  The digest pins the file's bytes, because a scene variant is
    only a one-value change away from the canonical scene.
    """
    if path is None:
        return None
    resolved = resolve_host_path(path, data_root())
    return {
        "file": str(path),
        "host_path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest() if resolved.is_file() else None,
    }


def preflight(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    psi_dir = resolve_host_path(args.psi_checkpoint_dir, root)
    groot_dir = resolve_host_path(args.groot_checkpoint_dir, root)
    psi_host = inside_container_path(str(psi_dir), root) if str(psi_dir).startswith("/") else psi_dir
    groot_host = inside_container_path(str(groot_dir), root) if str(groot_dir).startswith("/") else groot_dir
    problems: list[str] = []
    if not (psi_host / "run_config.json").is_file():
        problems.append(f"psi run directory has no run_config.json: {psi_host}")
    if not (psi_host / "checkpoints" / f"ckpt_{args.checkpoint_step}").is_dir():
        problems.append(f"psi checkpoint ckpt_{args.checkpoint_step} is missing under {psi_host}")
    for name in ("config.json", "processor_config.json", "statistics.json"):
        if not (groot_host / name).is_file():
            problems.append(f"GR00T checkpoint is missing {name}: {groot_host}")
    if problems:
        raise DriverError("; ".join(problems))
    return {"psi_run_dir": str(psi_host), "groot_checkpoint_dir": str(groot_host)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.headless and args.gui:
        print("error: --headless cannot be combined with --gui", file=sys.stderr)
        return 2
    if args.rollouts < 1:
        print("error: --rollouts must be >= 1", file=sys.stderr)
        return 2
    if args.policy_clock_timeout_s <= 0:
        print("error: --policy-clock-timeout-s must be positive", file=sys.stderr)
        return 2
    if args.rollout_sim_seconds is not None and args.rollout_sim_seconds <= 0:
        print("error: --rollout-sim-seconds must be positive", file=sys.stderr)
        return 2
    if args.policy_clock == "wall" and args.policy_clock_file is not None:
        print("error: --policy-clock-file requires --policy-clock simulation", file=sys.stderr)
        return 2
    if args.groot_capture_max_requests <= 0:
        print("error: --groot-capture-max-requests must be positive", file=sys.stderr)
        return 2
    if args.groot_capture_dir is not None and args.model != "groot":
        print("error: --groot-capture-dir requires --model groot", file=sys.stderr)
        return 2
    if args.initial_pose_handshake and args.warmstart_tokens is not None:
        print("error: --initial-pose-handshake and --warmstart-tokens both own the settle; "
              "configure one of them", file=sys.stderr)
        return 2
    try:
        warmstart_delay = warmstart_delay_seconds(args) if args.warmstart_tokens is not None else None
    except DriverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    root = data_root()
    try:
        identity = preflight(args, root)
    except DriverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tag = args.label or time.strftime("%Y%m%d-%H%M%S")
    out_dir = resolve_host_path(args.out_dir, root) if args.out_dir else root / "outputs" / "blockstacking-debug" / tag
    raw_dir = out_dir / "raw"
    logs_dir = out_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    policy_clock_host = None
    run_clock_host = None
    if args.policy_clock == "simulation":
        policy_clock_host = resolve_host_path(
            args.policy_clock_file or (raw_dir / "isaac-policy-clock.txt"), root
        )
        policy_clock_host.parent.mkdir(parents=True, exist_ok=True)
        policy_clock_host.unlink(missing_ok=True)
        run_clock_host = policy_clock_host
    elif args.rollout_sim_seconds is not None:
        run_clock_host = raw_dir / "isaac-run-clock.txt"
        run_clock_host.unlink(missing_ok=True)

    paths = {
        "telemetry": container_path(raw_dir / "telemetry", root),
        "video_raw": container_path(raw_dir / "video.raw.mp4", root),
        "video_timestamps": container_path(raw_dir / "video.timestamps.jsonl", root),
        "isaac_samples": container_path(raw_dir / "isaac.samples.jsonl", root),
        "isaac_tracking": container_path(raw_dir / "isaac.tracking.parquet", root),
        "isaac_metrics": container_path(raw_dir / "isaac.metrics.json", root),
    }
    if policy_clock_host is not None:
        paths["policy_clock"] = container_path(policy_clock_host, root)
    if run_clock_host is not None:
        paths["run_clock"] = container_path(run_clock_host, root)
    if args.groot_capture_dir is not None:
        capture_host = resolve_host_path(args.groot_capture_dir, root)
        capture_host.mkdir(parents=True, exist_ok=True)
        paths["groot_capture"] = container_path(capture_host, root)
    command = build_launcher_command(args, paths)
    log_path = out_dir / "launcher.log"
    model_id = args.model_id or ("fine-tuned" if args.model == "psi" else "groot")
    session: dict[str, Any] = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "tag": tag,
        "task": "BlockStacking",
        "requested_model": args.model,
        "requested_model_id": model_id,
        "checkpoints": identity,
        "checkpoint_step": args.checkpoint_step,
        "rollout_seconds": args.rollout_seconds,
        "rollout_sim_seconds": args.rollout_sim_seconds,
        "settle_seconds": args.settle_seconds,
        "policy_clock": {
            "mode": args.policy_clock,
            "file": paths.get("policy_clock"),
            "timeout_s": args.policy_clock_timeout_s,
        },
        "groot": {
            "capture_dir": paths.get("groot_capture"),
            "capture_max_requests": args.groot_capture_max_requests,
            "left_hand_contract": args.groot_left_hand_contract,
        },
        "psi0": {"rtc": not args.psi0_rtc_off},
        "reset_pose": {
            "file": None if args.reset_pose_file is None else str(args.reset_pose_file),
            "hold": bool(args.reset_pose_hold),
        },
        "warmstart": {
            "tokens": None if args.warmstart_tokens is None else str(args.warmstart_tokens),
            "window_seconds": (
                None if args.warmstart_tokens is None else float(args.warmstart_window_seconds)
            ),
            "delay_after_reset_seconds": warmstart_delay,
        },
        "initial_pose": {
            "handshake": bool(args.initial_pose_handshake),
        },
        "gravity_feedforward": {
            "enabled": bool(args.gravity_feedforward),
        },
        "scene_profile": scene_profile_record(args.scene_profile),
        "rollouts_requested": args.rollouts,
        "command": command,
        "host_paths": {
            "out_dir": str(out_dir), "raw_dir": str(raw_dir), "logs_dir": str(logs_dir),
            "launcher_log": str(log_path), **{key: str(raw_dir / Path(value).name) for key, value in paths.items()},
        },
        "container_paths": paths,
        "started_wall_ns": time.time_ns(),
        "host": {"hostname": socket.gethostname(), "cwd": str(Path.cwd())},
        "rollouts": [],
    }

    print(f"[rollout] session dir : {out_dir}")
    print(f"[rollout] command     : {' '.join(command)}")
    with log_path.open("wb") as log:
        launcher = subprocess.Popen(
            command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    ui = Ui(args.ui_port)
    exit_code = 0
    try:
        meta = wait_for_ui(ui, launcher, log_path)
        session["prompt"] = meta.get("prompt")
        session["ui_ready_wall_ns"] = time.time_ns()
        print(f"[rollout] UI ready (prompt {meta.get('prompt')!r})")
        if model_id != "fine-tuned":
            print(f"[rollout] selecting model {model_id!r}")
            ui.select_model(model_id)
        session["model_status"] = wait_for_model(ui, model_id)
        print(f"[rollout] serving: {model_id}")

        for index in range(1, args.rollouts + 1):
            rollout: dict[str, Any] = {"index": index, "start_wall_ns": time.time_ns()}
            print(f"[rollout] {index}/{args.rollouts}: reset")
            rollout["reset_response"] = ui.reset()
            rollout["reset_wall_ns"] = time.time_ns()
            time.sleep(args.settle_seconds)
            rollout["pre_start_status"] = ui.status()
            start_wall_ns = time.time_ns()
            print(f"[rollout] {index}/{args.rollouts}: start")
            rollout["start_response"] = ui.start(session["prompt"])
            rollout["start_wall_ns"] = start_wall_ns
            try:
                rollout["running_status"] = wait_for_running(ui)
            except DriverError as exc:
                rollout["error"] = str(exc)
                print(f"[rollout] {index}: {exc}", file=sys.stderr)
            if args.rollout_sim_seconds is None:
                time.sleep(args.rollout_seconds)
                time.sleep(args.stop_tail_seconds)
            else:
                assert run_clock_host is not None
                rollout["sim_horizon"] = wait_for_sim_horizon(
                    run_clock_host,
                    args.rollout_sim_seconds,
                    timeout_s=max(args.rollout_seconds, args.rollout_sim_seconds * 5.0) + 60.0,
                )
            rollout["stop_response"] = ui.stop()
            rollout["stop_wall_ns"] = time.time_ns()
            rollout["final_status"] = ui.status()
            rollout["end_wall_ns"] = time.time_ns()
            session["rollouts"].append(rollout)
            print(f"[rollout] {index}/{args.rollouts}: stopped "
                  f"({(rollout['end_wall_ns'] - rollout['start_wall_ns']) / 1e9:.1f}s wall)")
    except DriverError as exc:
        session["error"] = str(exc)
        print(f"[rollout] error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        session["finished_wall_ns"] = time.time_ns()
        if args.leave_running:
            print("[rollout] launcher left running (--leave-running)")
        else:
            # End the simulation on its own terms first: a killed Isaac leaves a
            # truncated video and no summary or tracking series behind.
            session["isaac_shutdown"] = request_isaac_shutdown()
            print(f"[rollout] isaac shutdown request: {session['isaac_shutdown']!r}")
            session["finalized"] = wait_for_artifacts(
                [raw_dir / "isaac.metrics.json", raw_dir / "isaac.tracking.parquet"],
                ISAAC_FINALIZE_TIMEOUT_S,
            )
            print(f"[rollout] artifacts finalized: {session['finalized']}")
            session["launcher_exit_code"] = terminate_launcher(launcher, log_path, grace_s=30.0)
        log_dir = find_launcher_log_dir(log_path)
        if log_dir:
            session["launcher_log_dir"] = log_dir
            source = inside_container_path(log_dir, root) if log_dir.startswith("/") else Path(log_dir)
            copied: dict[str, str] = {}
            for log_file in sorted(Path(source).glob("*.log")) if source.is_dir() else []:
                target = logs_dir / log_file.name
                try:
                    shutil.copy2(log_file, target)
                    copied[log_file.name] = str(target)
                except OSError as exc:
                    print(f"[rollout] cannot copy {log_file}: {exc}", file=sys.stderr)
            session["copied_logs"] = copied
        (out_dir / "session.json").write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[rollout] session.json written: {out_dir / 'session.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
