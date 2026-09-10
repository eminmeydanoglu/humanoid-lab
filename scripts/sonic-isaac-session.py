#!/usr/bin/env python3
"""``dev.sh sonic-isaac`` session owner: start, status and stop one SONIC run.

Runs inside the development container.  ``start`` is the single session owner:
it records the PIDs of the Isaac runner, the DDS bridge, the upstream SONIC
deploy and the input adapter, and ``stop`` signals only those recorded PIDs.

The launch order is the plan's lifecycle: Isaac comes up paused and publishes
its initial state, the DDS bridge joins the loopback domain, then the SONIC
deploy is allowed to start.  The timeline is never auto-played unless
``--auto-play`` is passed for an automated gate.
"""

from __future__ import annotations

import argparse
import json
import os
import pty
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from sonic_isaac_contract import (  # noqa: E402
    SIM_DDS_DOMAIN_ID,
    SIM_DDS_INTERFACE,
    SIM_ONLY_DEPLOY_FLAG,
    ContractError,
)
from sonic_isaac_ipc import DEFAULT_IPC_PORT  # noqa: E402
from sonic_isaac_keyboard import (  # noqa: E402
    STOP_KEY,
    KeyScheduler,
    drive_schedule,
    standing_schedule,
)
from sonic_isaac_session import (  # noqa: E402
    AGENT_STATE_READY_PENDING_USER_DRIVE,
    DEFAULT_RUNTIME_DIR,
    ChildProcess,
    SessionPaths,
    SessionState,
    build_status,
    derive_controller,
    derive_simulator,
    evaluate_launch,
    load_state,
    plan_children,
    save_state,
    stop_plan,
)

SONIC_ROOT = Path("/opt/src/sonic")
DEFAULT_DEPLOY_BINARY = Path("/data/models/sonic-deploy/g1_deploy_onnx_ref")
SONIC_SIM_PYTHON = Path("/opt/venvs/sonic-sim/bin/python")
DEPLOY_CONFIG = ROOT / "configs" / "sonic_isaac_deploy.json"

USAGE_SUMMARY = """\
1. GUI paused acildi.
2. Start ile SONIC'i baslat.
3. Isaac Timeline Play'e bas.
4. LB+RB ile Planner'a gec.
5. Sol stick ile sur; birakinca Idle beklenir.
6. B planner stop, Back tam SONIC stop."""


# --------------------------------------------------------------------------- #
# Host facts
# --------------------------------------------------------------------------- #


def host_interfaces() -> list[str]:
    """Interface names, from sysfs so no netlink dependency is needed."""
    try:
        return sorted(path.name for path in Path("/sys/class/net").iterdir())
    except OSError:
        return []


def _read_deploy_config() -> dict:
    if DEPLOY_CONFIG.is_file():
        return json.loads(DEPLOY_CONFIG.read_text())
    return {}


def deploy_argv(*, robot: str, input_mode: str, interface: str = SIM_DDS_INTERFACE) -> list[str]:
    """The upstream C++ deploy command line for the simulation path.

    Positional arguments are ``<interface> <policy_decoder.onnx> <motion_data>``.
    ``--disable-crc-check`` is what makes this a simulation command: the real
    robot's lowcmd stream carries CRCs the simulator does not produce.
    """
    config = _read_deploy_config()
    models = Path(config.get("models_dir", "/data/models/sonic/sonic_v1_1"))
    planner = Path(config.get("planner", "/data/models/sonic/planner_sonic.onnx"))
    reference = Path(
        config.get("reference_dir", str(SONIC_ROOT / "gear_sonic_deploy" / "reference" / "example"))
    )
    binary = Path(config.get("binary", "/data/models/sonic-deploy/g1_deploy_onnx_ref"))
    keyboard_type = config.get("keyboard_input_type", "keyboard")
    f310_type = config.get("f310_input_type", "f310_bridge")
    return [
        str(binary),
        str(interface),
        str(models / "model_decoder.onnx"),
        str(reference),
        "--obs-config", str(models / "observation_config.yaml"),
        "--encoder-file", str(models / "model_encoder.onnx"),
        "--planner-file", str(planner),
        "--input-type", f310_type if input_mode == "f310" else keyboard_type,
        config.get("sim_marker", SIM_ONLY_DEPLOY_FLAG),
    ]


def deploy_library_path() -> str:
    """Extra shared libraries the deploy needs that the dev image lacks."""
    config = _read_deploy_config()
    lib_dir = Path(config.get("lib_dir", "/data/models/sonic-deploy/lib"))
    return str(lib_dir)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def command_status(args: argparse.Namespace) -> int:
    paths = SessionPaths.under(args.runtime_dir)
    record = load_state(paths)
    if record is None:
        record = SessionState(robot_profile="g1-29dof", input="keyboard")
        record.simulator = "stopped"
        record.controller = "absent"
    else:
        alive = _any_alive(record)
        record.simulator = derive_simulator(
            running=alive,
            playing=record.simulator == "playing",
            paused=record.simulator in {"paused", "ready"},
        )
        record.controller = derive_controller(
            lowcmd_age_ms=record.last_lowcmd_age_ms if alive else None,
            sonic_running=bool(record.child_pids("sonic")) and alive,
        )
    print(json.dumps(build_status(record), indent=2, sort_keys=True))
    return 0


def _any_alive(record: SessionState) -> bool:
    for pid in record.child_pids():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            return True
        else:
            return True
    return False


def command_stop(args: argparse.Namespace) -> int:
    paths = SessionPaths.under(args.runtime_dir)
    record = load_state(paths)
    if record is None:
        print(json.dumps({"stopped": False, "reason": "no recorded session"}))
        return 0

    signalled: list[int] = []
    for pid in stop_plan(record):
        try:
            os.kill(pid, signal.SIGTERM)
            signalled.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            record.notes.append(f"not permitted to signal {pid}")

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and _any_alive(record):
        time.sleep(0.25)

    for child in record.children:
        try:
            os.kill(child.pid, 0)
        except ProcessLookupError:
            continue
        record.notes.append(f"still alive after TERM: {child.role}")

    for path in (paths.state_file, paths.ready_file):
        path.unlink(missing_ok=True)
    print(
        json.dumps(
            {"stopped": True, "signalled": signalled, "notes": record.notes},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_start(args: argparse.Namespace) -> int:
    paths = SessionPaths.under(args.runtime_dir)
    paths.evidence_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    plan_children(args.robot, args.input)

    # Keyboard mode feeds the upstream SONIC input handler directly, so it must
    # not silently continue against a missing or nested PTY.
    if args.input == "keyboard" and not sys.stdin.isatty():
        print(json.dumps({"started": False, "refused": ["keyboard_requires_tty"]}, indent=2))
        return 2

    existing = load_state(paths)
    if args.force:
        existing = None

    dds_uri = os.environ.get("CYCLONEDDS_URI", "")
    config_text = None
    if dds_uri.startswith("file://"):
        candidate = Path(dds_uri[len("file://") :])
        if candidate.is_file():
            config_text = candidate.read_text()

    reasons = evaluate_launch(
        env=os.environ,
        dds_config_text=config_text,
        domain_id=int(os.environ.get("ROS_DOMAIN_ID", "-1")),
        sonic_mode=args.sonic_mode,
        requested_interface=args.sonic_interface,
        argv=deploy_argv(
            robot=args.robot,
            input_mode=args.input,
            interface=args.sonic_interface,
        ),
        physical_interfaces=host_interfaces(),
        existing_session=existing,
        own_pid=os.getpid(),
    )
    if reasons:
        print(json.dumps({"started": False, "refused": list(reasons)}, indent=2))
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    evidence = paths.evidence_dir / f"{stamp}-{args.robot}-{args.input}.json"
    isaac_argv = [
        sys.executable,
        str(ROOT / "scripts" / "run-sonic-isaac.py"),
        "--robot", args.robot,
        "--input", args.input,
        "--ipc-port", str(args.ipc_port),
        "--evidence-path", str(evidence),
        "--ready-file", str(paths.ready_file),
        "--duration", str(args.duration),
        "--warmup", str(args.warmup),
        "--paused-hold", str(args.paused_hold),
    ]
    play_trigger = None
    if args.auto_play:
        isaac_argv.append("--auto-play")
    if args.auto_keys != "none":
        # Physics must not start until the controller is armed, so the runner
        # waits for a trigger file instead of playing on a timer.
        play_trigger = paths.runtime_dir / "play.trigger"
        play_trigger.unlink(missing_ok=True)
        isaac_argv += ["--play-trigger", str(play_trigger)]
    if args.fall_test:
        isaac_argv.append("--fall-test")
    if args.headless:
        isaac_argv.append("--headless")
    if args.video_path:
        isaac_argv += ["--video-path", str(args.video_path)]

    bridge_argv = [
        str(SONIC_SIM_PYTHON),
        str(ROOT / "scripts" / "sonic-isaac-dds-bridge.py"),
        "--domain-id", str(SIM_DDS_DOMAIN_ID),
        "--interface", SIM_DDS_INTERFACE,
        "--port", str(args.ipc_port),
        "--report-path", str(paths.evidence_dir / f"{stamp}-bridge.json"),
    ]

    children: list[ChildProcess] = []
    children.append(_spawn("isaac", isaac_argv, paths.log_dir / f"{stamp}-isaac.log"))
    if not _wait_for_file(paths.ready_file, args.ready_timeout):
        _terminate_all(children)
        print(json.dumps({"started": False, "refused": ["isaac_never_ready"]}, indent=2))
        return 3

    deploy_pty: PtyChild | None = None
    if args.skip_sonic:
        children.append(_spawn("bridge", bridge_argv, paths.log_dir / f"{stamp}-bridge.log"))
    else:
        children.append(_spawn("bridge", bridge_argv, paths.log_dir / f"{stamp}-bridge.log"))
        sonic_argv = deploy_argv(
            robot=args.robot, input_mode=args.input, interface=SIM_DDS_INTERFACE
        )
        if not Path(sonic_argv[0]).is_file():
            _terminate_all(children)
            print(json.dumps({"started": False, "refused": ["sonic_deploy_missing"],
                              "expected": sonic_argv[0]}, indent=2))
            return 4
        # The dev image lacks ONNX Runtime and TensorRT; the deploy's staged
        # libraries are added to its environment only, not to the runner's.
        sonic_env = dict(os.environ)
        sonic_env["LD_LIBRARY_PATH"] = ":".join(
            [deploy_library_path(), sonic_env.get("LD_LIBRARY_PATH", "")]
        ).strip(":")
        if args.auto_keys == "none":
            children.append(
                _spawn("sonic", sonic_argv, paths.log_dir / f"{stamp}-sonic.log", env=sonic_env)
            )
            deploy_pty = None
        else:
            deploy_pty = PtyChild(
                sonic_argv, paths.log_dir / f"{stamp}-sonic.log", env=sonic_env
            )
            children.append(
                ChildProcess(
                    role="sonic",
                    pid=deploy_pty.pid,
                    argv=tuple(sonic_argv),
                    started_at=time.time(),
                )
            )

    if args.auto_keys != "none" and deploy_pty is not None:
        return _drive_auto_keys(
            args=args,
            paths=paths,
            record_children=children,
            deploy=deploy_pty,
            play_trigger=play_trigger,
            evidence=evidence,
            stamp=stamp,
        )

    record = SessionState(
        robot_profile=args.robot,
        input=args.input,
        domain_id=SIM_DDS_DOMAIN_ID,
        interface=SIM_DDS_INTERFACE,
        simulator="paused",
        controller="absent",
        owner_pid=os.getpid(),
        children=children,
        agent_state=AGENT_STATE_READY_PENDING_USER_DRIVE,
    )
    save_state(paths, record)
    print(json.dumps({"started": True, "evidence": str(evidence), "children": len(children)}, indent=2))
    print(USAGE_SUMMARY)
    if args.auto_play:
        print("(auto-play enabled: an automated gate, not a manual session)")
    return 0


class PtyChild:
    """Run a child attached to a PTY so its keyboard interface sees a TTY.

    The deploy refuses to behave as a keyboard consumer without one, and the
    automated gates need to inject the keystrokes an operator would send.
    """

    def __init__(self, argv: list[str], log_path: Path, env: dict | None = None) -> None:
        self.argv = list(argv)
        self.log_path = log_path
        self.master, slave = pty.openpty()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("wb")
        self.process = subprocess.Popen(  # noqa: S603
            self.argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            cwd=str(ROOT),
            env=env,
        )
        os.close(slave)
        os.set_blocking(self.master, False)
        self._buffer = b""
        self.lines: list[str] = []

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def poll(self) -> int | None:
        return self.process.poll()

    def drain(self) -> list[str]:
        """Read whatever the child has written; returns any new complete lines."""
        while True:
            ready, _, _ = select.select([self.master], [], [], 0)
            if not ready:
                break
            try:
                chunk = os.read(self.master, 65536)
            except (BlockingIOError, OSError):
                break
            if not chunk:
                break
            self._log.write(chunk)
            self._log.flush()
            self._buffer += chunk
        fresh: list[str] = []
        while b"\n" in self._buffer:
            raw, self._buffer = self._buffer.split(b"\n", 1)
            text = raw.decode("utf-8", "replace").rstrip()
            self.lines.append(text)
            fresh.append(text)
        return fresh

    def send(self, keys: str) -> None:
        os.write(self.master, keys.encode())

    def terminate(self, timeout_s: float = 10.0) -> None:
        self.process.terminate()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.process.poll() is None:
            time.sleep(0.2)
        if self.process.poll() is None:
            self.process.kill()
        try:
            os.close(self.master)
        except OSError:
            pass
        self._log.close()


def _spawn(role: str, argv: list[str], log_path: Path, env: dict | None = None) -> ChildProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    process = subprocess.Popen(  # noqa: S603 - argv is constructed here, not user shell text
        argv,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(ROOT),
        env=env,
    )
    return ChildProcess(role=role, pid=process.pid, argv=tuple(argv), started_at=time.time())


def _wait_for_file(path: Path, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return True
        time.sleep(0.5)
    return False


def _terminate_all(children: list[ChildProcess]) -> None:
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue


#: Lines the deploy prints once it is up and simply waiting for robot state.
DEPLOY_WAITING_MARKERS = (
    "LowState is not available",
    "waiting for robot",
)
#: Lines that mean the deploy is past loading and safe to drive.
DEPLOY_READY_MARKERS = (
    "control",
    "Planner",
    "planner mode",
    "Standing",
    "F1",
)


def _wait_deploy_ready(deploy: "PtyChild", *, timeout_s: float, quiet_s: float) -> dict:
    """Wait until the deploy has its initial state and can accept the arm key.

    The deploy loads its models before it subscribes to ``rt/lowstate``. Once
    state flows it stops reporting that it is waiting, which is the signal that
    a keystroke will be consumed.
    """
    started = time.monotonic()
    last_waiting: float | None = None
    seen_waiting = False
    while (time.monotonic() - started) < timeout_s:
        fresh = deploy.drain()
        now = time.monotonic()
        for line in fresh:
            if any(marker in line for marker in DEPLOY_WAITING_MARKERS):
                seen_waiting = True
                last_waiting = now
            if any(marker in line for marker in DEPLOY_READY_MARKERS):
                return {"ready": True, "reason": "ready_marker", "line": line,
                        "waited_s": now - started}
        if seen_waiting and last_waiting is not None and (now - last_waiting) >= quiet_s:
            return {"ready": True, "reason": "state_flowing", "waited_s": now - started}
        if deploy.poll() is not None:
            return {"ready": False, "reason": "deploy_exited", "waited_s": now - started}
        time.sleep(0.05)
    return {"ready": False, "reason": "timeout", "waited_s": time.monotonic() - started}


def _child_exit_code(pid: int | None) -> int | None:
    """Return the exit code if the pid is gone, else None."""
    if not pid:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return 0
    except PermissionError:
        return None
    return None


def _drive_auto_keys(*, args, paths, record_children, deploy, play_trigger,
                     evidence: Path, stamp: str) -> int:
    """Replay the operator keystrokes and let physics run only once armed."""
    events = drive_schedule() if args.auto_keys == "drive" else standing_schedule()
    scheduler = KeyScheduler(events)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)

    record = SessionState(
        robot_profile=args.robot,
        input=args.input,
        domain_id=SIM_DDS_DOMAIN_ID,
        interface=SIM_DDS_INTERFACE,
        simulator="paused",
        controller="absent",
        owner_pid=os.getpid(),
        children=record_children,
        agent_state=AGENT_STATE_READY_PENDING_USER_DRIVE,
    )
    save_state(paths, record)

    readiness = _wait_deploy_ready(
        deploy, timeout_s=args.deploy_ready_timeout, quiet_s=args.deploy_quiet_s
    )
    print(json.dumps({"event": "deploy_readiness", **readiness}), flush=True)
    if not readiness.get("ready"):
        for child in record_children:
            _terminate_child(child)
        deploy.terminate()
        print(json.dumps({"started": False, "refused": ["deploy_not_ready"],
                          "detail": readiness}, indent=2))
        return 5

    isaac_pid = record_children[0].pid if record_children else None
    started = time.monotonic()
    armed_at: float | None = None
    triggered = False
    applied: list[dict] = []
    outcome = "completed"

    while True:
        now = time.monotonic()
        elapsed = now - started
        for line in deploy.drain():
            pass

        for event in scheduler.due(elapsed):
            deploy.send(event.keys)
            applied.append({"label": event.label, "at_s": round(event.at_s, 3)})
            if armed_at is None:
                armed_at = now
            print(json.dumps({"event": "key_sent", "label": event.label,
                              "keys": event.keys.replace("\n", "\\n"),
                              "at_s": round(event.at_s, 3)}), flush=True)

        if (
            play_trigger is not None
            and not triggered
            and armed_at is not None
            and (now - armed_at) >= args.arm_settle
        ):
            play_trigger.write_text("play\n")
            triggered = True
            print(json.dumps({"event": "play_triggered",
                              "after_arm_s": round(now - armed_at, 3)}), flush=True)

        if _child_exit_code(isaac_pid) is not None:
            outcome = "runner_exited"
            break
        if deploy.poll() is not None:
            outcome = "deploy_exited"
            break
        if elapsed > args.max_run_s:
            outcome = "max_run_time"
            break
        time.sleep(0.02)

    # A stop key then TERM, so SONIC disarms before the process goes away.
    try:
        deploy.send(STOP_KEY)
    except OSError:
        pass
    time.sleep(0.5)
    deploy.drain()
    for child in record_children:
        _terminate_child(child)
    deploy.terminate()
    play_trigger.unlink(missing_ok=True)
    paths.state_file.unlink(missing_ok=True)

    summary = {
        "started": True,
        "mode": args.auto_keys,
        "robot_profile": args.robot,
        "outcome": outcome,
        "readiness": readiness,
        "keys_applied": applied,
        "play_triggered": triggered,
        "evidence": str(evidence),
        "deploy_exit": deploy.poll(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if outcome == "runner_exited" else 6


def _terminate_child(child: ChildProcess) -> None:
    try:
        os.kill(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return


def command_accept(args: argparse.Namespace) -> int:
    """Gate B for one profile: no controller, Play must make the robot fall."""
    paths = SessionPaths.under(args.runtime_dir)
    paths.evidence_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    evidence = paths.evidence_dir / f"{stamp}-{args.robot}-fall.json"
    video = paths.evidence_dir / f"{stamp}-{args.robot}-fall.mp4"
    argv = [
        sys.executable,
        str(ROOT / "scripts" / "run-sonic-isaac.py"),
        "--robot", args.robot,
        "--input", "acceptance",
        "--evidence-path", str(evidence),
        "--video-path", str(video),
        "--duration", str(args.duration),
        "--paused-hold", str(args.paused_hold),
        "--fall-test",
        "--auto-play",
    ]
    if args.headless:
        argv.append("--headless")
    print(json.dumps({"accept": "fall", "evidence": str(evidence), "video": str(video)}))
    completed = subprocess.run(argv, cwd=str(ROOT))  # noqa: S603
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--ipc-port", type=int, default=DEFAULT_IPC_PORT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--robot", choices=("g1-29dof", "g1-inspire"), required=True)
    start.add_argument("--input", choices=("keyboard", "f310", "acceptance"), default="keyboard")
    start.add_argument("--duration", type=float, default=60.0)
    start.add_argument("--warmup", type=float, default=5.0)
    start.add_argument("--paused-hold", type=float, default=3.0)
    start.add_argument("--ready-timeout", type=float, default=300.0)
    start.add_argument("--auto-play", action="store_true")
    start.add_argument("--fall-test", action="store_true")
    start.add_argument("--headless", action="store_true")
    start.add_argument("--video-path", type=Path)
    start.add_argument("--force", action="store_true")
    start.add_argument("--skip-sonic", action="store_true")
    # Automated gates: replay the operator's keystrokes into the deploy's PTY.
    start.add_argument(
        "--auto-keys", choices=("none", "standing", "drive"), default="none",
        help="replay a keystroke schedule instead of expecting a human operator",
    )
    start.add_argument("--arm-settle", type=float, default=6.0,
                       help="seconds between arming SONIC and allowing physics to run")
    start.add_argument("--deploy-ready-timeout", type=float, default=600.0)
    start.add_argument("--deploy-quiet-s", type=float, default=4.0)
    start.add_argument("--max-run-s", type=float, default=900.0)
    # Simulation-only by construction: any other value is refused by the
    # launcher gates rather than silently connecting to a real robot.
    start.add_argument("--sonic-interface", default=SIM_DDS_INTERFACE)
    start.add_argument("--sonic-mode", default="sim")
    start.set_defaults(handler=command_start)

    status = subparsers.add_parser("status")
    status.set_defaults(handler=command_status)

    stop = subparsers.add_parser("stop")
    stop.set_defaults(handler=command_stop)

    accept = subparsers.add_parser("accept")
    accept.add_argument("--robot", choices=("g1-29dof", "g1-inspire"), required=True)
    accept.add_argument("--duration", type=float, default=6.0)
    accept.add_argument("--paused-hold", type=float, default=3.0)
    accept.add_argument("--headless", action="store_true")
    accept.set_defaults(handler=command_accept)

    args = parser.parse_args()
    try:
        return args.handler(args)
    except ContractError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
