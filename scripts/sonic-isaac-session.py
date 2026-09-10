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
    ContractError,
)
from sonic_isaac_ipc import DEFAULT_IPC_PORT  # noqa: E402
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
SONIC_DEPLOY = SONIC_ROOT / "gear_sonic_deploy" / "target" / "release" / "g1_deploy_onnx_ref"
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


def deploy_argv(*, robot: str, input_mode: str, interface: str = SIM_DDS_INTERFACE,
                mode: str = "sim") -> list[str]:
    """The upstream C++ deploy command line for the simulation path."""
    config = _read_deploy_config()
    models = Path(config.get("models_dir", "/data/models/sonic/sonic_v1_1"))
    planner = Path(config.get("planner", "/data/models/sonic/planner_sonic.onnx"))
    reference = Path(config.get("reference_dir", str(SONIC_ROOT / "gear_sonic_deploy" / "reference" / "example")))
    argv = [
        str(SONIC_DEPLOY),
        interface,
        str(models / "model_decoder.onnx"),
        str(reference),
        "--obs-config", str(models / "observation_config.yaml"),
        "--encoder-file", str(models / "model_encoder.onnx"),
        "--planner-file", str(planner),
        "--input-type", "f310_bridge" if input_mode == "f310" else config.get("keyboard_input_type", "manager"),
        mode,
    ]
    if input_mode == "f310":
        argv += ["--f310-bridge-port", str(config.get("f310_bridge_port", 49051))]
    return argv


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
            mode=args.sonic_mode,
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
    if args.auto_play:
        isaac_argv.append("--auto-play")
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

    if args.skip_sonic:
        children.append(_spawn("bridge", bridge_argv, paths.log_dir / f"{stamp}-bridge.log"))
    else:
        children.append(_spawn("bridge", bridge_argv, paths.log_dir / f"{stamp}-bridge.log"))
        sound_argv = deploy_argv(robot=args.robot, input_mode=args.input)
        if not SONIC_DEPLOY.is_file():
            print(json.dumps({"started": False, "refused": ["sonic_deploy_missing"]}), file=sys.stderr)
        else:
            children.append(_spawn("sonic", sound_argv, paths.log_dir / f"{stamp}-sonic.log"))

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


def _spawn(role: str, argv: list[str], log_path: Path) -> ChildProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    process = subprocess.Popen(  # noqa: S603 - argv is constructed here, not user shell text
        argv,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(ROOT),
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
