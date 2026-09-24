#!/usr/bin/env python3
"""Serve the real-robot VLA UI. Launching the page never starts motor control."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-host", required=True, help="Unitree Tailscale IPv4 or wired IPv4")
    parser.add_argument("--camera-port", type=int, default=8558)
    parser.add_argument("--psi-run-dir", type=Path)
    parser.add_argument("--psi-step", type=int)
    parser.add_argument("--psi-neck-policy", choices=("error", "discard"), default="error")
    parser.add_argument("--groot-checkpoint", type=Path)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8015)
    args = parser.parse_args()
    if bool(args.psi_run_dir) != (args.psi_step is not None):
        parser.error("--psi-run-dir and --psi-step must be provided together")
    if args.psi_run_dir is None and args.groot_checkpoint is None:
        parser.error("provide at least one model checkpoint")

    from humanoid_lab.psi0_bridge.action_router import ActionRouter
    from humanoid_lab.psi0_bridge.groot_backend import GrootProcessGroup
    from humanoid_lab.psi0_bridge.monitor import Monitor
    from humanoid_lab.psi0_bridge.policy_clock import PolicyClock
    from humanoid_lab.psi0_bridge.policy_server import PolicyServerProcess
    from humanoid_lab.psi0_bridge.session import Session, SessionConfig
    from humanoid_lab.robot_runtime.app import create_app
    from humanoid_lab.robot_runtime.camera_bridge import GrootCameraBridge
    from humanoid_lab.robot_runtime.controller import Model, RobotController, load_tasks
    from humanoid_lab.robot_runtime.sonic_status import SonicStatusMonitor

    source_root = Path(__file__).resolve().parents[1]
    tasks = load_tasks(source_root / "configs/datasets/psi0/unitree_dex3_sonic_v1.yaml")
    models = []
    if args.psi_run_dir is not None:
        checkpoint = args.psi_run_dir / "checkpoints" / f"ckpt_{args.psi_step}"
        if not checkpoint.is_dir():
            parser.error(f"Psi0 checkpoint missing: {checkpoint}")
        models.append(Model("psi0", "psi", args.psi_run_dir, args.psi_step))
    if args.groot_checkpoint is not None:
        from humanoid_lab.psi0_bridge.groot_backend import validate_groot_checkpoint
        validate_groot_checkpoint(args.groot_checkpoint)
        models.append(Model("groot", "groot", args.groot_checkpoint))

    monitor = Monitor(state_endpoint=f"tcp://{args.robot_host}:5557",
                      camera_endpoint=f"http://{args.robot_host}:{args.camera_port}")
    sonic = SonicStatusMonitor(f"tcp://{args.robot_host}:5561")
    router = ActionRouter(public_endpoint="tcp://*:5556",
                          groot_endpoint="tcp://127.0.0.1:5560",
                          validate_policy_packets=True)
    psi_session = Session(SessionConfig(
        ws_url="ws://127.0.0.1:8014/ws",
        state_endpoint=f"tcp://{args.robot_host}:5557",
        camera_endpoint=f"http://{args.robot_host}:{args.camera_port}",
        action_endpoint="tcp://*:5556",
        instruction=next(iter(tasks.values())),
        neck_policy=args.psi_neck_policy,
    ), monitor=monitor, publisher=router.psi_sink, policy_clock=PolicyClock("wall"))
    psi_server = PolicyServerProcess(port=8014,
                                     log_path=Path("/outputs/vla-robot/psi0-server.log"),
                                     policy_clock="wall")
    groot = GrootProcessGroup(
        args.groot_checkpoint or Path("/nonexistent"),
        prompt=next(iter(tasks.values())),
        log_dir=Path("/outputs/vla-robot/groot"),
        policy_clock="wall",
        state_zmq_host=args.robot_host,
    )
    controller = RobotController(
        monitor=monitor, sonic=sonic, router=router,
        camera_bridge=GrootCameraBridge(monitor),
        psi_session=psi_session, psi_server=psi_server, groot=groot,
        models=models, tasks=tasks,
    )
    import uvicorn
    uvicorn.run(create_app(controller), host=args.ui_host, port=args.ui_port,
                log_level="info")


if __name__ == "__main__":
    main()
