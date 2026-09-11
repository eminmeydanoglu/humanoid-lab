"""Command-line entry point for the clean Isaac G1 simulator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Sequence

from .contracts import RunProfile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--test", choices=("passive-fall",))
    parser.add_argument(
        "--head-camera-window",
        action="store_true",
        help="open a second GPU viewport for the head camera (costs render throughput)",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True
    # AppLauncher owns --device; only an explicit flag overrides the profile default.
    raw = list(sys.argv[1:] if argv is None else argv)
    args.device_explicit = any(item == "--device" or item.startswith("--device=") for item in raw)
    if args.duration is None:
        args.duration = 12.0 if args.test or args.headless else 300.0
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # AppLauncher pops its own flags (including device_explicit) from vars(args).
    device_override = args.device if args.device_explicit else None
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    exit_code = 2
    try:
        from .service import SimulatorService

        # The profile owns the physics device; --device only wins when passed explicitly.
        profile = RunProfile.load(args.profile).with_device(device_override)
        service = SimulatorService(
            profile,
            simulation_app,
            duration=args.duration,
            show_ui=not args.headless,
            show_head_camera=args.head_camera_window,
            test_mode=args.test,
        )
        summary = service.run()
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        exit_code = 0 if summary["result"] in {"PASS", "COMPLETED"} else 1
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(json.dumps({"event": "isaac_g1_failed", "reason": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        closer = threading.Thread(
            target=lambda: simulation_app.close(wait_for_replicator=False), daemon=True
        )
        closer.start()
        # Kit 5.1 can deadlock its GUI teardown after the window has stopped
        # pumping events. Keep the visible unresponsive window bounded; the
        # process wrapper also guarantees that no container child survives.
        closer.join(timeout=2.0 if not args.headless else 20.0)
        closed = not closer.is_alive()
        print(
            json.dumps(
                {"event": "isaac_g1_app_close", "closed": closed, "forced_exit": not closed}
            ),
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
