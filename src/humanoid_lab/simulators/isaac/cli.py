"""Command-line entry point for the clean Isaac G1 simulator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Sequence

from .contracts import RunProfile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("HUMANOID_OUTPUT_ROOT", "/outputs")),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--capture-every", type=int, default=4)
    parser.add_argument("--test", choices=("passive-fall",))
    parser.add_argument("--no-record", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True
    if args.duration is None:
        args.duration = 12.0 if args.test or args.headless else 300.0
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    exit_code = 2
    try:
        from .service import SimulatorService

        profile = RunProfile.load(args.profile)
        run_id = args.run_id or time.strftime("isaac-g1-%Y%m%d-%H%M%S")
        service = SimulatorService(
            profile,
            simulation_app,
            run_id=run_id,
            output_root=args.output_root,
            duration=args.duration,
            record=not args.no_record,
            show_ui=not args.headless,
            test_mode=args.test,
            capture_every=args.capture_every,
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
        closer.join(timeout=20.0)
        print(json.dumps({"event": "isaac_g1_app_close", "closed": not closer.is_alive()}), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
