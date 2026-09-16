#!/usr/bin/env python3
"""Run the official SONIC GEM-X live webcam example from this worktree.

The GEM-X environment lives in this worktree, while the official SONIC source
is provided by the SONIC runtime container.  This small wrapper keeps the
entrypoint in the worktree and delegates all pose estimation/conversion logic
to SONIC's official ``webcam_stream.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


DEFAULT_GEMX_ROOT = "/workspace/romp_faruk/third_party/GEM-X"
DEFAULT_SONIC_ROOT = "/opt/src/sonic"


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gemx-root", default=os.environ.get("GEMX_ROOT", DEFAULT_GEMX_ROOT))
    parser.add_argument("--sonic-root", default=os.environ.get("SONIC_ROOT", DEFAULT_SONIC_ROOT))
    parser.add_argument(
        "--sonic-script",
        default=None,
        help="Override SONIC's live webcam script path (normally not needed).",
    )
    args, forwarded = parser.parse_known_args()

    gemx_root = str(Path(args.gemx_root).expanduser().resolve())
    sonic_root = str(Path(args.sonic_root).expanduser().resolve())
    script = Path(args.sonic_script or (Path(sonic_root) / "gear_sonic/examples/live_camera_teleop/webcam_stream.py"))
    if not script.is_file():
        raise SystemExit(f"Official SONIC webcam script not found: {script}")

    env = os.environ.copy()
    env["GEMX_ROOT"] = gemx_root
    env["SONIC_ROOT"] = sonic_root
    python_path = [gemx_root, str(Path(gemx_root) / "third_party/soma")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    # The official example resolves inputs/onnx and inputs/pretrained relative
    # to GEM-X, so make that the child process working directory.
    os.chdir(gemx_root)
    command = [
        sys.executable,
        str(script),
        "--gemx-root",
        gemx_root,
        "--sonic-root",
        sonic_root,
        *forwarded,
    ]
    os.execvpe(sys.executable, command, env)


if __name__ == "__main__":
    main()
