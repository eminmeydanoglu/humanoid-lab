#!/usr/bin/env python3
"""Run the controller-free Isaac G1 simulator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
