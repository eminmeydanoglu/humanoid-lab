#!/usr/bin/env python3
"""Sanity tests for the ROMP->SONIC SMPL conversion (run in the SONIC env).

  /opt/venvs/sonic-sim/bin/python tests/test_coordinates.py

Checks the properties NVIDIA's soma_to_smpl.convert guarantees:
  - smpl_joints shape (24,3) float32, finite
  - pelvis at origin (root-local)
  - body_quat unit norm
  - deterministic
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from romp_sonic.coordinates import RompToSmpl24

SONIC_ROOT = os.environ.get("SONIC_ROOT", "/opt/src/sonic")


def main() -> int:
    conv = RompToSmpl24(sonic_root=SONIC_ROOT, root_mode="full")
    rng = np.random.default_rng(0)

    cases = {
        "zeros": (np.zeros(3, np.float32), np.zeros(69, np.float32)),
        "random": (rng.normal(0, 0.25, 3).astype(np.float32),
                   rng.normal(0, 0.25, 69).astype(np.float32)),
    }

    for name, (go, bp) in cases.items():
        j, q = conv.convert(go, bp)
        assert j.shape == (24, 3), f"{name}: shape {j.shape}"
        assert j.dtype == np.float32, f"{name}: dtype {j.dtype}"
        assert np.isfinite(j).all(), f"{name}: non-finite joints"
        assert np.isfinite(q).all(), f"{name}: non-finite quat"
        pelvis = float(np.abs(j[0]).max())
        qn = float(np.linalg.norm(q))
        assert pelvis < 1e-4, f"{name}: pelvis not centered -> {j[0]}"
        assert abs(qn - 1.0) < 1e-3, f"{name}: quat norm {qn}"

        j2, q2 = conv.convert(go, bp)
        assert np.allclose(j, j2, atol=1e-6), f"{name}: non-deterministic joints"
        assert np.allclose(q, q2, atol=1e-6), f"{name}: non-deterministic quat"
        print(f"[test] {name:7s} pelvis_off={pelvis:.2e} quat_norm={qn:.5f} "
              f"joint_range=[{j.min():.3f},{j.max():.3f}]")

    # GEM teleop preserves the complete gravity-aligned root orientation.
    # A non-yaw input must therefore produce a different full-root quaternion
    # from the diagnostic yaw-only path.
    go = np.array([0.55, 0.25, -0.35], dtype=np.float32)
    bp = np.zeros(69, dtype=np.float32)
    _, q_full = RompToSmpl24(sonic_root=SONIC_ROOT, root_mode="full").convert(go, bp)
    _, q_yaw = RompToSmpl24(sonic_root=SONIC_ROOT, root_mode="yaw").convert(go, bp)
    assert not np.allclose(q_full, q_yaw, atol=1e-3), "full root unexpectedly became yaw-only"
    print("[test] full root preserves pitch/roll relative to diagnostic yaw mode")

    print("test_coordinates OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
