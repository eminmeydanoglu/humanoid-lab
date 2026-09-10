#!/usr/bin/env python3
"""Empirical coordinate diagnostics for the ROMP -> SONIC transform.

Prints landmark vectors for:
  A) ROMP's own joints (joints24_raw from the NPZ)
  B) SONIC compute_human_joints with identity global orientation
  C) SONIC compute_human_joints with ROMP's actual global orientation
  D) the final root-local Z-up smpl_joints (as sent to SONIC)

Run inside the container (needs gear_sonic):
  python tools/diag_coordinates.py --npz webcam_smpl.npz --sonic-root /opt/src/sonic
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def vectors(p):
    return {
        "head-pelvis": p[15] - p[0],
        "Lsh-Rsh": p[16] - p[17],
        "Lhip-pelvis": p[1] - p[0],
        "Lankle-pelvis": p[7] - p[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--sonic-root", default="/opt/src/sonic")
    ap.add_argument("--root-mode", default="yaw")
    args = ap.parse_args()

    import torch
    from gear_sonic.trl.utils import torch_transform as tt
    from gear_sonic.trl.utils.torch_transform import (
        angle_axis_to_quaternion, compute_human_joints, quaternion_to_angle_axis,
    )
    from gear_sonic.isaac_utils.rotations import smpl_root_ytoz_up

    info_path = os.path.join(args.sonic_root, "gear_sonic", "data", "human",
                             "human_joints_info.pkl")
    if tt.human_joints_info is None:
        tt.human_joints_info = torch.load(info_path, weights_only=False)

    from romp_sonic.coordinates import RompToSmpl24
    conv = RompToSmpl24(sonic_root=args.sonic_root, root_mode=args.root_mode)

    data = np.load(args.npz, allow_pickle=True)
    thetas = np.asarray(data["smpl_thetas"], np.float32)
    raw = np.asarray(data["joints24_raw"], np.float32)

    for k in [0, len(thetas) // 2, len(thetas) - 1]:
        print(f"\n================ frame {k} ================")
        print("ROMP global_orient (aa):", thetas[k][:3].round(4))
        print("[A] ROMP joints    :", {n: v.round(3).tolist() for n, v in vectors(raw[k]).items()})

        go = torch.tensor(thetas[k][:3], dtype=torch.float32).view(1, 3)
        bp = torch.tensor(thetas[k][3:], dtype=torch.float32).view(1, -1)

        j_id = compute_human_joints(bp[..., :63],
                                    global_orient=torch.zeros(1, 3))[0].numpy()
        print("[B] SONIC go=ident :", {n: v.round(3).tolist() for n, v in vectors(j_id).items()})

        q = smpl_root_ytoz_up(angle_axis_to_quaternion(go)[:, :4])
        j_act = compute_human_joints(bp[..., :63],
                                     global_orient=quaternion_to_angle_axis(q))[0].numpy()
        print("[C] SONIC go=ROMP  :", {n: v.round(3).tolist() for n, v in vectors(j_act).items()})

        from romp_sonic.utils import flip_romp_global_orient
        go_f = torch.tensor(flip_romp_global_orient(thetas[k][:3]),
                            dtype=torch.float32).view(1, 3)
        qf = smpl_root_ytoz_up(angle_axis_to_quaternion(go_f)[:, :4])
        j_flip = compute_human_joints(bp[..., :63],
                                      global_orient=quaternion_to_angle_axis(qf))[0].numpy()
        print("[C2] SONIC go=FLIP :", {n: v.round(3).tolist() for n, v in vectors(j_flip).items()})

        sj, bq = conv.convert(thetas[k][:3], thetas[k][3:])
        print("[D] SONIC local Zup:", {n: v.round(3).tolist() for n, v in vectors(sj).items()})
        print("    body_quat:", bq.round(4).tolist(), "norm", float(np.linalg.norm(bq)))


if __name__ == "__main__":
    main()
