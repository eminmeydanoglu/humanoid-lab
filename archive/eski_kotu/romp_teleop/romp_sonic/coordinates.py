"""ROMP SMPL params -> SONIC Protocol-v3 root-local SMPL representation.

This deliberately reuses NVIDIA's own helpers (not a reimplementation) and
mirrors `gear_sonic/scripts/pico_manager_thread_server.py:process_smpl_joints`
step for step, so the produced `smpl_joints` / `body_quat` match what the
deployed SONIC SMPL encoder (mode 2) was trained/consumes.

Pipeline:
    ROMP global_orient (aa, SMPL Y-up)
      -> [optional root mode: yaw-only / identity]
      -> aa -> quat
      -> smpl_root_ytoz_up          (+90 deg about X, Y-up -> Z-up)
      -> compute_human_joints        (24 joints: SMPL 0..21 + thumb tips)
      -> remove_smpl_base_rot        (conjugate out SMPL rest orientation)
      -> quat_inv(root) * joints     (root-local, Z-up)

Requires `gear_sonic` importable (run inside the container with
`--sonic-root /opt/src/sonic`, or set PYTHONPATH to the SONIC source).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from .utils import flip_romp_global_orient, yaw_only_axis_angle


class RompToSmpl24:
    def __init__(self, sonic_root: str | None = None,
                 human_joints_info: str | None = None,
                 root_mode: str = "yaw",
                 flip_x: bool = True):
        assert root_mode in ("yaw", "full", "identity"), root_mode
        self.root_mode = root_mode
        self.flip_x = flip_x
        if sonic_root and sonic_root not in sys.path:
            sys.path.insert(0, sonic_root)

        import torch
        from gear_sonic.trl.utils import torch_transform as tt
        from gear_sonic.trl.utils.torch_transform import (
            angle_axis_to_quaternion,
            compute_human_joints,
            quat_apply,
            quat_inv,
            quaternion_to_angle_axis,
        )
        from gear_sonic.isaac_utils.rotations import (
            remove_smpl_base_rot,
            smpl_root_ytoz_up,
        )

        self._torch = torch
        self._tt = tt
        self._aa2quat = angle_axis_to_quaternion
        self._quat2aa = quaternion_to_angle_axis
        self._quat_apply = quat_apply
        self._quat_inv = quat_inv
        self._remove_base = remove_smpl_base_rot
        self._ytoz = smpl_root_ytoz_up

        # Resolve the joints info file: explicit arg, else alongside sonic_root.
        if human_joints_info is None:
            candidates = []
            if sonic_root:
                candidates.append(os.path.join(
                    sonic_root, "gear_sonic", "data", "human",
                    "human_joints_info.pkl"))
            candidates.append(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "human_joints_info.pkl"))
            for c in candidates:
                if os.path.isfile(c):
                    human_joints_info = c
                    break
        # PyTorch >= 2.6 defaults torch.load(weights_only=True), which cannot
        # unpickle the numpy objects inside human_joints_info.pkl. Load it
        # ourselves and inject the module global so upstream's lazy loader is
        # bypassed (no upstream file is modified).
        if human_joints_info and tt.human_joints_info is None:
            tt.human_joints_info = torch.load(human_joints_info, weights_only=False)
            print(f"[coords] loaded human_joints_info from {human_joints_info}")

        self._human_joints_info = human_joints_info
        self._compute_human_joints = compute_human_joints

    def convert(self, global_orient: np.ndarray, body_pose: np.ndarray):
        """global_orient: (3,) axis-angle; body_pose: (>=63,) axis-angle.

        Returns (smpl_joints (24,3) f32 root-local Z-up, body_quat (4,) f32).
        """
        torch = self._torch
        with torch.no_grad():
            go = np.asarray(global_orient, dtype=np.float32).reshape(3)
            if self.flip_x:
                # ROMP root is ~180deg about X (Y-down); bring to SMPL Y-up.
                go = flip_romp_global_orient(go)
            if self.root_mode == "identity":
                go = np.zeros(3, dtype=np.float32)
            elif self.root_mode == "yaw":
                go = yaw_only_axis_angle(go)

            go_t = torch.as_tensor(go, dtype=torch.float32).view(1, 3)
            bp = np.asarray(body_pose, dtype=np.float32).reshape(-1)
            bp_t = torch.as_tensor(bp, dtype=torch.float32).view(1, -1)

            q = self._aa2quat(go_t)[:, :4]
            q = self._ytoz(q)
            go_new = self._quat2aa(q)

            joints = self._compute_human_joints(
                bp_t[..., :63], global_orient=go_new,
                human_joints_info_path=self._human_joints_info,
            )  # (1,24,3)

            q = self._remove_base(q, w_last=False)
            q_inv = self._quat_inv(q).unsqueeze(1).repeat(1, joints.shape[1], 1)
            smpl_joints = self._quat_apply(q_inv, joints)

        return (smpl_joints[0].detach().cpu().numpy().astype(np.float32),
                q[0].detach().cpu().numpy().astype(np.float32))
