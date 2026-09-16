"""ROMP SMPL params -> SONIC Protocol-v3 root-local SMPL representation.

This mirrors NVIDIA's canonical live-camera conversion
(``gear_sonic/examples/live_camera_teleop/soma_to_smpl.py:SomaToSmpl.convert``)
line for line. The ONLY difference is the joint source: NVIDIA's path gets 24
SMPL joints from GEM-X/SOMA; we get them from ROMP's SMPL params via SONIC's own
``compute_human_joints``. Everything after the joints are obtained is identical:

    joints24 (Y-up, global applied)         <- compute_human_joints(ROMP pose)
    g_quat   = aa2quat(global_orient)       (Y-up)
    g_quat_z = smpl_root_ytoz_up(g_quat)    (+90 deg about X)
    g_quat_nobase = remove_smpl_base_rot(g_quat_z, w_last=False)
    joints0  = joints24 - joints24[0]       (pelvis at origin, Y-up)
    joints_z = joints0 @ R_up.T             (Z-up)
    smpl_joints = quat_apply(quat_inv(g_quat_nobase) tiled, joints_z)
    body_quat   = g_quat_nobase

``global_orient`` comes from ROMP in a camera (Y-down) frame; ``flip_x`` maps it
to SMPL Y-up (a convention mapping, not a control constraint). ``root_mode``
defaults to ``full`` (gravity-aligned full root orientation, as in the NVIDIA
path); ``yaw``/``identity`` exist only for debugging.

Requires `gear_sonic` importable (run inside the container with
`--sonic-root /opt/src/sonic`).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from .utils import YUP_TO_ZUP, flip_romp_global_orient, yaw_only_axis_angle


def _is_usable_torch_file(path: str) -> bool:
    """Return false for an unresolved Git-LFS pointer file."""
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        return not head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


class RompToSmpl24:
    def __init__(self, sonic_root: str | None = None,
                 human_joints_info: str | None = None,
                 root_mode: str = "full",
                 flip_x: bool = True):
        assert root_mode in ("full", "yaw", "identity"), root_mode
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
        )
        from gear_sonic.isaac_utils.rotations import (
            remove_smpl_base_rot,
            smpl_root_ytoz_up,
        )

        self._torch = torch
        self._tt = tt
        self._aa2quat = angle_axis_to_quaternion
        self._quat_apply = quat_apply
        self._quat_inv = quat_inv
        self._remove_base = remove_smpl_base_rot
        self._ytoz = smpl_root_ytoz_up
        self._R_up = torch.tensor(YUP_TO_ZUP, dtype=torch.float32)

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
                if _is_usable_torch_file(c):
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

        Returns (smpl_joints (24,3) f32 root-local Z-up, body_quat (4,) f32),
        exactly the fields SONIC's SMPL encoder (mode 2) consumes.
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

            # 24 SMPL joints, Y-up, with the global orientation applied
            # (NVIDIA obtains these from SOMA; we get them from ROMP's pose).
            joints24 = self._compute_human_joints(
                bp_t[..., :63], global_orient=go_t,
                human_joints_info_path=self._human_joints_info,
            )  # (1,24,3)

            # --- identical to soma_to_smpl.SomaToSmpl.convert ---------------
            g_quat = self._aa2quat(go_t)[:, :4]          # (1,4) wxyz, Y-up
            g_quat_z = self._ytoz(g_quat)                # (1,4) Z-up
            g_quat_nobase = self._remove_base(g_quat_z, w_last=False)  # (1,4)

            joints0 = joints24 - joints24[:, :1]         # pelvis at origin (Y-up)
            joints_z = joints0[0] @ self._R_up.T         # (24,3) Z-up
            inv = self._quat_inv(g_quat_nobase).reshape(-1, 4)
            inv = inv.repeat(joints_z.shape[0], 1)       # (24,4)
            smpl_joints = self._quat_apply(inv, joints_z)  # (24,3)

        return (smpl_joints.detach().cpu().numpy().astype(np.float32),
                g_quat_nobase.reshape(-1, 4)[0].detach().cpu().numpy().astype(np.float32))
