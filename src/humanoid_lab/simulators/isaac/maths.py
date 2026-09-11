"""Small rotation helpers used by the simulator's start-up support."""

from __future__ import annotations

import torch


def quat_to_rotation_vector(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert a batch of wxyz quaternions to rotation vectors (axis * angle)."""
    quaternion = quaternion_wxyz / quaternion_wxyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w = quaternion[..., 0].clamp(-1.0, 1.0)
    xyz = quaternion[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half.squeeze(-1), w)
    axis = xyz / sin_half.clamp(min=1e-8)
    # Near-zero rotation: the axis is arbitrary, so return the vector directly.
    small = sin_half.squeeze(-1) < 1e-6
    result = axis * angle.unsqueeze(-1)
    return torch.where(small.unsqueeze(-1), 2.0 * xyz, result)
