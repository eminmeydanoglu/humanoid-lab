"""Read simulator tracking traces and turn them into reviewable metrics.

The Isaac G1 service writes one row per applied control tick: the command it sent
(``body_target``, ``*_hand_target``) and what the joints actually did
(``body_measured``, ``*_hand_measured``), plus root state.  Both the command and
the measurement are in hardware/MuJoCo body order, which is the order the
deployment's DDS commands use, so the per-joint tables need that vocabulary, not
the reference/IsaacLab one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER

from .quality import joint_error_metrics

#: Fixed-base direct replay of a canonical reference is a strong statement, so
#: the thresholds gate gross failure rather than actuator bandwidth.  The wrist
#: joints are the slow ones by construction: the deployment drives them with the
#: small 4010 motors at low proportional gains, so a fast reference wrist leaves a
#: visible lag under pure PD tracking.  Their own threshold is set accordingly and
#: the per-joint table keeps the detail visible.
DIRECT_THRESHOLDS: dict[str, float] = {
    "tracking.body.mae_rad": 0.15,
    "tracking.body.p95_rad": 0.60,
    "tracking.body.max_rad": 1.30,
    "tracking.legs_waist.mae_rad": 0.08,
    "tracking.wrists.mae_rad": 0.60,
    "tracking.hand.mae_rad": 0.25,
    "tracking.fell": 0.5,
}
#: Free-standing SONIC tracking is judged on stability first; the deployment
#: drives the same joints through a policy that was never trained on these
#: synthetic-root references, so a loose error bound is the honest threshold.
FREE_THRESHOLDS: dict[str, float] = {
    "tracking.body.mae_rad": 0.35,
    "tracking.body.p95_rad": 0.90,
    "tracking.fell": 0.5,
}
FALL_ROOT_HEIGHT_M = 0.45
FALL_TILT_DEG = 60.0
LEG_WAIST_TOKENS = ("hip", "knee", "ankle", "waist")


def load_tracking(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(Path(path))
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"tracking trace is empty: {path}")

    def stack(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    def stack_hand(side: str, kind: str) -> np.ndarray:
        values = [row[f"{side}_hand_{kind}"] for row in rows]
        if any(value is None for value in values):
            raise ValueError(f"{path}: {side} hand {kind} is missing in some rows")
        return np.asarray(values, dtype=np.float64)

    root_quaternion = stack("root_quaternion_wxyz")
    up_z = 1.0 - 2.0 * (root_quaternion[:, 1] ** 2 + root_quaternion[:, 2] ** 2)
    result = {
        "path": str(path),
        "sim_s": stack("sim_s"),
        "wall_time_ns": stack("wall_time_ns").astype(np.int64) if "wall_time_ns" in table.column_names else np.empty(0, dtype=np.int64),
        "tick": stack("tick"),
        "support_active": np.asarray([bool(row["support_active"]) for row in rows]),
        "body_target": stack("body_target"),
        "body_measured": stack("body_measured"),
        "left_hand_target": stack_hand("left", "target"),
        "left_hand_measured": stack_hand("left", "measured"),
        "right_hand_target": stack_hand("right", "target"),
        "right_hand_measured": stack_hand("right", "measured"),
        "root_position": stack("root_position"),
        "root_quaternion_wxyz": root_quaternion,
        "root_up_z": np.clip(up_z, -1.0, 1.0),
        "body_joint_names": BODY_JOINT_ORDER,
        "rows": len(rows),
    }
    for name in ("body_velocity_target", "body_feedforward_torque", "body_kp", "body_kd",
                 "body_applied_torque", "body_measured_velocity", "body_gravity_feedforward_torque"):
        if name in table.column_names:
            result[name] = stack(name)
    for side in ("left", "right"):
        for name in ("velocity_target", "feedforward_torque", "kp", "kd", "applied_torque", "measured_velocity"):
            field = f"{side}_hand_{name}"
            if field in table.column_names:
                result[field] = stack_hand(side, name)
    return result


def free_window(tracking: dict[str, Any]) -> np.ndarray:
    """Rows after the pelvis support released; all rows when it never engaged."""
    mask = ~np.asarray(tracking["support_active"], dtype=bool)
    if not np.any(mask):
        return np.ones_like(mask)
    return mask


def motion_start_index(tracking: dict[str, Any], *, epsilon: float = 1e-6) -> int | None:
    """First row whose command differs from the pose the run started in.

    Fixed-base oracle runs spend their pre-roll holding the standing pose; scoring
    that approach phase as tracking error would blame the controller for the
    trajectory it has not been asked to follow yet.
    """
    target = np.asarray(tracking["body_target"], dtype=np.float64)
    moving = np.flatnonzero(np.abs(target - target[0]).max(axis=1) > epsilon)
    return None if moving.size == 0 else int(moving[0])


def motion_window(tracking: dict[str, Any]) -> np.ndarray:
    start = motion_start_index(tracking)
    mask = np.zeros(len(tracking["sim_s"]), dtype=bool)
    mask[0 if start is None else start :] = True
    return mask


def window_mask(tracking: dict[str, Any], window: str) -> np.ndarray:
    if window == "all":
        return np.ones(len(tracking["sim_s"]), dtype=bool)
    if window == "free":
        return free_window(tracking)
    if window == "motion":
        return motion_window(tracking)
    raise ValueError(f"unknown window {window!r}; expected all, free or motion")


def stability_metrics(tracking: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    up_z = np.asarray(tracking["root_up_z"])[mask]
    heights = np.asarray(tracking["root_position"])[:, 2][mask]
    tilt_deg = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
    fell = bool(np.any(heights < FALL_ROOT_HEIGHT_M) or np.any(tilt_deg > FALL_TILT_DEG))
    return {
        "fell": 1.0 if fell else 0.0,
        "min_root_height_m": float(heights.min()),
        "final_root_height_m": float(heights[-1]),
        "max_tilt_deg": float(tilt_deg.max()),
        "final_tilt_deg": float(tilt_deg[-1]),
        "frames": int(mask.sum()),
    }


def joint_groups(joint_names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Named joint groups used when a single aggregate would hide the detail."""
    legs_waist = tuple(name for name in joint_names if any(token in name for token in LEG_WAIST_TOKENS))
    wrists = tuple(name for name in joint_names if name.startswith(("left_wrist", "right_wrist")))
    arms = tuple(
        name
        for name in joint_names
        if name.startswith(("left_shoulder", "right_shoulder", "left_elbow", "right_elbow"))
    )
    return {"legs_waist": legs_waist, "arms": arms, "wrists": wrists}


def max_command_step(tracking: dict[str, Any]) -> dict[str, Any]:
    """Largest single-tick jump in the applied command.

    The deployment switches from its IDLE planner to the streamed reference when
    a replay starts, so one large step is expected at that moment; it is reported
    instead of being smoothed away, because it is where free-standing runs are
    most likely to lose balance.
    """
    target = np.asarray(tracking["body_target"], dtype=np.float64)
    if len(target) < 2:
        return {"value": 0.0, "sim_s": 0.0}
    step = np.abs(np.diff(target, axis=0)).max(axis=1)
    index = int(np.argmax(step))
    return {"value": float(step[index]), "sim_s": float(tracking["sim_s"][index + 1])}


def tracking_metrics(tracking: dict[str, Any], *, window: str = "free") -> dict[str, Any]:
    """Per-joint and aggregate target-vs-measured statistics for one window."""
    mask = window_mask(tracking, window)
    names = tuple(tracking["body_joint_names"])
    body = joint_error_metrics(tracking["body_target"][mask], tracking["body_measured"][mask], names)
    group_metrics: dict[str, Any] = {}
    for group, members in joint_groups(names).items():
        errors = np.abs(tracking["body_target"][mask] - tracking["body_measured"][mask])
        indices = [names.index(name) for name in members]
        group_metrics[group] = {
            "joints": list(members),
            "mae": float(errors[:, indices].mean()) if indices else 0.0,
            "p95": float(np.percentile(errors[:, indices], 95)) if indices else 0.0,
            "max": float(errors[:, indices].max()) if indices else 0.0,
        }
    hands: dict[str, Any] = {}
    for side in ("left", "right"):
        target = tracking[f"{side}_hand_target"][mask]
        measured = tracking[f"{side}_hand_measured"][mask]
        hands[side] = joint_error_metrics(target, measured, tuple(f"{side}_hand_{index}" for index in range(7)))
    stability = stability_metrics(tracking, mask)
    values = {
        "tracking.body.mae_rad": body["overall"]["mae"],
        "tracking.body.p95_rad": body["overall"]["p95"],
        "tracking.body.max_rad": body["overall"]["max"],
        "tracking.legs_waist.mae_rad": group_metrics["legs_waist"]["mae"],
        "tracking.arms.mae_rad": group_metrics["arms"]["mae"],
        "tracking.wrists.mae_rad": group_metrics["wrists"]["mae"],
        "tracking.hand.mae_rad": max(hands["left"]["overall"]["mae"], hands["right"]["overall"]["mae"]),
        "tracking.fell": stability["fell"],
    }
    return {
        "window": window,
        "frames": int(mask.sum()),
        "sim_seconds": float(tracking["sim_s"][mask][-1] - tracking["sim_s"][mask][0]) if np.any(mask) else 0.0,
        "body": body,
        "groups": group_metrics,
        "hands": hands,
        "stability": stability,
        "values": values,
    }
