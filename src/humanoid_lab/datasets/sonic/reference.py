"""Reference composition policies for datasets without a body trajectory.

Two policies exist and they are not interchangeable:

* :func:`compose_unitree_static_completion` — production policy for the Unitree
  arm-only collection.  The lower body and the root are frozen at one validated
  standing frame for the whole episode, and joint velocities are re-derived from
  the composed positions.  It is deterministic for any episode length, has no
  hidden loop back into a short capture, and is only valid for stationary
  manipulation: the arms move, the feet do not.
* :func:`compose_unitree_standing_reference` — earlier diagnostic policy that
  warps a captured IDLE *time series* onto the demonstration and refuses
  episodes longer than the capture.  Kept because the dataset contract tests and
  the direct-reference diagnostic still describe it; it is not used to produce
  processed training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_lab.controllers.sonic import (
    BODY_JOINT_ORDER,
    DEFAULT_STANDING_POSE_RAD,
    SONIC_REFERENCE_JOINT_ORDER,
    STANDING_ROOT_HEIGHT_M,
)

from .joints import reorder
from .schema import CanonicalEpisode
from .timeline import finite_difference, linear_resample, uniform_timeline

ARM_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

#: Declared applicability of the static-completion policy.
STANDING_COMPLETION_SCOPE = "stationary_manipulation_only"


@dataclass(frozen=True)
class StandingPose:
    """One validated standing frame in official SONIC reference/IsaacLab order."""

    joint_pos: np.ndarray
    body_pos: np.ndarray
    body_quat_wxyz: np.ndarray
    source: str
    provenance: dict[str, object]


def deployment_standing_pose() -> StandingPose:
    """The pose the pinned deployment ramps the robot to before control.

    Using it as the frozen lower body keeps the reference consistent with both
    the simulator spawn pose and the official IDLE planner takeover.
    """
    joint_pos = reorder(
        np.array([DEFAULT_STANDING_POSE_RAD[name] for name in BODY_JOINT_ORDER], dtype=np.float64),
        BODY_JOINT_ORDER,
        SONIC_REFERENCE_JOINT_ORDER,
    )
    return StandingPose(
        joint_pos=joint_pos,
        body_pos=np.array([0.0, 0.0, STANDING_ROOT_HEIGHT_M], dtype=np.float64),
        body_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        source="official_sonic_default_standing_pose",
        provenance={
            "source": "official_policy_parameters_default_angles",
            "body_joint_order": "official SONIC reference / IsaacLab",
            "root_height_m": STANDING_ROOT_HEIGHT_M,
        },
    )


def load_standing_pose(directory: Path) -> StandingPose:
    """Load a captured static standing frame, refusing moving captures."""
    directory = Path(directory)
    provenance: dict[str, object] = {}
    manifest = directory / "provenance.json"
    if manifest.is_file():
        import json

        provenance = json.loads(manifest.read_text(encoding="utf-8"))
    if provenance.get("lower_body_is_time_series"):
        raise ValueError(
            f"{directory} holds a moving IDLE capture; the static completion policy needs a single "
            "validated standing frame (pass --standing-reference at such a capture instead)"
        )
    joint_pos = np.loadtxt(directory / "joint_pos.csv", delimiter=",", ndmin=2)
    if joint_pos.shape[1] != 29:
        raise ValueError(f"{directory}/joint_pos.csv must hold 29 joints per row")
    if len(np.unique(joint_pos, axis=0)) != 1:
        raise ValueError(f"{directory}/joint_pos.csv is not a single static frame")
    return StandingPose(
        joint_pos=joint_pos[0],
        body_pos=np.loadtxt(directory / "body_pos.csv", delimiter=",", ndmin=2)[0],
        body_quat_wxyz=np.loadtxt(directory / "body_quat.csv", delimiter=",", ndmin=2)[0],
        source=str(provenance.get("source", f"captured_static_frame:{directory.name}")),
        provenance=provenance,
    )


def compose_unitree_static_completion(
    timestamps: np.ndarray,
    arms: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    *,
    standing: StandingPose,
) -> CanonicalEpisode:
    """Compose the Unitree reference from absolute arm actions and one standing frame.

    Nothing in the lower body or the root is time-varying, so a long episode
    cannot silently wrap a shorter capture: the tail of the episode is the same
    validated pose as its head, and the encoder look-ahead clamps onto that pose.
    """
    if np.asarray(arms).shape[1:] != (14,):
        raise ValueError("Unitree arm action must be 14D")
    target = uniform_timeline(timestamps)
    joint_pos = np.tile(np.asarray(standing.joint_pos, dtype=np.float64), (len(target), 1))
    arm_ids = [SONIC_REFERENCE_JOINT_ORDER.index(name) for name in ARM_NAMES]
    joint_pos[:, arm_ids] = linear_resample(timestamps, arms, target)
    episode = CanonicalEpisode(
        timestamps=target,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=finite_difference(joint_pos).astype(np.float32),
        body_quat_wxyz=np.tile(np.asarray(standing.body_quat_wxyz, dtype=np.float32), (len(target), 1)),
        body_pos=np.tile(np.asarray(standing.body_pos, dtype=np.float32), (len(target), 1)),
        left_hand_joints=linear_resample(timestamps, left_hand, target),
        right_hand_joints=linear_resample(timestamps, right_hand, target),
    )
    episode.validate()
    return episode


def compose_unitree_standing_reference(
    timestamps: np.ndarray,
    arms: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    *,
    idle_timestamps: np.ndarray,
    idle_joint_pos: np.ndarray,
    idle_joint_vel: np.ndarray,
    idle_body_pos: np.ndarray,
    idle_body_quat_wxyz: np.ndarray,
) -> CanonicalEpisode:
    """Legacy diagnostic policy: warp a captured IDLE capture onto the episode."""
    if np.asarray(arms).shape[1:] != (14,):
        raise ValueError("Unitree arm action must be 14D")
    target = uniform_timeline(timestamps)
    idle_ts = np.asarray(idle_timestamps, dtype=np.float64)
    idle_target = idle_ts[0] + (target - target[0])
    if idle_target[-1] > idle_ts[-1] + 1e-9:
        raise ValueError("SONIC IDLE trajectory is shorter than the resampled demonstration")
    q = linear_resample(idle_ts, idle_joint_pos, idle_target)
    qd = linear_resample(idle_ts, idle_joint_vel, idle_target)
    arm_ids = [SONIC_REFERENCE_JOINT_ORDER.index(name) for name in ARM_NAMES]
    arm_q = linear_resample(timestamps, arms, target)
    q[:, arm_ids] = arm_q
    qd[:, arm_ids] = finite_difference(arm_q)

    quat_source = np.asarray(idle_body_quat_wxyz, dtype=np.float64).copy()
    if quat_source.shape != (len(idle_ts), 4):
        raise ValueError("idle_body_quat_wxyz must be [frames, 4]")
    # Keep interpolation on one quaternion hemisphere, then renormalize.
    for index in range(1, len(quat_source)):
        if np.dot(quat_source[index - 1], quat_source[index]) < 0:
            quat_source[index] *= -1
    body_quat = linear_resample(idle_ts, quat_source, idle_target)
    norms = np.linalg.norm(body_quat, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("SONIC IDLE trajectory contains an invalid quaternion")
    body_quat = (body_quat / norms).astype(np.float32)
    episode = CanonicalEpisode(
        timestamps=target,
        joint_pos=q,
        joint_vel=qd,
        body_quat_wxyz=body_quat,
        body_pos=linear_resample(idle_ts, idle_body_pos, idle_target),
        left_hand_joints=linear_resample(timestamps, left_hand, target),
        right_hand_joints=linear_resample(timestamps, right_hand, target),
    )
    episode.validate()
    return episode
