"""Source-rate references for recorded-state visualisation.

Two different things are produced per episode and they must not be confused:

* ``reference.npz`` — the 50 Hz **action** trajectory.  This is the encoder
  input and the latent training source; it is never replaced by anything here.
* ``reference_state.npz`` — the source-rate **observation.state** trajectory:
  measured joint qpos, decoded into the canonical body order and the canonical
  Dex3 motor order, hands in physical radians.  It exists so the recorded motion
  can be shown frame-exactly in the simulator; it does not feed the encoder and
  it never opens a blocked 78D action.

A third artifact, ``reference_action_source.npz``, keeps the action trajectory at
source rate (row *i* ↔ frame *i*) for the kinematic action replay; its hand
channels are only applied when the source's action hands are physical radians.

The hand blocks are checked against the pinned MuJoCo limits.  A recorded state
may sit a little outside a modelled range (a real open hand at rest does not have
to respect the URDF's soft limits), so a small tolerance and a per-side violation
budget are explicit parameters rather than silent forgiveness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .joints import hand_order
from .schema import BODY_DIM, HAND_DIM

#: A recorded open hand can sit slightly outside the modelled range; anything
#: beyond this is treated as a mapping error rather than a real-robot offset.
DEFAULT_VIOLATION_TOLERANCE_RAD = 0.10
#: How many of the seven channels per side may exceed that tolerance.
DEFAULT_MAX_VIOLATIONS = 2


@dataclass(frozen=True)
class StateReference:
    """Measured joint state at source rate, in canonical orders."""

    timestamps: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    left_hand_joints: np.ndarray
    right_hand_joints: np.ndarray
    hands_applied: bool
    provenance: dict[str, Any]

    def validate(self) -> None:
        frames = len(self.timestamps)
        expected = {
            "joint_pos": (frames, BODY_DIM),
            "joint_vel": (frames, BODY_DIM),
            "left_hand_joints": (frames, HAND_DIM),
            "right_hand_joints": (frames, HAND_DIM),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if frames == 0:
            raise ValueError("state reference is empty")
        if frames > 1 and not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("state reference timestamps must be strictly increasing")

    def to_arrays(self) -> dict[str, np.ndarray]:
        # Provenance travels inside the artifact as utf-8 bytes: the replay only
        # reads arrays, but a reference file must stay self-describing.
        provenance = json.dumps(self.provenance, sort_keys=True).encode("utf-8")
        return {
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
            "joint_pos": np.asarray(self.joint_pos, dtype=np.float32),
            "joint_vel": np.asarray(self.joint_vel, dtype=np.float32),
            "left_hand_joints": np.asarray(self.left_hand_joints, dtype=np.float32),
            "right_hand_joints": np.asarray(self.right_hand_joints, dtype=np.float32),
            "hands_applied": np.array([bool(self.hands_applied)], dtype=bool),
            "provenance_json": np.frombuffer(provenance, dtype=np.uint8),
        }

    @classmethod
    def from_arrays(cls, payload: dict[str, np.ndarray], provenance: dict[str, Any] | None = None) -> "StateReference":
        encoded = payload.get("provenance_json")
        if provenance is None and encoded is not None:
            provenance = json.loads(bytes(np.asarray(encoded, dtype=np.uint8)).decode("utf-8"))
        applied = payload.get("hands_applied")
        reference = cls(
            timestamps=np.asarray(payload["timestamps"], dtype=np.float64),
            joint_pos=np.asarray(payload["joint_pos"], dtype=np.float32),
            joint_vel=np.asarray(payload["joint_vel"], dtype=np.float32),
            left_hand_joints=np.asarray(payload["left_hand_joints"], dtype=np.float32),
            right_hand_joints=np.asarray(payload["right_hand_joints"], dtype=np.float32),
            hands_applied=bool(np.asarray(applied).reshape(-1)[0]) if applied is not None else True,
            provenance=dict(provenance or {}),
        )
        reference.validate()
        return reference


def hand_limit_report(
    left: np.ndarray,
    right: np.ndarray,
    limits: dict[str, tuple[float, float]],
    *,
    tolerance: float = DEFAULT_VIOLATION_TOLERANCE_RAD,
    max_violations: int = DEFAULT_MAX_VIOLATIONS,
) -> dict[str, Any]:
    """Check decoded hand radians against the pinned limits, per side."""
    report: dict[str, Any] = {"tolerance_rad": tolerance, "max_violations": max_violations, "sides": {}}
    for side, values in (("left", left), ("right", right)):
        data = np.asarray(values, dtype=np.float64)
        names = hand_order(side)
        violations: list[dict[str, Any]] = []
        worst = 0.0
        for index, name in enumerate(names):
            lower, upper = limits[name]
            excess = np.maximum(np.maximum(lower - data[:, index], data[:, index] - upper), 0.0)
            if excess.max() > 0.0:
                violations.append(
                    {
                        "joint": name,
                        "max_excess_rad": float(excess.max()),
                        "frames": int(np.count_nonzero(excess > 0.0)),
                        "limits": [lower, upper],
                    }
                )
                worst = max(worst, float(excess.max()))
        hard = [item for item in violations if item["max_excess_rad"] > tolerance]
        if len(hard) > max_violations:
            raise ValueError(
                f"{side} hand state leaves the pinned Dex3 limits in {len(hard)} channels "
                f"(budget {max_violations}): {hard}"
            )
        report["sides"][side] = {
            "violating_channels": len(violations),
            "beyond_tolerance_channels": len(hard),
            "max_excess_rad": worst,
            "violations": violations,
        }
    return report


def build_state_reference(
    timestamps: np.ndarray,
    joint_pos: np.ndarray,
    left_hand_joints: np.ndarray,
    right_hand_joints: np.ndarray,
    *,
    source_field: str,
    source_fps: float,
    limits: dict[str, tuple[float, float]],
    mapping: dict[str, Any],
    hands_applied: bool = True,
) -> StateReference:
    """Assemble a state reference and derive its velocities at source rate."""
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    body = np.asarray(joint_pos, dtype=np.float64)
    left = np.asarray(left_hand_joints, dtype=np.float64)
    right = np.asarray(right_hand_joints, dtype=np.float64)
    if len(ts) < 2:
        raise ValueError("a state reference needs at least two frames")
    velocity = np.zeros_like(body)
    velocity[:-1] = np.diff(body, axis=0) * float(source_fps)
    limit_report = hand_limit_report(left, right, limits)
    reference = StateReference(
        timestamps=ts,
        joint_pos=body.astype(np.float32),
        joint_vel=velocity.astype(np.float32),
        left_hand_joints=left.astype(np.float32),
        right_hand_joints=right.astype(np.float32),
        hands_applied=hands_applied,
        provenance={
            "source_field": source_field,
            "semantics": "measured joint qpos (radians), not a commanded target",
            "source_fps": float(source_fps),
            "velocity_source": "finite difference of the state trajectory at source rate",
            "body_joint_order": "official SONIC reference / IsaacLab",
            "hand_joint_order": "canonical Dex3 motor order",
            "hand_units": "radians",
            "hands_applied_in_replay": bool(hands_applied),
            "hand_limit_report": limit_report,
            "mapping": mapping,
        },
    )
    reference.validate()
    return reference


def load_state_reference(path: Path) -> StateReference:
    with np.load(path) as payload:
        return StateReference.from_arrays({name: payload[name] for name in payload.files})


def save_state_reference(path: Path, reference: StateReference) -> None:
    np.savez_compressed(path, **reference.to_arrays())
