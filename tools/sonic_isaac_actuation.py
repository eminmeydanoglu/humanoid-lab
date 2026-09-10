"""Body torque application and the lowcmd fail-safe state machine.

Pure logic again: the shipped step function, the clamp and the PASSIVE
transition are all drivable without Isaac so tests exercise the same code path
the runner uses.

The effort law is the one the upstream MuJoCo sim2sim bridge applies in
``gear_sonic/utils/mujoco_sim/base_sim.py::compute_body_torques``::

    tau = motor_cmd.tau + motor_cmd.kp * (motor_cmd.q - q)
                        + motor_cmd.kd * (motor_cmd.dq - dq)

followed by a clip to the per-joint effort limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence
import math

from sonic_isaac_contract import BODY_JOINT_COUNT, LOWCMD_MAX_AGE_S

__all__ = [
    "ActuationMode",
    "ActuationStep",
    "BodyActuation",
    "LowCmdSample",
    "RateMeter",
    "SubmitOutcome",
    "body_effort",
    "is_finite_vector",
]


class ActuationMode(Enum):
    PASSIVE = "passive"
    CONTROLLED = "controlled"


@dataclass(frozen=True)
class LowCmdSample:
    """One validated ``rt/lowcmd`` message, in DDS motor index order."""

    sequence: int
    received_at: float
    q: tuple[float, ...]
    dq: tuple[float, ...]
    tau: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("q", "dq", "tau", "kp", "kd"):
            if len(getattr(self, name)) != BODY_JOINT_COUNT:
                raise ValueError(
                    f"lowcmd field {name!r} must hold {BODY_JOINT_COUNT} joints, "
                    f"got {len(getattr(self, name))}"
                )


@dataclass(frozen=True)
class SubmitOutcome:
    accepted: bool
    reason: str
    sequence: int


@dataclass(frozen=True)
class ActuationStep:
    efforts: tuple[float, ...]
    mode: ActuationMode
    reason: str
    applied_sequence: int | None
    lowcmd_age_s: float | None


class RateMeter:
    """Mean publish frequency and worst inter-message gap for a topic."""

    def __init__(self) -> None:
        self._timestamps: list[float] = []

    def mark(self, now: float) -> None:
        self._timestamps.append(float(now))

    @property
    def count(self) -> int:
        return len(self._timestamps)

    def mean_hz(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / span

    def max_gap_ms(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        return max(
            (later - earlier) * 1000.0
            for earlier, later in zip(self._timestamps, self._timestamps[1:])
        )

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "mean_hz": self.mean_hz(),
            "max_gap_ms": self.max_gap_ms(),
        }


def is_finite_vector(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def body_effort(
    q_cmd: Sequence[float],
    dq_cmd: Sequence[float],
    tau_ff: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    q: Sequence[float],
    dq: Sequence[float],
    effort_limits: Sequence[float],
) -> tuple[float, ...]:
    """Return the clamped body effort for one physics step."""
    if not (len(q_cmd) == len(dq_cmd) == len(tau_ff) == len(kp) == len(kd) == len(q)
            == len(dq) == len(effort_limits)):
        raise ValueError("body_effort inputs must all have the same length")
    efforts: list[float] = []
    for index in range(len(q_cmd)):
        torque = (
            float(tau_ff[index])
            + float(kp[index]) * (float(q_cmd[index]) - float(q[index]))
            + float(kd[index]) * (float(dq_cmd[index]) - float(dq[index]))
        )
        if not math.isfinite(torque):
            raise ValueError(f"non-finite effort for joint {index}")
        limit = abs(float(effort_limits[index]))
        efforts.append(max(-limit, min(limit, torque)))
    return tuple(efforts)


class BodyActuation:
    """Tracks the newest valid lowcmd and decides controlled vs passive."""

    def __init__(
        self,
        effort_limits: Sequence[float],
        *,
        max_age_s: float = LOWCMD_MAX_AGE_S,
    ) -> None:
        self.effort_limits = tuple(float(value) for value in effort_limits)
        self.max_age_s = float(max_age_s)
        self._last: LowCmdSample | None = None
        self._accepted = 0
        self._rejected = 0
        self.rejections: list[str] = []
        self.rate = RateMeter()

    # -- ingest ------------------------------------------------------------ #

    def submit(self, sample: LowCmdSample) -> SubmitOutcome:
        for name in ("q", "dq", "tau", "kp", "kd"):
            if not is_finite_vector(getattr(sample, name)):
                return self._reject(sample, f"non_finite:{name}")
        if self._last is not None and sample.sequence <= self._last.sequence:
            return self._reject(sample, "out_of_order")
        self._last = sample
        self._accepted += 1
        self.rate.mark(sample.received_at)
        return SubmitOutcome(True, "accepted", sample.sequence)

    def _reject(self, sample: LowCmdSample, reason: str) -> SubmitOutcome:
        self._rejected += 1
        self.rejections.append(reason)
        return SubmitOutcome(False, reason, sample.sequence)

    # -- apply ------------------------------------------------------------- #

    @property
    def mode(self) -> ActuationMode:
        return ActuationMode.PASSIVE if self._last is None else ActuationMode.CONTROLLED

    def step(self, now: float, q: Sequence[float], dq: Sequence[float]) -> ActuationStep:
        """Efforts to write for a physics step starting at monotonic time ``now``."""
        if self._last is None:
            return ActuationStep(
                efforts=(0.0,) * len(self.effort_limits),
                mode=ActuationMode.PASSIVE,
                reason="no_controller",
                applied_sequence=None,
                lowcmd_age_s=None,
            )

        age = float(now) - self._last.received_at
        if age > self.max_age_s:
            return ActuationStep(
                efforts=(0.0,) * len(self.effort_limits),
                mode=ActuationMode.PASSIVE,
                reason="lowcmd_stale",
                applied_sequence=self._last.sequence,
                lowcmd_age_s=age,
            )

        return ActuationStep(
            efforts=body_effort(
                self._last.q,
                self._last.dq,
                self._last.tau,
                self._last.kp,
                self._last.kd,
                q,
                dq,
                self.effort_limits,
            ),
            mode=ActuationMode.CONTROLLED,
            reason="applied",
            applied_sequence=self._last.sequence,
            lowcmd_age_s=age,
        )

    def invalidate(self, *, reason: str = "reset") -> None:
        """Drop the cached command so the body is passive until a fresh one."""
        self._last = None
        self.rejections.append(reason)

    def snapshot(self) -> dict:
        return {
            "mode": self.mode.value,
            "accepted": self._accepted,
            "rejected": self._rejected,
            "applied_sequence": None if self._last is None else self._last.sequence,
            "lowcmd": self.rate.snapshot(),
        }
