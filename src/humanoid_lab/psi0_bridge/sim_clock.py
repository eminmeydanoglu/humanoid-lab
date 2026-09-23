"""Simulation-time action progression for learned-policy deployment."""

from __future__ import annotations

import math
from dataclasses import dataclass


class SimulationClockError(RuntimeError):
    """The simulator clock cannot safely drive action progression."""


@dataclass(frozen=True)
class ClockUpdate:
    rows_due: int
    reset: bool
    stalled: bool


class SimulationClockPacer:
    """Convert one monotonic simulator clock into action-row progression.

    The caller supplies the latest simulator timestamp before each publish
    opportunity. ``rows_due`` is the number of dataset-rate boundaries crossed
    since the previous update; when it is greater than one, stale intermediate
    rows must be skipped rather than published as a wall-clock burst. A changed
    episode id or backwards timestamp starts a new timeline and never carries
    elapsed time across the reset.
    """

    def __init__(self, rate_hz: float) -> None:
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("rate_hz must be finite and positive")
        self.rate_hz = float(rate_hz)
        self._episode_id: int | None = None
        self._origin_s: float | None = None
        self._emitted = 0
        self._last_s: float | None = None

    def reset(self) -> None:
        self._episode_id = None
        self._origin_s = None
        self._emitted = 0
        self._last_s = None

    def update(self, sim_s: float, episode_id: int) -> ClockUpdate:
        sim_s = float(sim_s)
        episode_id = int(episode_id)
        if not math.isfinite(sim_s):
            raise SimulationClockError("simulator timestamp must be finite")

        reset = self._origin_s is not None and (
            self._episode_id != episode_id
            or (self._last_s is not None and sim_s < self._last_s)
        )
        if self._origin_s is None or reset:
            self._episode_id = episode_id
            self._origin_s = sim_s
            self._last_s = sim_s
            self._emitted = 1
            return ClockUpdate(rows_due=1, reset=reset, stalled=False)

        stalled = sim_s == self._last_s
        self._last_s = sim_s
        elapsed = max(0.0, sim_s - self._origin_s)
        target_emitted = int(math.floor(elapsed * self.rate_hz + 1e-9)) + 1
        due = max(0, target_emitted - self._emitted)
        self._emitted = target_emitted
        return ClockUpdate(rows_due=due, reset=False, stalled=stalled)

    def latency_rows(self, inference_start_s: float, inference_end_s: float, horizon: int) -> int:
        """Map inference latency to a chunk index on the simulator timeline."""
        start = float(inference_start_s)
        end = float(inference_end_s)
        if not math.isfinite(start) or not math.isfinite(end):
            raise SimulationClockError("inference timestamps must be finite")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if end < start:
            raise SimulationClockError("inference crossed a simulator reset")
        return min(int(math.floor((end - start) * self.rate_hz + 0.5)), horizon - 1)


def clamp_rtc_delay(rows: int, max_delay: int) -> int:
    """Bound one RTC delay to the horizon the checkpoint was trained with.

    ``rows`` is how many control rows elapsed while a replan ran; ``max_delay``
    is the ``model.max_delay`` the served run declares.  The two are not
    interchangeable: the trained RTC path only ever saw delays up to
    ``max_delay``, so handing it a larger value freezes more of the new chunk
    against a stale prefix than the weights were trained for.  A simulation
    clock makes that reachable, because a multi-second forward pass converts to
    a delay far above the bound (and is then capped by the chunk length, which
    is larger still).  The bound is applied here rather than inside the server so
    the rule is testable without the model.
    """
    if max_delay <= 0:
        raise ValueError("max_delay must be positive")
    if rows < 0:
        raise ValueError("rows must be non-negative")
    return min(int(rows), int(max_delay))
