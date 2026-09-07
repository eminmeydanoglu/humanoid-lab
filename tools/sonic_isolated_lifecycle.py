"""Fail-closed lifecycle used by the isolated SONIC native harness."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

ACTION_RATE_HZ = 50.0
INFERENCE_RATE_HZ = 2.5
TOKEN_SIZE = 64
HAND_SIZE = 7
BODY_OUTPUT_SIZE = 29


class LifecycleError(ValueError):
    """Raised when an isolated lifecycle transition is unsafe."""


class State(Enum):
    RESET = "reset"
    HOLDING = "holding"
    RUNNING = "running"
    PAUSED = "paused"
    TIMED_OUT = "timed_out"
    STOPPED = "stopped"


@dataclass
class IsolatedLifecycle:
    watchdog_seconds: float = 0.25
    state: State = State.RESET
    reference: tuple[float, ...] | None = None
    last_validated_at: float | None = None

    def initialize(self, reference: Sequence[float]) -> None:
        values = tuple(float(value) for value in reference)
        if self.state is not State.RESET or len(values) != TOKEN_SIZE or not any(values):
            raise LifecycleError("initial hold requires a nonzero 64D reference from RESET")
        self.reference = values
        self.state = State.HOLDING

    def start(self, now: float) -> None:
        if self.state not in (State.HOLDING, State.PAUSED) or self.reference is None:
            raise LifecycleError("start requires a validated holding reference")
        self.state, self.last_validated_at = State.RUNNING, now

    def validate_action(self, motion_token: Sequence[float], left_hand: Sequence[float], right_hand: Sequence[float], now: float) -> None:
        if self.tick(now) is not State.RUNNING:
            raise LifecycleError("action accepted only while running")
        if len(motion_token) != TOKEN_SIZE or len(left_hand) != HAND_SIZE or len(right_hand) != HAND_SIZE:
            raise LifecycleError("protocol action dimensions must be 64+7+7")
        self.last_validated_at = now

    def pause(self) -> None:
        if self.state is not State.RUNNING:
            raise LifecycleError("pause requires RUNNING")
        self.state = State.PAUSED

    def tick(self, now: float) -> State:
        if self.state is State.RUNNING and self.last_validated_at is not None and now - self.last_validated_at > self.watchdog_seconds:
            self.state = State.TIMED_OUT
        return self.state

    def hold(self, now: float) -> tuple[float, ...]:
        if self.tick(now) not in (State.HOLDING, State.PAUSED, State.TIMED_OUT) or self.reference is None:
            raise LifecycleError("hold is unavailable in current state")
        return self.reference

    def stop(self) -> None:
        if self.state not in (State.HOLDING, State.RUNNING, State.PAUSED, State.TIMED_OUT):
            raise LifecycleError("stop requires an initialized lifecycle")
        self.state = State.STOPPED

    def reset(self) -> None:
        if self.state is State.RUNNING:
            raise LifecycleError("stop or pause before reset")
        self.state, self.reference, self.last_validated_at = State.RESET, None, None
