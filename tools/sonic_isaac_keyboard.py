"""Keyboard driving for the upstream SONIC deploy's ``keyboard`` input mode.

The deploy expects a real TTY.  These schedules are the keystrokes an operator
would send, expressed as timed events so the automated gates can replay them.

Bindings are the pinned upstream ones (``keyboard_handler.hpp``):

    ]        start control
    Enter    toggle planner mode
    W/S      forward / backward          (planner mode)
    A/D      adjust left / right         (planner mode)
    R or `   emergency momentum reset -> IDLE
    O        emergency stop
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ARM_KEY",
    "DRIVE_SCHEDULE",
    "IDLE_SPEED_MPS",
    "MOMENTUM_RESET_KEY",
    "PLANNER_TOGGLE_KEY",
    "STOP_KEY",
    "KeyEvent",
    "KeyScheduler",
    "STANDING_SCHEDULE",
    "drive_schedule",
    "presses_during",
    "standing_schedule",
]

ARM_KEY = "]"
PLANNER_TOGGLE_KEY = "\n"
FORWARD_KEY = "w"
TURN_LEFT_KEY = "a"
TURN_RIGHT_KEY = "d"
MOMENTUM_RESET_KEY = "r"
STOP_KEY = "o"

#: Forward command is held by repeating the key; a plane motion below this is
#: treated as "Idle" for the release-decay acceptance.
IDLE_SPEED_MPS = 0.05


@dataclass(frozen=True)
class KeyEvent:
    at_s: float
    keys: str
    label: str


#: Gate C/D: arm the controller and leave it standing.
STANDING_SCHEDULE: tuple[KeyEvent, ...] = (
    KeyEvent(0.0, ARM_KEY, "arm"),
)


def drive_schedule(
    *, idle_s: float = 10.0, forward_s: float = 3.0, release_s: float = 5.0,
    turn_s: float = 3.0, settle_s: float = 5.0, turn_key: str = TURN_LEFT_KEY,
    hold_hz: float = 20.0,
) -> tuple[KeyEvent, ...]:
    """The plan's Gate E sequence, timings relative to arming."""
    events: list[KeyEvent] = [KeyEvent(0.0, ARM_KEY, "arm")]
    if idle_s > 0:
        events.append(KeyEvent(idle_s, PLANNER_TOGGLE_KEY, "planner_mode"))
    forward_at = idle_s + 0.5
    events.extend(presses_during(FORWARD_KEY, forward_at, forward_s, hold_hz, "forward"))
    release_at = forward_at + forward_s + release_s
    events.extend(presses_during(turn_key, release_at, turn_s, hold_hz, "turn"))
    reset_at = release_at + turn_s + settle_s
    events.append(KeyEvent(reset_at, MOMENTUM_RESET_KEY, "momentum_reset"))
    events.append(KeyEvent(reset_at + settle_s, STOP_KEY, "stop"))
    return tuple(events)


def presses_during(
    key: str, start_s: float, duration_s: float, hold_hz: float, label: str
) -> list[KeyEvent]:
    """Repeat ``key`` like a held terminal key for ``duration_s``."""
    period = 1.0 / hold_hz
    count = max(1, int(round(duration_s / period)))
    return [KeyEvent(start_s + index * period, key, label) for index in range(count)]


def standing_schedule() -> tuple[KeyEvent, ...]:
    return STANDING_SCHEDULE


class KeyScheduler:
    """Emits the keys due at a given elapsed time, in schedule order."""

    def __init__(self, events: tuple[KeyEvent, ...]) -> None:
        self.events = tuple(sorted(events, key=lambda event: event.at_s))
        self._next = 0

    @property
    def duration_s(self) -> float:
        return max((event.at_s for event in self.events), default=0.0)

    def due(self, elapsed_s: float) -> list[KeyEvent]:
        ready: list[KeyEvent] = []
        while self._next < len(self.events) and self.events[self._next].at_s <= elapsed_s:
            ready.append(self.events[self._next])
            self._next += 1
        return ready

    @property
    def finished(self) -> bool:
        return self._next >= len(self.events)
