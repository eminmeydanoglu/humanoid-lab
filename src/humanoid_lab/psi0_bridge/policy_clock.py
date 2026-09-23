"""Strict wall/simulation clock selection for learned-policy runners."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path


class PolicyClockError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyTime:
    seconds: float
    generation: int


class PolicyClock:
    """Read an atomically replaced Isaac clock file without changing wall watchdogs."""

    def __init__(self, mode: str = "wall", path: Path | None = None, *, stale_timeout_s: float = 5.0) -> None:
        if mode not in ("wall", "simulation"):
            raise ValueError(f"unknown policy clock {mode!r}")
        if mode == "simulation" and path is None:
            raise ValueError("simulation policy clock requires a clock file")
        if stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be positive")
        self.mode = mode
        self.path = None if path is None else Path(path)
        self.stale_timeout_s = float(stale_timeout_s)
        self._last_value: float | None = None
        self._last_revision: object | None = None
        self._last_advance_wall = time.monotonic()
        self._generation = 0

    @classmethod
    def from_environment(cls) -> "PolicyClock":
        mode = os.environ.get("HUMANOID_POLICY_CLOCK", "wall")
        raw_path = os.environ.get("HUMANOID_POLICY_CLOCK_FILE")
        timeout = float(os.environ.get("HUMANOID_POLICY_CLOCK_TIMEOUT_S", "5.0"))
        return cls(mode, None if raw_path is None else Path(raw_path), stale_timeout_s=timeout)

    def now(self) -> PolicyTime:
        if self.mode == "wall":
            return PolicyTime(time.monotonic(), 0)
        value, revision = self._read_file()
        wall = time.monotonic()
        if self._last_value is not None and value < self._last_value - 1e-9:
            self._generation += 1
        if self._last_value is None or value != self._last_value or revision != self._last_revision:
            self._last_advance_wall = wall
        elif wall - self._last_advance_wall >= self.stale_timeout_s:
            raise PolicyClockError("Isaac simulation clock stopped advancing")
        self._last_value = value
        self._last_revision = revision
        return PolicyTime(value, self._generation)

    def latency_seconds(self, start: PolicyTime, end: PolicyTime | None = None) -> float:
        finish = self.now() if end is None else end
        if finish.generation != start.generation or finish.seconds < start.seconds:
            raise PolicyClockError("inference crossed an Isaac simulation reset")
        return finish.seconds - start.seconds

    def _read_file(self) -> tuple[float, object]:
        assert self.path is not None
        for _ in range(5):
            try:
                before = self.path.stat()
                raw = self.path.read_bytes()
                after = self.path.stat()
            except OSError as exc:
                raise PolicyClockError(f"Isaac simulation clock unavailable: {self.path}") from exc
            before_revision = (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
            revision = (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
            if before_revision == revision:
                break
        else:
            raise PolicyClockError("Isaac simulation clock changed repeatedly during read")
        try:
            value = float(raw.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise PolicyClockError("Isaac simulation clock is not a finite float") from exc
        if not math.isfinite(value) or value < 0:
            raise PolicyClockError("Isaac simulation clock is not a finite non-negative float")
        return value, revision
