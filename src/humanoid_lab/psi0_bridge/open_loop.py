"""Unguided chunk execution for the psi0 SONIC deployment wrapper."""

from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np


class OpenLoopChunkController:
    """Execute independent chunks and replan after a fixed action horizon.

    The first chunk is predicted synchronously. Exactly ``execution_horizon``
    rows are emitted before the next independent prediction is requested. While
    that prediction is pending, the last executed row is held. A completed
    prediction is installed at row zero; no previous action is passed to it.
    """

    def __init__(
        self,
        predict: Callable[[dict[str, Any]], np.ndarray],
        initial_obs: dict[str, Any],
        *,
        execution_horizon: int,
    ) -> None:
        self._predict = predict
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._latest_obs = initial_obs
        self._chunk = self._validate_chunk(predict(initial_obs))
        if execution_horizon <= 0 or execution_horizon > len(self._chunk):
            raise ValueError(
                f"execution_horizon {execution_horizon} is not in (0, {len(self._chunk)}]"
            )
        self.execution_horizon = int(execution_horizon)
        self._index = 0
        self._pending: np.ndarray | None = None
        self._inference_error: BaseException | None = None
        self._inflight = False
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _validate_chunk(chunk: np.ndarray) -> np.ndarray:
        value = np.asarray(chunk, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] == 0:
            raise ValueError(f"predicted action chunk must be [H, D], got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError("predicted action chunk contains NaN or Inf")
        return value

    def step(self, obs_next: dict[str, Any]) -> np.ndarray:
        with self._lock:
            self._latest_obs = obs_next
            if self._inference_error is not None:
                error = self._inference_error
                self._inference_error = None
                raise RuntimeError("open-loop action inference failed") from error
            if self._pending is not None:
                self._chunk = self._pending
                self._pending = None
                self._index = 0
                self._inflight = False

            if self._index < self.execution_horizon:
                row = self._chunk[self._index]
                self._index += 1
                if self._index == self.execution_horizon and not self._inflight:
                    self._inflight = True
                    self._wake.set()
            else:
                row = self._chunk[self.execution_horizon - 1]
            return row[np.newaxis, :]

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=join_timeout)

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                return
            with self._lock:
                observation = self._latest_obs
            try:
                chunk = self._validate_chunk(self._predict(observation))
                if self.execution_horizon > len(chunk):
                    raise ValueError(
                        f"execution_horizon {self.execution_horizon} exceeds new chunk {len(chunk)}"
                    )
            except Exception as exc:
                with self._lock:
                    self._inflight = False
                    self._inference_error = exc
                continue
            with self._lock:
                if not self._stop.is_set():
                    self._pending = chunk
