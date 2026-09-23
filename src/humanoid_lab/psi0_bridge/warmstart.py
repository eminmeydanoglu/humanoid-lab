"""Opt-in demonstration-token warm start for the GR00T selection.

The evaluation stack starts GR00T from a robot the pre-policy settle put
wherever the *initial-pose latent* the VLA client sends happens to drive it.
Experiment 06 showed that the reset *state* is not a free variable -- the SONIC
deployment owns the limb through its decoder -- and left one route open: change
what the low-level controller is *commanded*, not what the robot *is*.

This module is that route, as an opt-in.  A stream of motion tokens recorded in
a real BlockStacking demonstration's start phase is published to the SONIC
action port through :class:`~humanoid_lab.psi0_bridge.action_router.ActionRouter`
while the settle runs, so:

* the deployment decodes them with the **live measured state** it reads from the
  simulator (the 994D observation is assembled inside the deployment, not here);
* its own history blocks (joint positions/velocities, last actions, gravity) and
  its decode recurrence evolve from that closed loop -- nothing is written into
  the robot and nothing is held;
* at ``Start`` the router switches source, so the GR00T token flow continues in
  the same controller instance, with the history the demonstration left behind.

Only the token stream is sent, exactly in the wire format the VLA client uses
(protocol v4 ``pose`` messages), so the deployment cannot tell the two apart.
The stream is replayed at the deployment's control rate; once it is exhausted
its last token is repeated, which is how the client's own initial-pose command
holds a pose (a static token, re-decoded every tick against the live state).

Default behaviour is untouched: without ``--warmstart-tokens`` the launcher
never builds this object and the session takes exactly the path it took before.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

TOKEN_DIM = 64
HAND_DIM = 7
#: The deployment's control rate; the demonstration is 30 Hz and is resampled
#: onto this grid by holding each token, exactly as the live bridge does.
DEFAULT_CONTROL_HZ = 50.0


class WarmStartError(RuntimeError):
    """The warm-start stream cannot be used; the message is meant for the operator."""


@dataclass(frozen=True)
class SimulationClockSample:
    """One stable observation from an external simulation clock."""

    sim_s: float
    revision: object


class SimulationClock(Protocol):
    """Clock source used by the opt-in simulation-paced publisher."""

    def read(self) -> SimulationClockSample | None:
        """Return the latest complete sample, or ``None`` while unavailable."""


class FileSimulationClock:
    """Read Isaac's float clock file without accepting a partial write."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> SimulationClockSample | None:
        try:
            before = self.path.stat()
            raw = self.path.read_bytes()
            after = self.path.stat()
        except OSError:
            return None
        revision = (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
        if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != revision:
            return None
        if not raw.endswith(b"\n"):
            return None
        try:
            value = float(raw.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError):
            return None
        if not np.isfinite(value) or value < 0.0:
            return None
        return SimulationClockSample(sim_s=value, revision=revision)


@dataclass(frozen=True)
class TokenStream:
    """A prepared token stream: 64-D motion tokens plus both 7-D hand commands.

    The arrays are already on the deployment's control grid (``control_hz``), so
    tick ``i`` is one action message.  ``source`` keeps whatever the preparation
    stage wrote about where the frames came from; it is recorded verbatim in the
    session telemetry and the manifest.
    """

    tokens: np.ndarray
    left_hand_joints: np.ndarray
    right_hand_joints: np.ndarray
    control_hz: float = DEFAULT_CONTROL_HZ
    source: dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None
    sha256: Optional[str] = None

    @property
    def ticks(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def duration_s(self) -> float:
        return self.ticks / float(self.control_hz)

    def summary(self) -> dict[str, Any]:
        return {
            "path": None if self.path is None else str(self.path),
            "sha256": self.sha256,
            "ticks": self.ticks,
            "duration_s": round(self.duration_s, 4),
            "control_hz": self.control_hz,
            "source": dict(self.source),
        }


def load_token_stream(path: Path) -> TokenStream:
    """Read and validate one prepared stream file.

    The file is the artifact the preparation stage wrote (``tokens`` /
    ``left_hand_joints`` / ``right_hand_joints`` at ``control_hz``); every shape,
    the finiteness of every value and a positive control rate are checked here,
    so a truncated or hand-edited file fails before a session starts rather than
    during a settle.
    """
    import hashlib

    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise WarmStartError(f"cannot read the warm-start token file {path}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WarmStartError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WarmStartError(f"{path} is not a JSON object")

    def array(key: str, width: int) -> np.ndarray:
        values = payload.get(key)
        if values is None:
            raise WarmStartError(f"{path} has no {key!r}")
        try:
            array_value = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise WarmStartError(f"{path}:{key} is not a numeric array: {exc}") from exc
        if array_value.ndim != 2 or array_value.shape[1] != width:
            raise WarmStartError(
                f"{path}:{key} must be a 2-D array of width {width}, got {array_value.shape}"
            )
        if array_value.shape[0] == 0:
            raise WarmStartError(f"{path}:{key} is empty")
        if not np.isfinite(array_value).all():
            raise WarmStartError(f"{path}:{key} contains a non-finite value")
        return array_value

    tokens = array("tokens", TOKEN_DIM)
    left = array("left_hand_joints", HAND_DIM)
    right = array("right_hand_joints", HAND_DIM)
    if not (tokens.shape[0] == left.shape[0] == right.shape[0]):
        raise WarmStartError(
            f"{path}: the three arrays disagree on length "
            f"({tokens.shape[0]}, {left.shape[0]}, {right.shape[0]})"
        )
    control_hz = float(payload.get("control_hz", DEFAULT_CONTROL_HZ))
    if not (control_hz > 0.0) or not np.isfinite(control_hz):
        raise WarmStartError(f"{path}: control_hz must be a positive number, got {control_hz!r}")
    source = payload.get("source")
    if source is not None and not isinstance(source, dict):
        raise WarmStartError(f"{path}:source must be an object when present")
    return TokenStream(
        tokens=tokens, left_hand_joints=left, right_hand_joints=right,
        control_hz=control_hz, source=dict(source or {}), path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def default_packer() -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], bytes]:
    """The VLA client's own protocol v4 ``pose`` packer.

    Imported lazily: it lives in the pinned SONIC tree, which only the container
    has, and a session that does not use the warm start must not need it.
    """

    def pack(token: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray, index: int) -> bytes:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

        return pack_pose_message(
            {
                "token_state": np.asarray(token, dtype=np.float32).reshape(1, TOKEN_DIM),
                "frame_index": np.array([int(index)], dtype=np.int64),
                "left_hand_joints": np.asarray(left_hand, dtype=np.float32).reshape(1, HAND_DIM),
                "right_hand_joints": np.asarray(right_hand, dtype=np.float32).reshape(1, HAND_DIM),
            },
            topic="pose", version=4,
        )

    return pack


class WarmStartStream:
    """Publish one prepared token stream through the router, then hold its end.

    ``arm(delay_s)`` schedules the stream to begin ``delay_s`` after the call (the
    settle's own tail, so the canonical part of the settle is untouched);
    ``halt()`` stops it and is called on every session transition.  The router
    remains the only owner of the SONIC action port and the only place a source
    switch happens, so the hand-off at ``Start`` is the same atomic generation
    change the PSI/GR00T switch already uses.
    """

    def __init__(
        self,
        stream: TokenStream,
        router: Any,
        *,
        telemetry: Any | None = None,
        packer: Callable[[np.ndarray, np.ndarray, np.ndarray, int], bytes] | None = None,
        source: str = "warmstart",
        publish: Callable[[bytes], bool] | None = None,
        simulation_clock: SimulationClock | None = None,
        clock_timeout_s: float = 5.0,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.stream = stream
        self.router = router
        self.telemetry = telemetry
        #: The router source this stream feeds and the telemetry kind its own
        #: lifecycle events use.  ``warmstart`` is the demonstration-token
        #: warm start; the initial-pose handshake passes its own label so a
        #: recorded message says which of the two published it.
        self.source = source
        self._publish = publish if publish is not None else router.submit_warmstart
        self._packer = packer
        self._simulation_clock = simulation_clock
        self._clock_timeout_s = float(clock_timeout_s)
        if self._simulation_clock is not None and self._clock_timeout_s <= 0.0:
            raise WarmStartError("clock_timeout_s must be positive in simulation-clock mode")
        self._log = log
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sent = 0
        self._started_at: float | None = None
        self._halted_at: float | None = None
        self._delay_s = 0.0
        self._clock_baseline: SimulationClockSample | None = None
        self._error: str | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def armed(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop.is_set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "armed": self.armed, "sent": self._sent, "delay_s": self._delay_s,
                "started_at": self._started_at, "halted_at": self._halted_at,
                "error": self._error,
                "clock_mode": "simulation" if self._simulation_clock is not None else "wall",
                "clock_timeout_s": self._clock_timeout_s if self._simulation_clock is not None else None,
                **self.stream.summary(),
            }

    def clock_sample(self) -> SimulationClockSample | None:
        """Capture one stable clock sample before an externally validated reset."""
        return None if self._simulation_clock is None else self._simulation_clock.read()

    def arm(
        self,
        delay_s: float = 0.0,
        *,
        clock_baseline: SimulationClockSample | None = None,
    ) -> None:
        """(Re)schedule the stream; an already running stream is replaced."""
        self.halt()
        delay_s = max(0.0, float(delay_s))
        self._stop = threading.Event()
        self._delay_s = delay_s
        self._clock_baseline = clock_baseline
        self._error = None
        thread = threading.Thread(target=self._serve, name="warmstart-stream", daemon=True)
        with self._lock:
            self._thread = thread
            self._started_at = None
            self._halted_at = None
        payload = {**self.stream.summary(), "state": "armed", "delay_s": delay_s}
        self._event(self.source, **payload)
        thread.start()
        self._log(
            f"[{self.source}] armed: {self.stream.ticks} ticks at {self.stream.control_hz:g} Hz "
            f"({self.stream.duration_s:.2f}s), starting in {delay_s:.2f}s"
        )

    def halt(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        if thread.is_alive():
            thread.join(timeout=max(2.0, 2.0 / float(self.stream.control_hz)))
        with self._lock:
            self._thread = None
            self._halted_at = time.time()
        self._event(
            self.source, state="halted", sent=self._sent, **self.stream.summary()
        )
        self._log(f"[{self.source}] halted after {self._sent} ticks")

    # -- publisher ---------------------------------------------------------

    def _serve(self) -> None:
        if self._packer is None:
            try:
                self._packer = default_packer()
            except Exception as exc:  # noqa: BLE001 - reported, never raised into the session
                with self._lock:
                    self._error = f"cannot import the protocol v4 packer: {type(exc).__name__}: {exc}"
                self._event(self.source, state="error", error=self._error)
                return
        if self._simulation_clock is None:
            self._serve_wall()
        else:
            self._serve_simulation_clock()

    def _publish_index(self, index: int) -> None:
        position = min(index, self.stream.ticks - 1)
        payload = self._packer(
            self.stream.tokens[position],
            self.stream.left_hand_joints[position],
            self.stream.right_hand_joints[position],
            index,
        )
        if self._publish(payload):
            self._sent += 1

    def _serve_wall(self) -> None:
        if self._stop.wait(max(0.0, self._delay_s)):
            return
        period = 1.0 / float(self.stream.control_hz)
        index = 0
        next_tick = time.monotonic()
        self._event(self.source, state="started", clock_mode="wall", **self.stream.summary())
        while not self._stop.is_set():
            self._publish_index(index)
            index += 1
            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                self._stop.wait(remaining)
            else:
                next_tick = time.monotonic()

    def _serve_simulation_clock(self) -> None:
        assert self._simulation_clock is not None
        baseline = self._clock_baseline
        if baseline is None:
            baseline = self._simulation_clock.read()
        baseline_revision = None if baseline is None else baseline.revision
        fresh_deadline = time.monotonic() + self._clock_timeout_s
        start: SimulationClockSample | None = None
        while not self._stop.is_set() and time.monotonic() < fresh_deadline:
            sample = self._simulation_clock.read()
            revised = sample is not None and sample.revision != baseline_revision
            reset_clock = (
                baseline is None or baseline.sim_s <= 0.1
                or (sample is not None and sample.sim_s < baseline.sim_s - 1e-9)
            )
            if revised and reset_clock:
                start = sample
                break
            self._stop.wait(0.005)
        if self._stop.is_set():
            return
        if start is None:
            self._clock_error("no fresh post-reset Isaac simulation clock sample")
            return

        origin_sim_s = start.sim_s
        last = start
        last_advance_wall = time.monotonic()
        index = 0
        self._event(
            self.source, state="started", clock_mode="simulation",
            origin_sim_s=origin_sim_s, **self.stream.summary(),
        )
        while not self._stop.is_set():
            sample = self._simulation_clock.read()
            now = time.monotonic()
            if sample is None:
                if now - last_advance_wall >= self._clock_timeout_s:
                    self._clock_error("Isaac simulation clock became unavailable or stale")
                    return
                self._stop.wait(0.005)
                continue
            if sample.sim_s < last.sim_s - 1e-9:
                self._clock_error(
                    f"Isaac simulation clock moved backwards ({last.sim_s:.6f} -> {sample.sim_s:.6f})"
                )
                return
            if sample.sim_s > last.sim_s + 1e-9:
                last_advance_wall = now
                last = sample
            elif now - last_advance_wall >= self._clock_timeout_s:
                self._clock_error("Isaac simulation clock stopped advancing")
                return

            elapsed = sample.sim_s - origin_sim_s
            due = int(np.floor(max(0.0, elapsed - self._delay_s) * self.stream.control_hz + 1e-9))
            if elapsed + 1e-9 >= self._delay_s:
                while index <= due and not self._stop.is_set():
                    self._publish_index(index)
                    index += 1
            self._stop.wait(0.005)

    def _clock_error(self, message: str) -> None:
        with self._lock:
            self._error = message
        self._event(self.source, state="error", error=message, sent=self._sent)
        self._log(f"[{self.source}] error: {message}")

    def _event(self, kind: str, **fields: Any) -> None:
        if self.telemetry is None:
            return
        try:
            self.telemetry.event(kind, **fields)
        except Exception:  # noqa: BLE001 - recording must never break the action path
            pass


def stream_from_tokens(
    tokens: Sequence[Sequence[float]],
    left_hand: Sequence[Sequence[float]],
    right_hand: Sequence[Sequence[float]],
    *,
    control_hz: float = DEFAULT_CONTROL_HZ,
    source: dict[str, Any] | None = None,
) -> TokenStream:
    """Build an in-memory stream (used by tests and by the preparation stage)."""
    return TokenStream(
        tokens=np.asarray(tokens, dtype=np.float64),
        left_hand_joints=np.asarray(left_hand, dtype=np.float64),
        right_hand_joints=np.asarray(right_hand, dtype=np.float64),
        control_hz=float(control_hz),
        source=dict(source or {}),
    )
