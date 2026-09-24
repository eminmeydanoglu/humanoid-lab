"""The bridge's one control loop and its four-state session machine.

States are exactly ``IDLE``, ``RUNNING``, ``STOPPED`` and ``ERROR``:

* ``Start`` is accepted only from ``IDLE``/``STOPPED`` and only once the state
  source and camera both have a fresh sample; it then connects the WebSocket,
  binds the ``pose`` publisher and streams observations.  Every action must have
  the width the served ``/info`` declared, otherwise the session fails closed.
* ``Stop`` clears the send gate immediately and joins the loop, so no further
  action is published even if the socket is mid-receive.
* ``Reset`` stops first, waits out the SONIC command TTL, asks the Isaac control
  endpoint to restore the scene, then drops the monitor's cached state/frame and
  the action sequence, so the next ``Start`` needs a fresh observation.
* A connection or schema failure moves to ``ERROR``, halts publishing, and
  never repeats the last action.  ``ERROR`` requires a ``Reset`` before the next
  ``Start``.

The loop reads cached snapshots from :class:`~humanoid_lab.psi0_bridge.monitor.Monitor`
instead of opening ZMQ sockets, so exactly one thread owns each read endpoint
and the frame the policy consumes is the frame the UI previews.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from .actions import ActionAdapter
from .camera import CAMERA_ENDPOINT
from .contracts import NECK_NOOP_TOLERANCE, ServerInfo
from .monitor import Monitor
from .policy_clock import PolicyClock, PolicyClockError
from .prompt import CANONICAL_PROMPT
from .sim_clock import SimulationClockPacer
from .publisher import PublisherError, PosePublisher
from .psi0_client import (
    DEFAULT_WS_URL,
    ActionReply,
    Psi0ClientError,
    Psi0Connection,
    decode_response,
    fetch_info,
    serialize_request,
)
from .publisher import ACTION_ENDPOINT, PosePublisher
from .reset_client import DEFAULT_RESET_ENDPOINT, IsaacResetClient, ResetError
from .state_source import STATE_ENDPOINT, STATE_TOPIC

IDLE = "IDLE"
RUNNING = "RUNNING"
STOPPED = "STOPPED"
ERROR = "ERROR"

DEFAULT_CONTROL_HZ = 30.0
DEFAULT_RECV_TIMEOUT_S = 60.0  # the first action absorbs the server-side model build
DEFAULT_READY_TIMEOUT_S = 10.0
DEFAULT_START_TIMEOUT_S = 8.0
DEFAULT_STOP_TIMEOUT_S = 3.0

#: A snapshot older than this is treated as "not ready" / "stale" and is not
#: sent.  The stream is 30 Hz, the camera 25 Hz (40 ms) and g1_debug 200 Hz, so a
#: fresh sample is at most one monitor poll (one camera round trip) old; half a
#: second leaves room for a slow poll without ever feeding the policy a frame
#: from an episode the robot no longer occupies.
STATE_MAX_AGE_S = 0.5
FRAME_MAX_AGE_S = 0.5

STALE_FLUSH_S = 0.005
POLL_SLICE_S = 0.5


class SessionError(RuntimeError):
    """A session transition was refused."""


@dataclass(frozen=True)
class SessionConfig:
    ws_url: str = DEFAULT_WS_URL
    state_endpoint: str = STATE_ENDPOINT
    state_topic: str = STATE_TOPIC
    camera_endpoint: str = CAMERA_ENDPOINT
    action_endpoint: str = ACTION_ENDPOINT
    reset_endpoint: str = DEFAULT_RESET_ENDPOINT
    instruction: str = CANONICAL_PROMPT
    control_hz: float = DEFAULT_CONTROL_HZ
    recv_timeout_s: float = DEFAULT_RECV_TIMEOUT_S
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S
    start_timeout_s: float = DEFAULT_START_TIMEOUT_S
    stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S
    reset_timeout_ms: int = 2000
    #: Wait between Stop and the Isaac reset request, so the SONIC controller's
    #: command TTL (``controller.command_ttl_s``, 0.25s in the profile) expires
    #: and no already-published pose can land mid-reset.
    command_ttl_s: float = 0.3
    neck_tolerance: float = NECK_NOOP_TOLERANCE
    #: ``"error"`` (default) fails closed on an 80D neck block outside the no-op
    #: tolerance.  ``"discard"`` is the explicit opt-in that drops it, for a pack
    #: whose neck columns are unsupervised padding (this repo's own 80D pack);
    #: every drop is recorded through telemetry.
    neck_policy: str = "error"


def _neck_discard_fields(adapter: ActionAdapter) -> dict[str, Any]:
    """Telemetry fields describing a neck block the adapter had to drop.

    Empty for the ordinary case (78D action, or an in-tolerance no-op), so the
    existing ``target_action`` schema is unchanged unless a discard happened.
    """
    adapted = adapter.last_adapted
    if adapted is None or adapted.discarded_neck is None:
        return {}
    return {
        "neck_discarded": True,
        "neck_discarded_values": [float(value) for value in adapted.discarded_neck.tolist()],
        "neck_discard_count": adapter.neck_discards,
    }


def info_summary(info: Optional[ServerInfo]) -> Optional[dict[str, Any]]:
    if info is None:
        return None
    return {
        "run_dir": info.run_dir,
        "action_dim": info.action_dim,
        "action_chunk_size": info.action_chunk_size,
        "action_exec_horizon": info.action_exec_horizon,
        "state_dim": info.state_dim,
        "history_length": info.history_length,
        "image_key": info.image_key,
        "resize": list(info.resize_size),
        "center_crop": list(info.center_crop_size),
        "normalize_state": info.normalize_state,
        "rtc_enabled": info.rtc_enabled,
        "ckpt_step": info.ckpt_step,
        "dataset_name": info.dataset_name,
    }


class Session:
    """Owns the lifecycle of one Ψ₀ -> Protocol v4 action stream."""

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        *,
        monitor: Optional[Monitor] = None,
        publisher: Any | None = None,
        telemetry: Any | None = None,
        policy_clock: PolicyClock | None = None,
    ) -> None:
        self.config = config or SessionConfig()
        #: Opt-in rollout recorder (see :mod:`humanoid_lab.psi0_bridge.telemetry`).
        #: ``None`` -- the default and every unit test -- records nothing.
        self.telemetry = telemetry
        self.policy_clock = policy_clock or PolicyClock("wall")
        self.monitor = monitor if monitor is not None else Monitor(
            state_endpoint=self.config.state_endpoint,
            state_topic=self.config.state_topic,
            camera_endpoint=self.config.camera_endpoint,
            poll_hz=self.config.control_hz,
            idle_poll_hz=5.0,
        )

        self._lock = threading.Lock()
        self._state = IDLE
        self._error: Optional[str] = None
        self._starting = False
        self._generation = 0
        self._thread: Optional[threading.Thread] = None

        self._stop_event = threading.Event()
        self._send_gate = threading.Event()
        self._ready_event = threading.Event()

        self._info: Optional[ServerInfo] = None
        self._ws_connected = False
        self._last_action_time: Optional[float] = None
        self._last_action_index: Optional[int] = None
        self._sent_count = 0
        self._neck_discard_count = 0

        self._policy_frame: Optional[np.ndarray] = None
        self._policy_frame_time: Optional[float] = None

        # Own the action socket for the whole service lifetime: a busy port must
        # fail the UI service at startup (not fail later at Start), and Stop must
        # only close the send gate, never release the socket.
        self._owns_publisher = publisher is None
        try:
            self._publisher = publisher or PosePublisher(self.config.action_endpoint)
        except PublisherError as exc:
            raise SessionError(f"action socket: {exc}") from exc

    # -- description -------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def info(self) -> Optional[ServerInfo]:
        with self._lock:
            return self._info

    def refresh_info(self) -> ServerInfo:
        """GET ``/info`` and validate it; safe to call outside the loop thread."""
        info = fetch_info(self.config.ws_url)
        with self._lock:
            self._info = info
        return info

    def adopt_policy(self, info: ServerInfo) -> None:
        """Record a freshly verified policy identity after a checkpoint switch.

        The caller has already stopped the session and verified ``/info`` for the
        new checkpoint; a switch resolves a previous policy-link error, so ERROR
        clears and the session becomes ready (IDLE) again.
        """
        with self._lock:
            self._info = info
            self._error = None
            self._starting = False
            if self._state != RUNNING:
                self._state = IDLE

    def mark_error(self, message: str) -> dict[str, Any]:
        """Fail closed from outside the loop thread (refused/failed switch).

        Same guarantees as a stream failure: the send gate stays closed, the
        publisher is halted and the session can only be restarted after a Reset.
        """
        with self._lock:
            self._state = ERROR
            self._error = message
            self._starting = False
            self._ws_connected = False
        self._send_gate.clear()
        self._publisher.halt()
        self._stop_event.set()
        self._ready_event.set()
        return self.status()

    def preview_frame(self) -> Optional[tuple[np.ndarray, float]]:
        """The frame most recently handed to the policy, else the latched camera frame."""
        with self._lock:
            frame = self._policy_frame
            timestamp = self._policy_frame_time
        if frame is not None and timestamp is not None:
            return frame, timestamp
        snapshot = self.monitor.frame()
        if snapshot is None:
            return None
        return snapshot.frame, snapshot.timestamp_s

    # -- transitions -------------------------------------------------------

    def start(self) -> dict[str, Any]:
        """Begin streaming; refuses unless the session is idle/stopped and ready."""
        with self._lock:
            if self._starting or self._state == RUNNING:
                return self.status()
            if self._state == ERROR:
                raise SessionError("session is in ERROR; Reset before Start")
            self._starting = True
            self._error = None
            self._generation += 1
            generation = self._generation

        self._ready_event.clear()
        self._stop_event.clear()
        self._send_gate.set()
        self._publisher.resume()  # the socket is owned since construction; Start opens the gate
        self._note_transition("STARTING", instruction=self.config.instruction)

        thread = threading.Thread(target=self._run_thread, args=(generation,), name="psi0-session", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()
        self.monitor.set_active(True)
        self._ready_event.wait(timeout=self.config.start_timeout_s)
        return self.status()

    def stop(self) -> dict[str, Any]:
        """Halt publishing immediately, then join the loop.

        Only the send gate closes: the action socket stays owned by the service
        so a later Start (or another pilot trying to bind :5556) cannot race it.
        """
        self._send_gate.clear()
        self._publisher.halt()
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.config.stop_timeout_s)
        self.monitor.set_active(False)
        with self._lock:
            if self._state != ERROR and (self._starting or self._state == RUNNING):
                self._state = STOPPED
            self._starting = False
            self._thread = None
            self._ws_connected = False
        self._note_transition(self._state)
        return self.status()

    def close(self) -> None:
        """Stop the session and release its output when this session owns it."""
        self.stop()
        if self._owns_publisher:
            self._publisher.close()

    def reset(self) -> dict[str, Any]:
        """Stop, let the command TTL expire, restore the scene, and clear all history.

        The monitor's cached state and frame are dropped *after* the reset is
        requested, so a Start can never stream an observation captured before
        the robot and scene moved back.
        """
        self.stop()
        if self.config.command_ttl_s > 0:
            # Stop() already joined the loop; this only gives the SONIC controller
            # time to age out the last pose it received before the scene moves.
            time.sleep(self.config.command_ttl_s)
        reset_error: Optional[str] = None
        try:
            IsaacResetClient(
                self.config.reset_endpoint, timeout_ms=self.config.reset_timeout_ms
            ).request_reset()
        except ResetError as exc:
            reset_error = str(exc)
        self.monitor.invalidate()

        with self._lock:
            self._generation += 1  # any late write from the stopped loop is now stale
            self._last_action_time = None
            self._last_action_index = None
            self._sent_count = 0
            self._neck_discard_count = 0
            self._policy_frame = None
            self._policy_frame_time = None
            self._error = reset_error
            self._state = ERROR if reset_error else IDLE
            self._starting = False
        self._note_transition(self._state, reset=True, error=reset_error)
        return self.status()

    # -- status ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            error = self._error
            starting = self._starting
            info = self._info
            connected = self._ws_connected
            last_action_time = self._last_action_time
            last_index = self._last_action_index
            sent = self._sent_count
            neck_discards = self._neck_discard_count

        monitor_status = self.monitor.status() if self.monitor is not None else {}
        now = time.time()
        return {
            "state": state,
            "error": error,
            "starting": starting,
            "psi0": {
                "url": self.config.ws_url,
                "connected": bool(connected and state == RUNNING),
                "info": info_summary(info),
                "policy_clock": self.policy_clock.mode,
            },
            "sonic_state": monitor_status.get("state", {}),
            "camera": monitor_status.get("camera", {}),
            "action": {
                "endpoint": self.config.action_endpoint,
                "last_time": None if last_action_time is None else _iso(last_action_time),
                "last_age_s": None if last_action_time is None else now - last_action_time,
                "last_index": last_index,
                "sent": sent,
                "neck_padding_discards": neck_discards,
            },
        }

    # -- loop --------------------------------------------------------------

    def _run_thread(self, generation: int) -> None:
        try:
            asyncio.run(self._run(generation))
        except BaseException as exc:  # noqa: BLE001 - the thread must report any failure
            self._fail(generation, f"{type(exc).__name__}: {exc}")

    async def _run(self, generation: int) -> None:
        try:
            info = self.refresh_info()
        except Exception as exc:  # Psi0ClientError / ContractError
            self._fail(generation, str(exc))
            return
        if not await self._wait_ready(generation):
            return
        try:
            async with Psi0Connection(self.config.ws_url) as connection:
                if not self._is_current(generation):
                    return
                adapter = ActionAdapter(
                    self.config.neck_tolerance,
                    expected_dim=info.action_dim,
                    neck_policy=self.config.neck_policy,
                )
                with self._lock:
                    if generation != self._generation:
                        return
                    self._ws_connected = True
                    self._state = RUNNING
                    self._starting = False
                self._ready_event.set()
                self._note_transition(
                    RUNNING,
                    action_dim=info.action_dim,
                    chunk_size=info.action_chunk_size,
                    state_dim=info.state_dim,
                    dataset_name=info.dataset_name,
                    run_dir=info.run_dir,
                    ckpt_step=info.ckpt_step,
                    policy_clock=self.policy_clock.mode,
                    neck_policy=self.config.neck_policy,
                )
                try:
                    # The socket is owned since construction; the stream only
                    # publishes through it while the send gate is open.
                    await self._stream(generation, connection, adapter)
                finally:
                    with self._lock:
                        self._ws_connected = False
        except Psi0ClientError as exc:
            self._fail(generation, str(exc))
        except Exception as exc:  # noqa: BLE001 - schema/transport failure stops the stream
            self._fail(generation, f"{type(exc).__name__}: {exc}")

    async def _wait_ready(self, generation: int) -> bool:
        deadline = time.monotonic() + self.config.ready_timeout_s
        while time.monotonic() < deadline:
            if not self._is_current(generation):
                return False
            if (
                self.monitor.state(max_age_s=STATE_MAX_AGE_S) is not None
                and self.monitor.frame(max_age_s=FRAME_MAX_AGE_S) is not None
            ):
                return True
            await asyncio.sleep(0.1)
        self._fail(generation, "not ready: no fresh g1_debug state or camera frame")
        return False

    async def _stream(
        self,
        generation: int,
        connection: Psi0Connection,
        adapter: ActionAdapter,
    ) -> None:
        period = 1.0 / max(self.config.control_hz, 1.0)
        next_tick = time.monotonic()
        pacer = SimulationClockPacer(max(self.config.control_hz, 1.0))
        last_version = 0
        info = self.info
        # The served dataset name is the single source of truth; there is no
        # client-side override (a mismatch would silently change the condition).
        if info is None or not info.dataset_name:
            raise SessionError("no validated /info dataset_name; refusing to stream")
        dataset_name = info.dataset_name
        # The served run names its single camera; the bridge sends its frame under
        # that key so a checkpoint that calls the camera something else still sees
        # the observation it was built for.
        image_key = info.image_key
        if self.policy_clock.mode == "simulation":
            await self._stream_simulation(
                generation, connection, adapter, pacer, dataset_name, image_key
            )
            return

        while self._is_current(generation):
            state_snapshot = self.monitor.state(max_age_s=STATE_MAX_AGE_S)
            frame_snapshot = self.monitor.frame(max_age_s=FRAME_MAX_AGE_S)
            if state_snapshot is None or frame_snapshot is None:
                # A stale observation must not be paired with a fresh action.
                next_tick = time.monotonic() + period
                await asyncio.sleep(period)
                continue

            payload = serialize_request(
                image={image_key: frame_snapshot.frame},
                state=state_snapshot.raw_state,
                instruction=self.config.instruction,
                dataset_name=dataset_name,
                timestamp=f"{time.time():.6f}",
            )
            self._record_observation(generation, frame_snapshot.frame)
            if self.telemetry is not None:
                self.telemetry.observation(
                    state=state_snapshot.raw_state,
                    frame=frame_snapshot.frame,
                    state_time_s=state_snapshot.timestamp_s,
                    frame_time_s=frame_snapshot.timestamp_s,
                    dataset_name=dataset_name,
                    instruction=self.config.instruction,
                    image_key=image_key,
                )

            last_version = await self._flush_stale(generation, connection, last_version)
            if not self._is_current(generation):
                break
            await connection.send(payload)
            reply = await self._wait_action(connection, last_version)
            if reply is None:
                break
            last_version = reply.version

            if not self._is_current(generation):
                break
            packed = adapter.pack(reply.action)  # raises on a bad width/neck/NaN
            if adapter.last_adapted is not None and adapter.last_adapted.neck_discarded:
                with self._lock:
                    self._neck_discard_count += 1
            published = False
            if self._send_gate.is_set() and self._is_current(generation):
                published = bool(self._publisher.publish(packed))
                if published:
                    self._record_action(generation, adapter.next_frame_index - 1)
            if self.telemetry is not None:
                self.telemetry.target_action(
                    reply.action,
                    published=published,
                    frame_index=adapter.next_frame_index - 1,
                    version=reply.version,
                    **_neck_discard_fields(adapter),
                )

            if self.policy_clock.mode == "wall":
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_tick = time.monotonic()

    async def _stream_simulation(
        self,
        generation: int,
        connection: Psi0Connection,
        adapter: ActionAdapter,
        pacer: SimulationClockPacer,
        dataset_name: str,
        image_key: str,
    ) -> None:
        async def send_observations() -> None:
            while self._is_current(generation):
                await self._wait_simulation_tick(generation, pacer)
                if not self._is_current(generation):
                    return
                state_snapshot = self.monitor.state(max_age_s=STATE_MAX_AGE_S)
                frame_snapshot = self.monitor.frame(max_age_s=FRAME_MAX_AGE_S)
                if state_snapshot is None or frame_snapshot is None:
                    continue
                payload = serialize_request(
                    image={image_key: frame_snapshot.frame},
                    state=state_snapshot.raw_state,
                    instruction=self.config.instruction,
                    dataset_name=dataset_name,
                    timestamp=f"{time.time():.6f}",
                )
                self._record_observation(generation, frame_snapshot.frame)
                if self.telemetry is not None:
                    self.telemetry.observation(
                        state=state_snapshot.raw_state,
                        frame=frame_snapshot.frame,
                        state_time_s=state_snapshot.timestamp_s,
                        frame_time_s=frame_snapshot.timestamp_s,
                        dataset_name=dataset_name,
                        instruction=self.config.instruction,
                        image_key=image_key,
                    )
                await connection.send(payload)

        async def receive_actions() -> None:
            last_version = 0
            while self._is_current(generation):
                reply = await self._wait_action(connection, last_version)
                if reply is None:
                    return
                if reply.version <= last_version:
                    continue
                last_version = reply.version
                packed = adapter.pack(reply.action)
                if adapter.last_adapted is not None and adapter.last_adapted.neck_discarded:
                    with self._lock:
                        self._neck_discard_count += 1
                published = False
                if self._send_gate.is_set() and self._is_current(generation):
                    published = bool(self._publisher.publish(packed))
                    if published:
                        self._record_action(generation, adapter.next_frame_index - 1)
                if self.telemetry is not None:
                    self.telemetry.target_action(
                        reply.action,
                        published=published,
                        frame_index=adapter.next_frame_index - 1,
                        version=reply.version,
                        **_neck_discard_fields(adapter),
                    )

        sender = asyncio.create_task(send_observations())
        receiver = asyncio.create_task(receive_actions())
        done, pending = await asyncio.wait(
            (sender, receiver), return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        results = await asyncio.gather(*done, *pending, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                raise result

    async def _wait_simulation_tick(
        self, generation: int, pacer: SimulationClockPacer,
    ) -> int:
        while self._is_current(generation):
            policy_time = self.policy_clock.now()
            update = pacer.update(policy_time.seconds, policy_time.generation)
            if update.reset:
                raise SessionError("Isaac simulation clock reset during a running psi0 session")
            if update.rows_due >= 1:
                return update.rows_due
            await asyncio.sleep(0.001)
        return 0

    async def _flush_stale(self, generation: int, connection: Psi0Connection, last_version: int) -> int:
        """Drop actions buffered from an earlier observation; return the newest version."""
        while True:
            if not self._is_current(generation):
                return last_version
            try:
                text = await asyncio.wait_for(connection.recv(), timeout=STALE_FLUSH_S)
            except asyncio.TimeoutError:
                return last_version
            try:
                reply = decode_response(text)
            except Psi0ClientError:
                continue
            last_version = max(last_version, reply.version)

    async def _wait_action(self, connection: Psi0Connection, last_version: int) -> Optional[ActionReply]:
        """Read the newest action strictly newer than ``last_version``.

        Returns ``None`` when a stop was requested; raises when the deadline passes
        or a message cannot be decoded.
        """
        deadline = time.monotonic() + self.config.recv_timeout_s
        while True:
            if self._stop_event.is_set() or not self._send_gate.is_set():
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SessionError(f"no action from the server within {self.config.recv_timeout_s}s")
            try:
                text = await asyncio.wait_for(connection.recv(), timeout=min(POLL_SLICE_S, remaining))
            except asyncio.TimeoutError:
                continue
            reply = decode_response(text)
            if reply.version > last_version:
                return reply

    # -- state helpers -----------------------------------------------------

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation and not self._stop_event.is_set()

    def _record_observation(self, generation: int, frame: np.ndarray) -> None:
        if not self._is_current(generation):
            return
        with self._lock:
            self._policy_frame = frame
            self._policy_frame_time = time.monotonic()

    def _record_action(self, generation: int, frame_index: int) -> None:
        if not self._is_current(generation):
            return
        with self._lock:
            self._last_action_time = time.time()
            self._last_action_index = int(frame_index)
            self._sent_count += 1

    def _fail(self, generation: int, message: str) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._state = ERROR
            self._error = message
            self._starting = False
            self._ws_connected = False
        self._send_gate.clear()
        self._publisher.halt()  # a failed session never publishes; the socket stays owned
        self._stop_event.set()
        self._ready_event.set()
        self._note_transition(ERROR, error=message)

    def _note_transition(self, state: str, **fields: Any) -> None:
        if self.telemetry is not None:
            self.telemetry.transition(state, **fields)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
