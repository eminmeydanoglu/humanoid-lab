"""Single UI controller for PSI checkpoints and NVIDIA GR00T."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .action_router import ActionRouter
from .checkpoints import CheckpointError, CheckpointUnavailable, SwitchInProgress, UnknownCheckpoint
from .groot_backend import GrootProcessGroup
from .reset_client import IsaacResetClient, ResetError
from .session import ERROR, IDLE, RUNNING, STOPPED, Session


@dataclass(frozen=True)
class ModelEntry:
    id: str
    label: str
    kind: str
    run_dir: Path | None
    step: int | None
    available: bool = True
    reason: str | None = None
    detail: dict[str, Any] | None = None

    def brief(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "kind": self.kind,
            "run_dir": None if self.run_dir is None else str(self.run_dir),
            "step": self.step, "available": self.available, "reason": self.reason,
        }


class ModelController:
    """Keep the UI alive while switching the active policy backend."""

    def __init__(
        self,
        *,
        psi_session: Session,
        psi_server: Any,
        psi_entries: list[ModelEntry],
        verify_psi: Callable[[ModelEntry], Any],
        groot: GrootProcessGroup,
        router: ActionRouter,
        initial_id: str,
        reset_endpoint: str = "tcp://localhost:5559",
        keyboard_endpoint: str = "tcp://*:5580",
        warmstart: Any | None = None,
        warmstart_delay_s: float = 0.0,
        initial_pose: Any | None = None,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        import zmq

        self.psi_session = psi_session
        self.psi_server = psi_server
        self.groot = groot
        self.router = router
        #: Opt-in demonstration-token warm start (``None`` unless asked for).
        self.warmstart = warmstart
        self.warmstart_delay_s = float(warmstart_delay_s)
        #: Opt-in delivery of the VLA client's own initial-pose command over the
        #: settle (``None`` unless asked for); mutually exclusive with the warm
        #: start, which owns the same window.
        self.initial_pose = initial_pose
        self.monitor = psi_session.monitor
        self.reset_endpoint = reset_endpoint
        self._verify_psi = verify_psi
        self._entries = {entry.id: entry for entry in psi_entries}
        self._entries["groot"] = ModelEntry(
            id="groot", label="GR00T", kind="groot", run_dir=groot.checkpoint, step=None,
        )
        self._order = [entry.id for entry in psi_entries] + ["groot"]
        self.selected_id = initial_id
        self._state = IDLE
        self._error: str | None = None
        self._switching = False
        self._lock = threading.Lock()
        self._switch_lock = threading.Lock()
        self._log = log
        self._context = zmq.Context()
        self._keys = self._context.socket(zmq.PUB)
        self._keys.setsockopt(zmq.LINGER, 0)
        self._keys.bind(keyboard_endpoint)
        self.router.select("psi")

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def info(self):
        return self.psi_session.info if self._active().kind == "psi" else None

    def entry(self, entry_id: str) -> ModelEntry:
        try:
            return self._entries[entry_id]
        except KeyError:
            raise UnknownCheckpoint(f"unknown model {entry_id!r}; allowed: {self._order}") from None

    def start(self, instruction: str | None = None) -> dict[str, Any]:
        del instruction
        if self._switching:
            raise CheckpointError("model switch is in progress")
        if self.state == ERROR:
            raise CheckpointError("session is in ERROR; Reset before Start")
        active = self._active()
        if active.kind == "psi":
            self.router.select("psi")
            self._ensure_sonic_pose()
            self.router.resume()
            result = self.psi_session.start()
            with self._lock:
                self._state = result["state"]
                self._error = result.get("error")
        else:
            if not self.groot.alive:
                raise CheckpointError("GR00T backend is not running")
            # Both settle commands end here for the same reason: the source
            # switch below is the atomic generation change that hands the SONIC
            # action port to the policy, so no settle token can follow it.
            self._halt_settle()
            self.router.select("groot")
            self.router.resume()
            self._send_key("p")
            with self._lock:
                self._state = RUNNING
                self._error = None
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._halt_settle()
        self.router.halt()
        active = self._active()
        if active.kind == "psi":
            self.psi_session.stop()
        elif self.state == RUNNING and self.groot.alive:
            self._send_key("p")
        with self._lock:
            if self._state != ERROR:
                self._state = STOPPED
        return self.status()

    def reset(self) -> dict[str, Any]:
        clock_baseline = None if self.warmstart is None else self.warmstart.clock_sample()
        self.stop()
        active = self._active()
        if active.kind == "groot":
            self.groot.stop()
        time.sleep(0.3)
        try:
            self._request_reset()
        except ResetError as exc:
            with self._lock:
                self._state = ERROR
                self._error = str(exc)
            return self.status()
        self.monitor.invalidate()
        try:
            if active.kind == "groot":
                self.groot.start()
                self._prepare_groot()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._state = ERROR
                self._error = f"reset failed: {type(exc).__name__}: {exc}"
            return self.status()
        with self._lock:
            self._state = IDLE
            self._error = None
        self._arm_settle(active, clock_baseline=clock_baseline)
        return self.status()

    def _request_reset(self) -> None:
        """Apply the reset, waiting for the real boundary only when it matters.

        The endpoint's ``reset_queued`` reply means the loop *accepted* the work,
        not that it ran it: the callback sets a flag the simulation loop consumes
        at its next iteration.  That is enough for a plain selection, whose next
        ``Start`` waits for a fresh observation anyway, so the default stays the
        legacy queue-acknowledged call and no existing run changes timing.

        A session that publishes a *settle command* -- the opt-in initial-pose
        handshake or warm start -- is different: that stream is aimed at the
        controller's first post-reset decode, so firing it at the queue ack lets
        the deployment reach SONIC before the scene moved, which is the race
        ``hatalar.md`` A5 records.  With either one configured this waits for the
        loop's ``episode_id`` to advance, so the settle always starts from the
        applied reset.  A loop that never applies it fails the Reset closed.
        """
        client = IsaacResetClient(self.reset_endpoint, timeout_ms=2000)
        if self.initial_pose is None and self.warmstart is None:
            client.request_reset()
            return
        applied = client.request_reset_applied()
        self._log(f"[reset] applied before the settle (episode_id {applied})")

    def switch(self, entry_id: str) -> dict[str, Any]:
        target = self.entry(entry_id)
        if not target.available:
            raise CheckpointUnavailable(target.reason or f"{target.label} is unavailable")
        if not self._switch_lock.acquire(blocking=False):
            raise SwitchInProgress("another model switch is already in progress")
        self._switching = True
        previous = self._active()
        try:
            if target.id != previous.id:
                self._log(f"[model] {previous.label} -> {target.label}")
                self.stop()
                self._stop_backend(previous)
                try:
                    self._start_backend(target)
                except Exception as exc:  # noqa: BLE001
                    failure = f"model switch to {target.label} failed: {type(exc).__name__}: {exc}"
                    rollback = self._rollback(previous)
                    with self._lock:
                        self._state = ERROR
                        self._error = f"{failure}; {rollback}"
                    raise CheckpointError(self._error) from exc
                self.selected_id = target.id
                with self._lock:
                    self._state = IDLE
                    self._error = None
        finally:
            self._switching = False
            self._switch_lock.release()
        return self.status()

    def status(self) -> dict[str, Any]:
        active = self._active()
        monitor = self.monitor.status()
        router = self.router.status()
        with self._lock:
            state, error = self._state, self._error
        policy_alive = self.groot.alive if active.kind == "groot" else bool(self.psi_server.alive)
        return {
            "state": state, "error": error, "starting": self._switching,
            "psi0": {
                "url": "tcp://127.0.0.1:5550" if active.kind == "groot" else self.psi_session.config.ws_url,
                "connected": policy_alive,
                "info": {"action_dim": 64, "action_chunk_size": 40} if active.kind == "groot"
                        else self.psi_session.status()["psi0"]["info"],
            },
            "sonic_state": monitor.get("state", {}),
            "camera": monitor.get("camera", {}),
            "action": {
                "endpoint": router["endpoint"], "last_time": _iso(router["last_time"]),
                "last_index": router["source"], "sent": router["sent"],
            },
            "warmstart": None if self.warmstart is None else self.warmstart.status(),
            "initial_pose": None if self.initial_pose is None else self.initial_pose.status(),
            "model": active.brief(),
        }

    def checkpoints_status(self) -> dict[str, Any]:
        active = self._active()
        return {
            "options": [self._entries[key].brief() for key in self._order],
            "selected": active.brief(),
            "active": {**active.brief(), "action_dim": 64 if active.kind == "groot" else None},
            "serving_selected": self.groot.alive if active.kind == "groot" else bool(self.psi_server.alive),
            "switching": self._switching,
            "server": ({
                "pid": self.groot.pid, "alive": self.groot.alive,
                "run_dir": str(self.groot.checkpoint), "step": None,
            } if active.kind == "groot" else {
                "pid": self.psi_server.pid, "alive": bool(self.psi_server.alive),
                "run_dir": None if self.psi_server.run_dir is None else str(self.psi_server.run_dir),
                "step": self.psi_server.step,
            }),
        }

    # app.py's checkpoint-controller surface
    def status_models(self) -> dict[str, Any]:
        return self.checkpoints_status()

    def refresh_info(self):
        if self._active().kind == "groot":
            return self.groot.status()
        return self.psi_session.refresh_info()

    def preview_frame(self):
        return self.psi_session.preview_frame()

    def close(self) -> None:
        self._halt_settle()
        self.router.halt()
        self.groot.stop()
        self.psi_server.stop()
        self.psi_session.close()
        self.router.close()
        self._keys.close()
        self._context.term()

    def _active(self) -> ModelEntry:
        return self._entries[self.selected_id]

    # -- opt-in settle commands --------------------------------------------

    def _arm_settle(self, active: ModelEntry, *, clock_baseline: Any | None = None) -> None:
        """Schedule whichever opt-in command owns this settle, if any.

        The two are mutually exclusive by construction (the launcher refuses a
        session configured with both): each one owns the whole window from the
        end of the ``Reset`` to ``Start``, and each selects its own router source
        so the deployment is driven by exactly one of them at a time.
        """
        if self._arm_initial_pose(active):
            return
        self._arm_warmstart(active, clock_baseline=clock_baseline)

    def _arm_initial_pose(self, active: ModelEntry) -> bool:
        """Deliver the VLA client's own initial-pose command over the settle.

        Armed at the end of every ``Reset`` -- the start of the settle -- and
        published at the deployment's control rate until ``Start`` halts it.  The
        point of the handshake is that the *intended* initial-pose command is
        what the deployment holds when the policy takes over, instead of whatever
        token the previous rollout happened to leave behind.
        """
        if self.initial_pose is None:
            return False
        if active.kind != "groot":
            self._log("[initial-pose] not armed: the selected backend is not GR00T")
            return False
        self.router.select("initial_pose")
        self.router.resume()
        self.initial_pose.arm(0.0)
        self._log(
            "[initial-pose] VLA client's initial-pose command armed for the settle "
            f"(token sha256 {self.initial_pose.stream.source.get('token_sha256_f32', '')[:12]})"
        )
        return True

    def _arm_warmstart(self, active: ModelEntry, *, clock_baseline: Any | None = None) -> None:
        """Schedule the prepared token stream over the tail of the settle.

        The router selects the warm-start source for the whole window, so the
        settle's own commands (and nothing else) reach SONIC; the stream starts
        after ``warmstart_delay_s`` and keeps publishing its last token until
        ``Start`` stops it.  The GR00T backend is the only one this applies to:
        a PSI selection is driven by this bridge's own client and never runs a
        warm start.
        """
        if self.warmstart is None:
            return
        if active.kind != "groot":
            self._log("[warmstart] not armed: the selected backend is not GR00T")
            return
        self.router.select("warmstart")
        self.router.resume()
        self.warmstart.arm(self.warmstart_delay_s, clock_baseline=clock_baseline)
        self._log(
            f"[warmstart] demo token stream armed for the settle tail "
            f"(delay {self.warmstart_delay_s:.2f}s)"
        )

    def _halt_settle(self) -> None:
        """Stop every opt-in settle command; safe to call at any transition."""
        self._halt_initial_pose()
        self._halt_warmstart()

    def _halt_initial_pose(self) -> None:
        if self.initial_pose is not None:
            self.initial_pose.halt()

    def _halt_warmstart(self) -> None:
        if self.warmstart is not None:
            self.warmstart.halt()

    def _stop_backend(self, entry: ModelEntry) -> None:
        self._halt_settle()
        self.router.halt()
        if entry.kind == "psi":
            self.psi_session.stop()
            self.psi_server.stop()
        else:
            self.groot.stop()

    def _start_backend(self, entry: ModelEntry) -> None:
        if entry.kind == "psi":
            assert entry.run_dir is not None and entry.step is not None
            self.psi_server.start(entry.run_dir, entry.step)
            info = self._verify_psi(entry)
            self.psi_session.adopt_policy(info)
            self.router.select("psi")
        else:
            self.groot.start()
            self.router.select("groot")
            self._prepare_groot()

    def _prepare_groot(self) -> None:
        self.router.select("groot")
        self.router.resume()
        time.sleep(1.0)
        self._send_key("k")
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if self.monitor.status().get("state", {}).get("alive"):
                break
            time.sleep(0.25)
        else:
            # A PUB/SUB subscriber can join after the first command. Toggle the
            # upstream VLA state once, then start again, matching NVIDIA's path.
            self._send_key("k")
            time.sleep(1.0)
            self._send_key("k")
            retry_deadline = time.monotonic() + 8.0
            while time.monotonic() < retry_deadline:
                if self.monitor.status().get("state", {}).get("alive"):
                    break
                time.sleep(0.25)
            else:
                self.router.halt()
                raise CheckpointError("SONIC controller produced no g1_debug state")
        self._send_key("i")
        time.sleep(3.0)
        self.router.halt()

    def _rollback(self, previous: ModelEntry) -> str:
        try:
            self._start_backend(previous)
            self.selected_id = previous.id
            return f"rolled back to {previous.label}"
        except Exception as exc:  # noqa: BLE001
            return f"rollback to {previous.label} failed: {type(exc).__name__}: {exc}"

    def _ensure_sonic_pose(self) -> None:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

        self.router.send_control(build_command_message(start=True, stop=False, planner=True))
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self.monitor.status().get("state", {}).get("alive"):
                break
            time.sleep(0.25)
        self.router.send_control(build_command_message(start=True, stop=False, planner=False))
        time.sleep(0.1)

    def _send_key(self, key: str) -> None:
        self._keys.send_string(key)


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
