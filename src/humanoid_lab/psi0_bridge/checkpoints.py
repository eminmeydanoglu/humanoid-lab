"""The operator-selectable checkpoint allowlist and its controlled switch.

The UI can only ever pick one of the entries the launcher built from real
artifacts: the fine-tuned run it was given on the command line, plus the
training-start (base) run dir materialized from that run's own lineage.  There
is no path-typed input anywhere in the API.

A switch is a real server change, in this order: the session is stopped (send
gate closed, WebSocket loop joined), the owned policy server process is
terminated, a new one is started for the selected run dir/step, and the served
``/info`` must match that entry's canonical run dir, step, action width, state
width and transforms before the session is marked ready again.  Any failure is
fail-closed: the new server is torn down, the previous checkpoint is started
again (rollback) and the session is left in ``ERROR`` with the reason -- no
action from the old checkpoint is ever repeated.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .contracts import ServerInfo


class CheckpointError(RuntimeError):
    """The requested checkpoint cannot be served or the switch failed."""


class UnknownCheckpoint(CheckpointError):
    """The requested id is not in the launcher's allowlist."""


class SwitchInProgress(CheckpointError):
    """Another switch is running; the caller must wait for it to finish."""


class CheckpointUnavailable(CheckpointError):
    """The allowlisted entry has no verified artifact to serve."""


@dataclass(frozen=True)
class CheckpointEntry:
    """One allowlisted policy checkpoint, already validated against its run dir."""

    id: str
    label: str
    run_dir: Optional[Path]
    step: int
    detail: Mapping[str, Any] = field(default_factory=dict)
    available: bool = True
    reason: Optional[str] = None

    def brief(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "run_dir": None if self.run_dir is None else str(self.run_dir),
            "step": self.step,
            "available": self.available,
            "reason": self.reason,
        }


def _canonical(path: str | Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:  # pragma: no cover - only on exotic filesystems
        return Path(path)


class CheckpointController:
    """Owns the allowlist, the policy server process and the switch transaction."""

    def __init__(
        self,
        session: Any,
        server: Any,
        entries: list[CheckpointEntry],
        *,
        verify: Callable[[CheckpointEntry], ServerInfo],
        initial_id: str,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        if not entries:
            raise CheckpointError("at least one checkpoint entry is required")
        self.session = session
        self.server = server
        self._entries: dict[str, CheckpointEntry] = {entry.id: entry for entry in entries}
        if initial_id not in self._entries:
            raise CheckpointError(f"initial checkpoint {initial_id!r} is not in the allowlist")
        self._order = [entry.id for entry in entries]
        self.selected_id = initial_id
        self._verify = verify
        self._log = log
        self._switch_lock = threading.Lock()
        self._switching = False

    # -- description -------------------------------------------------------

    @property
    def entries(self) -> list[CheckpointEntry]:
        return [self._entries[entry_id] for entry_id in self._order]

    def entry(self, entry_id: str) -> CheckpointEntry:
        try:
            return self._entries[entry_id]
        except KeyError:
            raise UnknownCheckpoint(
                f"unknown checkpoint {entry_id!r}; allowed: {sorted(self._entries)}"
            ) from None

    def _served(self, entry: CheckpointEntry) -> bool:
        """Whether the currently validated ``/info`` is exactly this entry."""
        info = self.session.info
        if info is None or entry.run_dir is None:
            return False
        if info.ckpt_step != entry.step:
            return False
        return _canonical(info.run_dir) == _canonical(entry.run_dir)

    def status(self) -> dict[str, Any]:
        info = self.session.info
        selected = self._entries.get(self.selected_id)
        return {
            "options": [entry.brief() for entry in self.entries],
            "selected": None if selected is None else selected.brief(),
            "active": None if info is None else {
                "run_dir": info.run_dir,
                "step": info.ckpt_step,
                "action_dim": info.action_dim,
                "state_dim": info.state_dim,
                "dataset_name": info.dataset_name,
            },
            "serving_selected": bool(selected is not None and self._served(selected)),
            "switching": self._switching,
            "server": {
                "pid": self.server.pid,
                "alive": bool(self.server.alive),
                "run_dir": None if self.server.run_dir is None else str(self.server.run_dir),
                "step": self.server.step,
            },
        }

    # -- switch ------------------------------------------------------------

    def switch(self, entry_id: str) -> dict[str, Any]:
        """Stop the session, restart the owned policy server, verify, then go ready."""
        entry = self.entry(entry_id)
        if not entry.available:
            raise CheckpointUnavailable(
                f"checkpoint {entry.label} is not available: {entry.reason or 'no artifact'}"
            )
        if not self._switch_lock.acquire(blocking=False):
            raise SwitchInProgress("another checkpoint switch is already in progress")
        self._switching = True
        try:
            if not (entry_id == self.selected_id and self.server.alive and self._served(entry)):
                previous = self._entries.get(self.selected_id)
                self._log(f"[checkpoint] switch -> {entry.id} ({entry.label}) at {entry.run_dir} "
                          f"step {entry.step}; stopping the session and the policy server")
                try:
                    self.session.stop()  # send gate closed, WebSocket loop joined
                    self.server.stop()
                    self.server.start(entry.run_dir, entry.step)
                    info = self._verify(entry)
                except Exception as exc:  # noqa: BLE001 - every failure is fail-closed
                    failure = f"checkpoint switch to {entry.label} failed: " \
                              f"{type(exc).__name__}: {exc}"
                    rollback_note = self._rollback(previous)
                    self.session.mark_error(f"{failure}; {rollback_note}")
                    self._log(f"[checkpoint] {failure}; {rollback_note}")
                    raise CheckpointError(f"{failure}; {rollback_note}") from exc

                self.session.adopt_policy(info)
                self.selected_id = entry_id
                self._log(f"[checkpoint] serving {entry.id}: run_dir {info.run_dir} "
                          f"step {info.ckpt_step} action {info.action_dim}D")
        finally:
            self._switching = False
            self._switch_lock.release()
        # Built after the flag clears, so the response already shows the ready state.
        return self.status()

    def _rollback(self, previous: Optional[CheckpointEntry]) -> str:
        """Bring the previous checkpoint back; report honestly when that fails too."""
        self.server.stop()
        if previous is None:
            return "no previous checkpoint to roll back to"
        if not previous.available:
            return f"previous checkpoint {previous.label} is unavailable"
        try:
            self.server.start(previous.run_dir, previous.step)
            self._verify(previous)
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            self.server.stop()
            return (f"rollback to {previous.label} failed: {type(exc).__name__}: {exc}; "
                    "no policy server is running")
        return f"rolled back to {previous.label}"

    def close(self) -> None:
        self.server.stop()
