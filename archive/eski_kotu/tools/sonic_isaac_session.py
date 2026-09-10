"""Session state for ``dev.sh sonic-isaac``: start, status and stop agree.

Pure logic only.  The process, socket and clock work is injected, so the status
derivation and the launcher gate wiring can be tested without Isaac or Docker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
import json
import os
import time

from sonic_isaac_contract import (
    LOWCMD_MAX_AGE_S,
    SIM_DDS_DOMAIN_ID,
    SIM_DDS_INTERFACE,
    ContractError,
    SessionClaim,
    collect_refusals,
    load_dds_profile,
)

__all__ = [
    "AGENT_STATE_READY_PENDING_USER_DRIVE",
    "ChildProcess",
    "SessionPaths",
    "SessionState",
    "build_status",
    "derive_controller",
    "derive_simulator",
    "load_state",
    "plan_children",
    "save_state",
    "stop_plan",
]

AGENT_STATE_READY_PENDING_USER_DRIVE = "READY_PENDING_USER_DRIVE"

# `/data/runtime` is bind-mounted read-only, so session state and run evidence
# live under the writable outputs mount.
DEFAULT_RUNTIME_DIR = "/outputs/sonic-isaac"

CHILD_ROLES = ("isaac", "bridge", "sonic", "input")


@dataclass
class SessionPaths:
    runtime_dir: Path
    state_file: Path
    ready_file: Path
    evidence_dir: Path
    log_dir: Path

    @classmethod
    def under(cls, root: str | Path) -> "SessionPaths":
        base = Path(root)
        return cls(
            runtime_dir=base,
            state_file=base / "session.json",
            ready_file=base / "runner-ready.json",
            evidence_dir=base / "evidence",
            log_dir=base / "logs",
        )


@dataclass(frozen=True)
class ChildProcess:
    role: str
    pid: int
    argv: tuple[str, ...] = ()
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if self.role not in CHILD_ROLES:
            raise ContractError(f"unknown child role {self.role!r}")


@dataclass
class SessionState:
    """Everything ``status`` needs, plus what ``stop`` is allowed to signal."""

    robot_profile: str
    input: str
    domain_id: int = SIM_DDS_DOMAIN_ID
    interface: str = SIM_DDS_INTERFACE
    simulator: str = "stopped"
    controller: str = "absent"
    actuation: str = "passive"
    owner_pid: int = 0
    children: list[ChildProcess] = field(default_factory=list)
    lowstate_hz: float = 0.0
    lowcmd_hz: float = 0.0
    last_lowcmd_age_ms: float | None = None
    root_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_roll_pitch_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sonic_seen: bool = False
    agent_state: str = AGENT_STATE_READY_PENDING_USER_DRIVE
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["children"] = [asdict(child) for child in self.children]
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "SessionState":
        data = dict(payload)
        data["children"] = [
            ChildProcess(
                role=str(item["role"]),
                pid=int(item["pid"]),
                argv=tuple(item.get("argv", ())),
                started_at=float(item.get("started_at", 0.0)),
            )
            for item in data.get("children", [])
        ]
        for key in ("root_position", "root_roll_pitch_yaw"):
            if key in data:
                data[key] = tuple(float(value) for value in data[key])
        return cls(**data)

    def claim(self) -> SessionClaim:
        return SessionClaim(
            pid=int(self.owner_pid),
            domain_id=int(self.domain_id),
            interface=str(self.interface),
            owner=f"sonic-isaac:{self.robot_profile}",
        )

    def child_pids(self, role: str | None = None) -> tuple[int, ...]:
        return tuple(
            child.pid for child in self.children if role is None or child.role == role
        )


def save_state(paths: SessionPaths, state: SessionState) -> None:
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    temporary = paths.state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(state.to_json(), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, paths.state_file)


def load_state(paths: SessionPaths) -> SessionState | None:
    if not paths.state_file.is_file():
        return None
    try:
        return SessionState.from_json(json.loads(paths.state_file.read_text()))
    except (ValueError, TypeError, KeyError):
        return None


def derive_simulator(
    *, running: bool, playing: bool | None, paused: bool | None
) -> str:
    if not running:
        return "stopped"
    if playing:
        return "playing"
    if paused:
        return "paused"
    return "ready"


def derive_controller(
    *,
    lowcmd_age_ms: float | None,
    sonic_running: bool,
    max_age_ms: float = LOWCMD_MAX_AGE_S * 1000.0,
) -> str:
    """One of absent | waiting | controlled | stale."""
    if lowcmd_age_ms is None:
        return "waiting" if sonic_running else "absent"
    if lowcmd_age_ms <= max_age_ms:
        return "controlled"
    return "stale"


def build_status(state: SessionState) -> dict:
    """Assemble the documented status JSON, deriving actuation from lowcmd age."""
    age_ms = state.last_lowcmd_age_ms
    controller = state.controller
    actuation = "controlled" if controller == "controlled" else "passive"
    return {
        "simulator": state.simulator,
        "controller": controller,
        "actuation": actuation,
        "input": state.input,
        "robot_profile": state.robot_profile,
        "dds_domain": int(state.domain_id),
        "dds_interface": state.interface,
        "lowstate_hz": float(state.lowstate_hz),
        "lowcmd_hz": float(state.lowcmd_hz),
        "last_lowcmd_age_ms": 0.0 if age_ms is None else float(age_ms),
        "root_position": [float(value) for value in state.root_position],
        "root_roll_pitch_yaw": [float(value) for value in state.root_roll_pitch_yaw],
        "agent_state": state.agent_state,
    }


def plan_children(robot_profile: str, input_mode: str) -> tuple[str, ...]:
    """Roles that ``start`` will own, in launch order."""
    if robot_profile not in {"g1-29dof", "g1-inspire"}:
        raise ContractError(f"unknown robot profile {robot_profile!r}")
    if input_mode not in {"keyboard", "f310", "acceptance"}:
        raise ContractError(f"unknown input mode {input_mode!r}")
    return ("isaac", "bridge", "sonic", "input")


def stop_plan(state: SessionState) -> tuple[int, ...]:
    """Only recorded child PIDs, most recent first; never a pattern kill."""
    ordered = sorted(state.children, key=lambda child: child.started_at, reverse=True)
    pids = []
    for child in ordered:
        if int(child.pid) > 0 and int(child.pid) != int(state.owner_pid):
            pids.append(int(child.pid))
    return tuple(dict.fromkeys(pids))


def evaluate_launch(
    *,
    env: Mapping[str, str],
    dds_config_text: str | None,
    domain_id: int,
    sonic_mode: str,
    requested_interface: str,
    argv: Sequence[str],
    physical_interfaces: Sequence[str],
    existing_session: SessionState | None,
    own_pid: int,
) -> tuple[str, ...]:
    """Launcher gate result for a real start attempt."""
    profile = None
    if dds_config_text is not None:
        try:
            profile = load_dds_profile(dds_config_text, source=str(env.get("CYCLONEDDS_URI")))
        except ContractError:
            profile = None
    return collect_refusals(
        cyclonedds_uri=env.get("CYCLONEDDS_URI"),
        profile=profile,
        domain_id=domain_id,
        sonic_mode=sonic_mode,
        requested_interface=requested_interface,
        argv=argv,
        physical_interfaces=physical_interfaces,
        existing_session=None if existing_session is None else existing_session.claim(),
        own_pid=own_pid,
    )
