"""Simulation contract between Isaac Sim and the pinned upstream SONIC deploy.

Everything here is pure logic: no Isaac, DDS, ROS or pygame import may appear in
this module, so the isolation rules, the name-based joint mapping and the limit
validation can all be driven directly by tests.

DDS index order
---------------
``lowcmd.motor_cmd[i]`` / ``lowstate.motor_state[i]`` are indexed in the
hardware (URDF) order declared by upstream ``G1JointIndex`` in
``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp``.
That order is what :data:`SONIC_BODY_JOINT_NAMES` below reproduces, and it is
also the order of the policy's per-joint arrays in ``policy_parameters.hpp``.
Isaac's articulation exposes its own DOF order, which is why the joints are
matched by NAME and never by position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import ipaddress
import math
import xml.etree.ElementTree as ET

__all__ = [
    "BODY_JOINT_COUNT",
    "CONTRACT_SOURCE",
    "DDS_MOTOR_ARRAY_SIZE",
    "DEFAULT_LOWCMD_HZ",
    "LOWCMD_MAX_AGE_S",
    "PHYSICS_DT",
    "PHYSICS_HZ",
    "SIM_DDS_DOMAIN_ID",
    "SIM_DDS_INTERFACE",
    "SONIC_BODY_JOINT_NAMES",
    "ContractError",
    "DdsProfile",
    "JointLimits",
    "JointMapping",
    "REFUSAL_REASONS",
    "SIM_ONLY_DEPLOY_FLAG",
    "SessionClaim",
    "build_joint_mapping",
    "check_launch_safety",
    "check_sim_dds_isolation",
    "load_dds_profile",
    "mapping_evidence_summary",
    "resolve_cyclonedds_uri",
    "validate_limits",
]

#: Upstream commit the contract was transcribed from.
CONTRACT_SOURCE = "NVlabs/GR00T-WholeBodyControl@a0732b642c0333077e127a2f56ab0014c196bca4"

SIM_DDS_DOMAIN_ID = 42
SIM_DDS_INTERFACE = "lo"

#: Physics runs at 200 Hz; the controller is expected to publish lowcmd at 50 Hz.
PHYSICS_HZ = 200
PHYSICS_DT = 1.0 / PHYSICS_HZ
DEFAULT_LOWCMD_HZ = 50

#: A lowcmd older than this is no longer applied; the body drops to PASSIVE.
LOWCMD_MAX_AGE_S = 0.1

BODY_JOINT_COUNT = 29

#: Upstream publishes fixed 35-entry motor arrays; only 0..28 are populated.
DDS_MOTOR_ARRAY_SIZE = 35

#: The 29 body joints in DDS/hardware index order (upstream ``G1JointIndex``).
SONIC_BODY_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
assert len(SONIC_BODY_JOINT_NAMES) == BODY_JOINT_COUNT
assert len(set(SONIC_BODY_JOINT_NAMES)) == BODY_JOINT_COUNT

#: Stable identifiers for every way the launcher may refuse to start.
REFUSAL_REASONS = (
    "unsafe_cyclonedds_uri",
    "unsafe_dds_domain",
    "unsafe_sonic_mode",
    "unsafe_physical_interface",
    "session_conflict",
)

#: The upstream deploy skips CRC validation only for simulation. Without it the
#: deploy expects a real robot's CRC-checked lowcmd stream, so its presence is
#: what distinguishes a simulation command from a physical-robot one.
SIM_ONLY_DEPLOY_FLAG = "--disable-crc-check"


class ContractError(RuntimeError):
    """Raised when a simulation-contract precondition does not hold."""


@dataclass(frozen=True)
class DdsProfile:
    """The subset of a CycloneDDS configuration the isolation gate inspects."""

    source: str
    interface_addresses: tuple[str, ...]
    allow_multicast: bool
    dont_route: bool

    def is_loopback_only(self) -> bool:
        return bool(self.interface_addresses) and all(
            address == SIM_DDS_INTERFACE or address.startswith("127.")
            for address in self.interface_addresses
        )


@dataclass(frozen=True)
class JointLimits:
    name: str
    position_min: float
    position_max: float
    effort_max: float
    velocity_max: float


@dataclass(frozen=True)
class JointMapping:
    """Name-based bijection between DDS motor indices and Isaac DOF indices."""

    sonic_names: tuple[str, ...]
    isaac_names: tuple[str, ...]
    sonic_to_isaac: tuple[int, ...]
    isaac_to_sonic: tuple[int, ...]

    @property
    def isaac_body_indices(self) -> tuple[int, ...]:
        return self.sonic_to_isaac

    def summary(self) -> dict:
        return {
            "contract_source": CONTRACT_SOURCE,
            "matcher": "name",
            "body_joint_count": len(self.sonic_names),
            "sonic_names": list(self.sonic_names),
            "isaac_dof_order": list(self.isaac_names),
            "sonic_to_isaac_index": list(self.sonic_to_isaac),
            "isaac_to_sonic_index": list(self.isaac_to_sonic),
            "order_assumed": False,
        }


@dataclass(frozen=True)
class SessionClaim:
    """A recorded run that owns the simulation DDS domain."""

    pid: int
    domain_id: int
    interface: str
    owner: str

    def conflicts_with(self, *, pid: int, domain_id: int, interface: str) -> bool:
        if self.domain_id != domain_id or self.interface != interface:
            return False
        return int(self.pid) != int(pid)


# --------------------------------------------------------------------------- #
# CycloneDDS configuration
# --------------------------------------------------------------------------- #


def resolve_cyclonedds_uri(uri: str | None, *, root: Path | str = "/") -> Path:
    """Return the config path a ``CYCLONEDDS_URI`` must point at.

    Only ``file://`` URIs are accepted: an empty or inline configuration would
    let DDS fall back to autodiscovery on a physical interface.
    """
    if not uri:
        raise ContractError("CYCLONEDDS_URI is not set; refusing to start with DDS autodiscovery")
    if not uri.startswith("file://"):
        raise ContractError(f"CYCLONEDDS_URI must use file:// (got {uri!r})")
    path = Path(root) / uri[len("file://") :].lstrip("/")
    if not path.is_file():
        raise ContractError(f"CYCLONEDDS_URI target does not exist: {path}")
    return path


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _flatten(root: ET.Element) -> Iterable[ET.Element]:
    for element in root.iter():
        yield element


def load_dds_profile(xml_text: str, *, source: str = "<memory>") -> DdsProfile:
    """Parse the loopback-relevant fields out of a CycloneDDS configuration."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise ContractError(f"invalid CycloneDDS config {source}: {exc}") from exc

    addresses: list[str] = []
    multicast_flags: list[bool] = []
    dont_route = False

    for element in _flatten(root):
        name = _local_name(element.tag)
        if name == "NetworkInterface":
            address = element.get("address")
            if address:
                addresses.append(address.strip())
            raw = element.get("multicast")
            if raw is not None:
                multicast_flags.append(raw.strip().lower() in {"true", "1", "yes"})
        elif name == "AllowMulticast":
            multicast_flags.append((element.text or "").strip().lower() in {"true", "1", "yes"})
        elif name == "DontRoute":
            dont_route = (element.text or "").strip().lower() in {"true", "1", "yes"}

    return DdsProfile(
        source=source,
        interface_addresses=tuple(addresses),
        allow_multicast=any(multicast_flags),
        dont_route=dont_route,
    )


def check_sim_dds_isolation(
    profile: DdsProfile,
    *,
    domain_id: int,
    expected_domain: int = SIM_DDS_DOMAIN_ID,
    expected_interface: str = SIM_DDS_INTERFACE,
) -> None:
    """Fail unless the DDS profile is the loopback-only simulation profile."""
    if int(domain_id) != int(expected_domain):
        raise ContractError(
            f"DDS domain {domain_id} is not the simulation domain {expected_domain}"
        )
    if not profile.interface_addresses:
        raise ContractError("CycloneDDS config declares no NetworkInterface")
    for address in profile.interface_addresses:
        if address != expected_interface and not address.startswith("127."):
            raise ContractError(
                f"CycloneDDS config binds {address!r}; only {expected_interface!r} is allowed"
            )
    if profile.allow_multicast:
        raise ContractError("CycloneDDS multicast must be disabled for the simulation profile")


# --------------------------------------------------------------------------- #
# Joint mapping and limits
# --------------------------------------------------------------------------- #


def build_joint_mapping(
    sonic_names: Sequence[str],
    isaac_dof_names: Sequence[str],
    *,
    expected: Sequence[str] = SONIC_BODY_JOINT_NAMES,
) -> JointMapping:
    """Match the DDS motor slots to Isaac DOFs by name.

    The order of ``isaac_dof_names`` is whatever the articulation exposes; the
    only requirement is that every expected name appears exactly once on both
    sides, so neither ordering is assumed.
    """
    sonic = tuple(str(name) for name in sonic_names)
    isaac = tuple(str(name) for name in isaac_dof_names)
    expected = tuple(expected)

    if len(sonic) != len(expected):
        raise ContractError(
            f"expected {len(expected)} SONIC body joints, got {len(sonic)}"
        )

    duplicated = sorted({name for name in sonic if sonic.count(name) > 1})
    if duplicated:
        raise ContractError(f"duplicate SONIC joint names: {duplicated}")

    missing = [name for name in expected if name not in sonic]
    unknown = sorted({name for name in sonic if name not in expected})
    if missing or unknown:
        raise ContractError(
            "SONIC joint names do not match the 29-joint contract: "
            f"missing={missing}, unknown={unknown}"
        )

    isaac_position = {name: index for index, name in enumerate(isaac)}
    absent = [name for name in expected if name not in isaac_position]
    if absent:
        raise ContractError(f"Isaac asset is missing body joints: {absent}")
    repeated = sorted({name for name in expected if isaac.count(name) > 1})
    if repeated:
        raise ContractError(f"Isaac asset repeats body joints: {repeated}")

    sonic_to_isaac = tuple(isaac_position[name] for name in sonic)
    if len(set(sonic_to_isaac)) != len(sonic_to_isaac):
        raise ContractError("SONIC->Isaac mapping is not injective")

    # Kept at full asset length: non-body DOFs (for example the Inspire hands)
    # stay -1 so the inverse map keeps indexing Isaac DOFs directly.
    isaac_to_sonic = [-1] * len(isaac)
    for sonic_index, isaac_index in enumerate(sonic_to_isaac):
        isaac_to_sonic[isaac_index] = sonic_index

    for sonic_index, isaac_index in enumerate(sonic_to_isaac):
        if isaac_to_sonic[isaac_index] != sonic_index:
            raise ContractError("SONIC->Isaac and Isaac->SONIC mappings are not inverse")

    return JointMapping(
        sonic_names=sonic,
        isaac_names=isaac,
        sonic_to_isaac=sonic_to_isaac,
        isaac_to_sonic=tuple(isaac_to_sonic),
    )


def validate_limits(limits: Sequence[JointLimits]) -> None:
    """Every effort/velocity limit must be finite and positive; bounds ordered."""
    if not limits:
        raise ContractError("no joint limits supplied")
    for limit in limits:
        for field in ("position_min", "position_max", "effort_max", "velocity_max"):
            value = getattr(limit, field)
            if not math.isfinite(value):
                raise ContractError(f"{limit.name}: {field} is not finite ({value!r})")
        if limit.effort_max <= 0.0:
            raise ContractError(f"{limit.name}: effort_max must be positive ({limit.effort_max})")
        if limit.velocity_max <= 0.0:
            raise ContractError(f"{limit.name}: velocity_max must be positive ({limit.velocity_max})")
        if limit.position_min >= limit.position_max:
            raise ContractError(
                f"{limit.name}: position bounds are not ordered "
                f"({limit.position_min} >= {limit.position_max})"
            )


def mapping_evidence_summary(
    mapping: JointMapping, limits: Sequence[JointLimits]
) -> dict:
    """The mapping record written into the run evidence log."""
    validate_limits(limits)
    by_name = {limit.name: limit for limit in limits}
    missing = [name for name in mapping.sonic_names if name not in by_name]
    if missing:
        raise ContractError(f"missing limits for body joints: {missing}")
    summary = mapping.summary()
    summary["limits"] = {
        name: {
            "position_min": by_name[name].position_min,
            "position_max": by_name[name].position_max,
            "effort_max": by_name[name].effort_max,
            "velocity_max": by_name[name].velocity_max,
        }
        for name in mapping.sonic_names
    }
    return summary


# --------------------------------------------------------------------------- #
# Launcher safety gates
# --------------------------------------------------------------------------- #


def _is_loopback_literal(token: str) -> bool:
    return token == "localhost" or token.startswith("127.") or token in {"::1", "lo"}


def collect_refusals(
    *,
    cyclonedds_uri: str | None,
    profile: DdsProfile | None,
    domain_id: int,
    sonic_mode: str,
    requested_interface: str,
    argv: Sequence[str],
    physical_interfaces: Sequence[str],
    existing_session: SessionClaim | None,
    own_pid: int,
    expected_domain: int = SIM_DDS_DOMAIN_ID,
) -> tuple[str, ...]:
    """Return every reason the launcher must refuse to start (empty == safe)."""
    reasons: list[str] = []

    if not cyclonedds_uri or not str(cyclonedds_uri).startswith("file://"):
        reasons.append("unsafe_cyclonedds_uri")
    elif profile is None or not profile.is_loopback_only() or profile.allow_multicast:
        reasons.append("unsafe_cyclonedds_uri")

    if int(domain_id) != int(expected_domain):
        reasons.append("unsafe_dds_domain")

    if str(sonic_mode).strip().lower() != "sim":
        reasons.append("unsafe_sonic_mode")
    elif SIM_ONLY_DEPLOY_FLAG not in {str(token) for token in argv}:
        # Declaring sim mode is not enough: the command itself must carry the
        # simulation-only flag, or the deploy would expect a real robot stream.
        reasons.append("unsafe_sonic_mode")

    forbidden = {name for name in physical_interfaces if name != SIM_DDS_INTERFACE}
    unsafe_tokens: list[str] = []
    if str(requested_interface) != SIM_DDS_INTERFACE:
        unsafe_tokens.append(str(requested_interface))
    for token in argv:
        token = str(token)
        if token in forbidden:
            unsafe_tokens.append(token)
        elif not _is_loopback_literal(token) and _looks_like_address(token):
            unsafe_tokens.append(token)
    if unsafe_tokens:
        reasons.append("unsafe_physical_interface")

    if existing_session is not None and existing_session.conflicts_with(
        pid=own_pid, domain_id=int(domain_id), interface=str(requested_interface)
    ):
        reasons.append("session_conflict")

    return tuple(dict.fromkeys(reasons))


def _looks_like_address(token: str) -> bool:
    """True only for a literal IP address, so paths and flags are never matched."""
    try:
        ipaddress.ip_address(token)
    except ValueError:
        return False
    return True


def check_launch_safety(**kwargs) -> None:
    """Raise :class:`ContractError` listing every failed safety gate."""
    reasons = collect_refusals(**kwargs)
    if reasons:
        raise ContractError("launcher safety gates failed: " + ", ".join(reasons))
