"""Canonical fail-closed CloudWalk Isaac ↔ upstream SONIC protocol-v4 adapter.

The module owns no DDS, ZMQ socket, policy, or Isaac import.  It exactly emits
upstream v4 packets and returns data for the caller to apply to Isaac Lab.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

ACTION_HORIZON = 40
MOTION_TOKEN_SIZE = 64
HAND_ACTION_SIZE = 7
ACTION_SIZE = 78
BODY_JOINT_COUNT = 29
INSPIRE_HAND_JOINT_COUNT = 24
ISAAC_JOINT_COUNT = 53
ACTION_RATE_HZ = 50.0
INFERENCE_RATE_HZ = 2.5
PROTOCOL_VERSION = 4
HEADER_SIZE = 1280
SONIC_V1_1_DECODER_SHA256 = "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509"

INSPIRE_HAND_JOINTS = (
    "L_index_proximal_joint", "L_middle_proximal_joint", "L_pinky_proximal_joint", "L_ring_proximal_joint", "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint", "R_middle_proximal_joint", "R_pinky_proximal_joint", "R_ring_proximal_joint", "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint", "L_middle_intermediate_joint", "L_pinky_intermediate_joint", "L_ring_intermediate_joint", "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint", "R_middle_intermediate_joint", "R_pinky_intermediate_joint", "R_ring_intermediate_joint", "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint", "R_thumb_intermediate_joint", "L_thumb_distal_joint", "R_thumb_distal_joint",
)


class ContractError(ValueError):
    """Raised when a boundary input is malformed, stale, or unproven."""


def _finite_tuple(values: Any, size: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != size:
        raise ContractError(f"{name} must contain exactly {size} values")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{name} values must be numeric") from error
    if not all(math.isfinite(value) for value in result):
        raise ContractError(f"{name} contains non-finite values")
    return result


@dataclass(frozen=True)
class V4Action:
    motion_token: tuple[float, ...]
    left_hand: tuple[float, ...]
    right_hand: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "motion_token", _finite_tuple(self.motion_token, 64, "motion_token"))
        object.__setattr__(self, "left_hand", _finite_tuple(self.left_hand, 7, "left_hand"))
        object.__setattr__(self, "right_hand", _finite_tuple(self.right_hand, 7, "right_hand"))


def split_groot_action_chunk(value: Any) -> tuple[V4Action, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != ACTION_HORIZON:
        raise ContractError("GR00T action must have shape [40,78]")
    actions: list[V4Action] = []
    for index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != ACTION_SIZE:
            raise ContractError(f"GR00T action must have shape [40,78]; row {index} is malformed")
        actions.append(V4Action(tuple(row[:64]), tuple(row[64:71]), tuple(row[71:78])))
    return tuple(actions)


def pack_protocol_v4(action: V4Action, frame_index: int) -> bytes:
    """Byte-for-byte equivalent to upstream pack_latent_action_message."""
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ContractError("frame index must be a non-negative integer")
    fields = [
        {"name": "token_state", "dtype": "f32", "shape": [1, 64]},
        {"name": "frame_index", "dtype": "i64", "shape": [1]},
        {"name": "left_hand_joints", "dtype": "f32", "shape": [1, 7]},
        {"name": "right_hand_joints", "dtype": "f32", "shape": [1, 7]},
    ]
    header = json.dumps({"v": 4, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":")).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ContractError("v4 header exceeds upstream fixed header size")
    return b"pose" + header.ljust(HEADER_SIZE, b"\0") + struct.pack("<64fq7f7f", *action.motion_token, frame_index, *action.left_hand, *action.right_hand)


def unpack_protocol_v4(packet: bytes) -> tuple[V4Action, int]:
    if not isinstance(packet, bytes) or len(packet) != 4 + HEADER_SIZE + 320 or not packet.startswith(b"pose"):
        raise ContractError("packet is not an upstream protocol-v4 pose frame")
    try:
        header = json.loads(packet[4:4 + HEADER_SIZE].rstrip(b"\0"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("packet has malformed protocol-v4 header") from error
    expected = [("token_state", "f32", [1, 64]), ("frame_index", "i64", [1]), ("left_hand_joints", "f32", [1, 7]), ("right_hand_joints", "f32", [1, 7])]
    actual = [(item.get("name"), item.get("dtype"), item.get("shape")) for item in header.get("fields", []) if isinstance(item, dict)]
    if header.get("v") != 4 or header.get("endian") != "le" or header.get("count") != 1 or actual != expected:
        raise ContractError("packet does not match upstream protocol-v4 latent schema")
    values = struct.unpack("<64fq7f7f", packet[4 + HEADER_SIZE:])
    return V4Action(values[:64], values[65:72], values[72:79]), values[64]


@dataclass(frozen=True)
class SafeReference:
    action: V4Action
    decoder_sha256: str
    source: str

    @classmethod
    def load(cls, path: Path) -> "SafeReference":
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict):
                raise TypeError("manifest root is not an object")
            return cls(V4Action(tuple(record["motion_token"]), tuple(record["left_hand"]), tuple(record["right_hand"])), str(record["decoder_sha256"]), str(record["source"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ContractError) as error:
            raise ContractError(f"malformed safe-reference manifest {path}: {error}") from error

    def validate(self) -> None:
        if self.decoder_sha256 != SONIC_V1_1_DECODER_SHA256:
            raise ContractError("safe reference decoder hash does not match pinned sonic_v1_1")
        if not self.source.startswith("recorded-upstream-sonic-"):
            raise ContractError("safe reference must identify a recorded upstream SONIC source")
        if not any(self.action.motion_token):
            raise ContractError("zero 64D token is never a safe hold reference")


class Lifecycle(Enum):
    RESET = "reset"
    HOLDING = "holding"
    RUNNING = "running"
    PAUSED = "paused"
    TIMED_OUT = "timed_out"
    STOPPED = "stopped"


@dataclass
class LifecycleGuard:
    watchdog_seconds: float = 0.25
    lifecycle: Lifecycle = Lifecycle.RESET
    reference: SafeReference | None = None
    last_action_time: float | None = None

    def initialize(self, reference: SafeReference) -> None:
        if self.lifecycle is not Lifecycle.RESET:
            raise ContractError("initialize is only allowed from RESET")
        reference.validate(); self.reference = reference; self.lifecycle = Lifecycle.HOLDING

    def start(self, now: float) -> None:
        if self.lifecycle not in (Lifecycle.HOLDING, Lifecycle.PAUSED) or self.reference is None:
            raise ContractError("start is only allowed from HOLDING or PAUSED with a safe reference")
        self.lifecycle, self.last_action_time = Lifecycle.RUNNING, now

    def pause(self) -> None:
        if self.lifecycle is not Lifecycle.RUNNING:
            raise ContractError("pause is only allowed from RUNNING")
        self.lifecycle = Lifecycle.PAUSED

    def stop(self) -> None:
        if self.lifecycle not in (Lifecycle.HOLDING, Lifecycle.RUNNING, Lifecycle.PAUSED, Lifecycle.TIMED_OUT):
            raise ContractError("stop is not allowed from current lifecycle state")
        self.lifecycle = Lifecycle.STOPPED

    def reset(self) -> None:
        if self.lifecycle is Lifecycle.RUNNING:
            raise ContractError("pause or stop before reset")
        self.lifecycle, self.reference, self.last_action_time = Lifecycle.RESET, None, None

    def tick(self, now: float) -> Lifecycle:
        if self.lifecycle is Lifecycle.RUNNING and self.last_action_time is not None and now - self.last_action_time > self.watchdog_seconds:
            self.lifecycle = Lifecycle.TIMED_OUT
        return self.lifecycle

    def accept_validated(self, now: float) -> None:
        if self.tick(now) is not Lifecycle.RUNNING:
            raise ContractError("action rejected outside active simulator lifecycle")
        self.last_action_time = now

    def output(self, now: float) -> V4Action:
        state = self.tick(now)
        if state in (Lifecycle.HOLDING, Lifecycle.PAUSED, Lifecycle.TIMED_OUT) and self.reference:
            return self.reference.action
        raise ContractError("no simulator command is allowed in current lifecycle")


class InspireFTPGripMapper:
    """Map verified normalized CloudWalk closures to Inspire FTP target positions."""
    evidence = "scene_parameters.json hand_action_contract; Raider right-thumb controlled joint-state probe"

    def __init__(self, dataset_channel_order_verified: bool = False, right_thumb_yaw_closes_at_upper: bool = False) -> None:
        self.dataset_channel_order_verified = dataset_channel_order_verified
        self.right_thumb_yaw_closes_at_upper = right_thumb_yaw_closes_at_upper

    @staticmethod
    def _side(prefix: str, values: tuple[float, ...]) -> dict[str, float]:
        thumb_yaw, thumb_pitch, thumb_distal, index_prox, index_intermediate, middle_prox, middle_intermediate = values
        return {
            f"{prefix}_index_proximal_joint": index_prox, f"{prefix}_middle_proximal_joint": middle_prox,
            f"{prefix}_pinky_proximal_joint": middle_prox, f"{prefix}_ring_proximal_joint": middle_prox,
            f"{prefix}_thumb_proximal_yaw_joint": thumb_yaw, f"{prefix}_index_intermediate_joint": index_intermediate,
            f"{prefix}_middle_intermediate_joint": middle_intermediate, f"{prefix}_pinky_intermediate_joint": middle_intermediate,
            f"{prefix}_ring_intermediate_joint": middle_intermediate, f"{prefix}_thumb_proximal_pitch_joint": thumb_pitch,
            f"{prefix}_thumb_intermediate_joint": thumb_distal, f"{prefix}_thumb_distal_joint": thumb_distal,
        }

    def normalized_targets(self, action: V4Action) -> tuple[float, ...]:
        if not self.dataset_channel_order_verified or not self.right_thumb_yaw_closes_at_upper:
            raise ContractError("Inspire hand physics application requires verified channel order and right-thumb closure sign")
        values = self._side("L", action.left_hand) | self._side("R", action.right_hand)
        if set(values) != set(INSPIRE_HAND_JOINTS):
            raise ContractError("Inspire mapping does not cover exactly 24 articulation joints")
        if not all(0.0 <= value <= 1.0 for value in values.values()):
            raise ContractError("CloudWalk hand closures must be normalized to [0,1]")
        return tuple(values[name] for name in INSPIRE_HAND_JOINTS)

    def targets(self, action: V4Action, joint_limits: Sequence[tuple[float, float]], open_positions: Sequence[float] | None = None) -> tuple[float, ...]:
        normalized = self.normalized_targets(action)
        if len(joint_limits) != INSPIRE_HAND_JOINT_COUNT:
            raise ContractError("Inspire joint limits must cover exactly 24 articulation joints")
        opens = (0.0,) * INSPIRE_HAND_JOINT_COUNT if open_positions is None else _finite_tuple(open_positions, INSPIRE_HAND_JOINT_COUNT, "Inspire open positions")
        targets = []
        for name, closure, limits, opened in zip(INSPIRE_HAND_JOINTS, normalized, joint_limits, opens, strict=True):
            lower, upper = _finite_tuple(limits, 2, f"limits for {name}")
            if lower >= upper:
                raise ContractError(f"invalid joint limits for {name}")
            opened = min(max(opened, lower), upper)
            if name == "R_thumb_proximal_yaw_joint":
                closed = upper
            else:
                closed = lower if opened - lower >= upper - opened else upper
            targets.append(opened + closure * (closed - opened))
        return tuple(targets)


@dataclass(frozen=True)
class IsaacCommand:
    protocol_v4: bytes
    frame_index: int
    motion_token: tuple[float, ...]
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]


@dataclass
class SimulatorV4Endpoint:
    guard: LifecycleGuard
    mapper: InspireFTPGripMapper
    joint_limits: tuple[tuple[float, float], ...] = ()
    frame_index: int = 0

    def submit(self, action: V4Action, now: float) -> IsaacCommand:
        joints = self.mapper.targets(action, self.joint_limits)
        packet = pack_protocol_v4(action, self.frame_index)
        self.guard.accept_validated(now)
        result = IsaacCommand(packet, self.frame_index, action.motion_token, INSPIRE_HAND_JOINTS, joints)
        self.frame_index += 1
        return result

    def safe_command(self, now: float) -> IsaacCommand:
        action = self.guard.output(now)
        return IsaacCommand(pack_protocol_v4(action, self.frame_index), self.frame_index, action.motion_token, (), ())


@dataclass
class Scheduler:
    next_action_at: float | None = None
    last_inference_at: float | None = None

    def install_chunk(self, now: float) -> None:
        self.next_action_at = now
        self.last_inference_at = now

    def action_due(self, now: float) -> bool:
        return self.next_action_at is not None and now >= self.next_action_at

    def consume_action(self, now: float) -> None:
        if not self.action_due(now):
            raise ContractError("no scheduled action is due")
        self.next_action_at = now + 1.0 / ACTION_RATE_HZ

    def inference_due(self, now: float) -> bool:
        return self.last_inference_at is None or now - self.last_inference_at >= 1.0 / INFERENCE_RATE_HZ

    def mark_inference(self, now: float) -> None:
        self.last_inference_at = now
