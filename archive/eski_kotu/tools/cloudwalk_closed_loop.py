"""Versioned loopback contracts for the CloudWalk closed-loop workers."""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Sequence

from sonic_isaac_inspire_adapter import BODY_JOINT_COUNT, ContractError

STATE_MAGIC = b"CWST1"
BODY_MAGIC = b"CWBD1"
STATE_VALUES = 93  # base angular velocity, body q/v, last action, projected gravity
STATE_PACKET_SIZE = 5 + 8 + 8 + 4 * STATE_VALUES
BODY_PACKET_SIZE = 5 + 8 + 8 + 4 * BODY_JOINT_COUNT * 2
ISAACLAB_TO_MUJOCO = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28)
MUJOCO_TO_ISAACLAB = (0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
DEFAULT_ANGLES = (-0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0)


def native_decoder_state(body_q_mujoco: Sequence[float], body_qd_mujoco: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    q = _values(body_q_mujoco, BODY_JOINT_COUNT, "MuJoCo body q")
    qd = _values(body_qd_mujoco, BODY_JOINT_COUNT, "MuJoCo body qd")
    return (
        tuple(q[index] - DEFAULT_ANGLES[index] for index in MUJOCO_TO_ISAACLAB),
        tuple(qd[index] for index in MUJOCO_TO_ISAACLAB),
    )


def _values(value: Sequence[float], count: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (bytes, str)) or len(value) != count:
        raise ContractError(f"{label} must have exactly {count} values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be numeric") from error
    if not all(math.isfinite(item) for item in result):
        raise ContractError(f"{label} contains non-finite values")
    return result


@dataclass(frozen=True)
class SonicState:
    sequence: int
    monotonic_ns: int
    base_angular_velocity: tuple[float, ...]
    body_q: tuple[float, ...]
    body_qd: tuple[float, ...]
    last_body_action: tuple[float, ...]
    projected_gravity: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int) or self.sequence < 0 or not isinstance(self.monotonic_ns, int) or self.monotonic_ns < 0:
            raise ContractError("state sequence and monotonic_ns must be non-negative integers")
        object.__setattr__(self, "base_angular_velocity", _values(self.base_angular_velocity, 3, "base angular velocity"))
        object.__setattr__(self, "body_q", _values(self.body_q, BODY_JOINT_COUNT, "body q"))
        object.__setattr__(self, "body_qd", _values(self.body_qd, BODY_JOINT_COUNT, "body qd"))
        object.__setattr__(self, "last_body_action", _values(self.last_body_action, BODY_JOINT_COUNT, "last body action"))
        object.__setattr__(self, "projected_gravity", _values(self.projected_gravity, 3, "projected gravity"))

    def pack(self) -> bytes:
        values = self.base_angular_velocity + self.body_q + self.body_qd + self.last_body_action + self.projected_gravity
        return struct.pack("<5sQQ93f", STATE_MAGIC, self.sequence, self.monotonic_ns, *values)

    @classmethod
    def unpack(cls, packet: bytes) -> "SonicState":
        if len(packet) != STATE_PACKET_SIZE:
            raise ContractError("SONIC state packet length is invalid")
        magic, sequence, monotonic_ns, *values = struct.unpack("<5sQQ93f", packet)
        if magic != STATE_MAGIC:
            raise ContractError("SONIC state packet magic is invalid")
        return cls(sequence, monotonic_ns, tuple(values[:3]), tuple(values[3:32]), tuple(values[32:61]), tuple(values[61:90]), tuple(values[90:93]))


@dataclass(frozen=True)
class BodyCommand:
    sequence: int
    monotonic_ns: int
    positions: tuple[float, ...]
    raw_actions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int) or self.sequence < 0 or not isinstance(self.monotonic_ns, int) or self.monotonic_ns < 0:
            raise ContractError("body command sequence and monotonic_ns must be non-negative integers")
        object.__setattr__(self, "positions", _values(self.positions, BODY_JOINT_COUNT, "body command"))
        object.__setattr__(self, "raw_actions", _values(self.raw_actions, BODY_JOINT_COUNT, "raw body actions"))

    def pack(self) -> bytes:
        return struct.pack("<5sQQ58f", BODY_MAGIC, self.sequence, self.monotonic_ns, *self.positions, *self.raw_actions)

    @classmethod
    def unpack(cls, packet: bytes) -> "BodyCommand":
        if len(packet) != BODY_PACKET_SIZE:
            raise ContractError("SONIC body packet length is invalid")
        magic, sequence, monotonic_ns, *values = struct.unpack("<5sQQ58f", packet)
        if magic != BODY_MAGIC:
            raise ContractError("SONIC body packet magic is invalid")
        return cls(sequence, monotonic_ns, tuple(values[:BODY_JOINT_COUNT]), tuple(values[BODY_JOINT_COUNT:]))
