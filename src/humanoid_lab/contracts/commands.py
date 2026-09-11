"""Typed joint commands shared by every controller and the simulator.

A controller is anything that can hand the simulator a set of body and hand
joint commands.  The simulator owns the physics loop, applies those commands
while they are valid, and falls back to passive behaviour when they stop
arriving.  Nothing here imports a simulator, a DDS stack or a policy runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

BODY_COMMAND_SCHEMA = "g1_body_lowcmd_v1"
DEX3_COMMAND_SCHEMA = "dex3_joint_command_v1"
INSPIRE_COMMAND_SCHEMA = "inspire_ftp_joint_command_v1"


class CommandError(ValueError):
    """A command producer or a joint layout violates the declared contract."""


def _finite(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise CommandError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class JointLayout:
    """Declared joint order plus where each name lives in the articulation.

    The layout is the only place a name is mapped to an index.  Mapping is
    resolved once, by name, and fails closed: a missing, duplicated or
    unexpected joint is an error rather than a silent reordering.
    """

    names: tuple[str, ...]
    indices: tuple[int, ...]

    @classmethod
    def resolve(cls, names: Sequence[str], available: Sequence[str], *, what: str) -> "JointLayout":
        if len(set(names)) != len(names):
            raise CommandError(f"{what} declares duplicate joint names")
        lookup: dict[str, list[int]] = {}
        for index, name in enumerate(available):
            lookup.setdefault(name, []).append(index)
        resolved: list[int] = []
        for name in names:
            matches = lookup.get(name, [])
            if not matches:
                raise CommandError(f"{what} joint {name!r} is not present in the asset")
            if len(matches) > 1:
                raise CommandError(f"{what} joint {name!r} is ambiguous in the asset")
            resolved.append(matches[0])
        return cls(tuple(names), tuple(resolved))

    def __len__(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class JointCommand:
    """One group of joints (the body, or a single hand) with the same schema."""

    schema: str
    sequence: int
    joint_names: tuple[str, ...]
    q: tuple[float, ...]
    dq: tuple[float, ...]
    tau: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    valid_until_tick: int

    @classmethod
    def build(
        cls,
        *,
        schema: str,
        sequence: int,
        joint_names: Sequence[str],
        q: Sequence[float],
        dq: Sequence[float] | None = None,
        tau: Sequence[float] | None = None,
        kp: Sequence[float] | None = None,
        kd: Sequence[float] | None = None,
        valid_until_tick: int,
    ) -> "JointCommand":
        names = tuple(str(name) for name in joint_names)
        if not names:
            raise CommandError("a joint command needs at least one joint")
        size = len(names)
        zeros = (0.0,) * size
        return cls(
            schema=schema,
            sequence=int(sequence),
            joint_names=names,
            q=_finite(q, "q"),
            dq=_finite(zeros if dq is None else dq, "dq"),
            tau=_finite(zeros if tau is None else tau, "tau"),
            kp=_finite(zeros if kp is None else kp, "kp"),
            kd=_finite(zeros if kd is None else kd, "kd"),
            valid_until_tick=int(valid_until_tick),
        )

    def validate(self, layout: JointLayout) -> None:
        """Fail closed unless this command matches the declared simulator layout."""
        if self.joint_names != layout.names:
            raise CommandError(
                f"{self.schema} joint names do not match the declared layout: "
                f"expected {layout.names}, got {self.joint_names}"
            )
        size = len(layout)
        for field in ("q", "dq", "tau", "kp", "kd"):
            values = getattr(self, field)
            if len(values) != size:
                raise CommandError(f"{self.schema}.{field} has {len(values)} values, expected {size}")

    def is_valid_at(self, tick: int) -> bool:
        return tick <= self.valid_until_tick


@dataclass(frozen=True)
class CompleteRobotCommand:
    """Everything the simulator needs to drive one robot for one tick."""

    episode_id: int
    body: JointCommand
    left_hand: JointCommand | None = None
    right_hand: JointCommand | None = None

    @property
    def sequence(self) -> int:
        return self.body.sequence

    def is_valid_at(self, tick: int) -> bool:
        return self.body.is_valid_at(tick) and all(
            hand is None or hand.is_valid_at(tick) for hand in (self.left_hand, self.right_hand)
        )


def resolve_layouts(
    *,
    available_joints: Sequence[str],
    body_names: Sequence[str],
    left_hand_names: Sequence[str] = (),
    right_hand_names: Sequence[str] = (),
) -> tuple[JointLayout, JointLayout | None, JointLayout | None]:
    """Resolve the profile's declared joint orders against the asset."""
    body = JointLayout.resolve(body_names, available_joints, what="body")
    left = (
        JointLayout.resolve(left_hand_names, available_joints, what="left hand")
        if left_hand_names
        else None
    )
    right = (
        JointLayout.resolve(right_hand_names, available_joints, what="right hand")
        if right_hand_names
        else None
    )
    return body, left, right


def body_layout_from_profile(controller: Mapping[str, Any]) -> tuple[str, ...]:
    names = controller.get("body_joint_names")
    if not names:
        raise CommandError("controller.body_joint_names is required when a controller is selected")
    return tuple(str(name) for name in names)
