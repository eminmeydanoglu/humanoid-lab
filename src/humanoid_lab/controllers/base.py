"""Controller interface seen by the simulator.

A controller publishes the robot state it observes and hands back the newest
command it has.  Both directions cross a thread or process boundary owned by the
controller itself; the simulator never waits for a controller and never lets one
step physics.

``ControllerInterface`` is the other half of the contract: the joint order and
effort limits a provider speaks, declared by the provider so a profile does not
have to restate them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RobotStateSample:
    """One observation published to a controller, in device-independent units.

    Body values are already reordered into the controller's declared body joint
    order, and each hand into its declared hand order, so a controller never has
    to know how the articulation stores its joints.
    """

    episode_id: int
    physics_tick: int
    simulated_time_s: float
    root_position_m: tuple[float, float, float]
    root_quaternion_wxyz: tuple[float, float, float, float]
    root_linear_velocity_mps: tuple[float, float, float]
    root_angular_velocity_rps: tuple[float, float, float]
    body_q: tuple[float, ...]
    body_dq: tuple[float, ...]
    body_tau: tuple[float, ...]
    torso_quaternion_wxyz: tuple[float, float, float, float]
    torso_angular_velocity_rps: tuple[float, float, float]
    left_hand_q: tuple[float, ...] = ()
    left_hand_dq: tuple[float, ...] = ()
    right_hand_q: tuple[float, ...] = ()
    right_hand_dq: tuple[float, ...] = ()


@dataclass(frozen=True)
class ControllerInterface:
    """Joint order and effort limits a controller provider speaks."""

    body_joint_names: tuple[str, ...]
    body_effort_limits_nm: tuple[float, ...] = ()
    hand_kind: str = ""
    left_hand_joint_names: tuple[str, ...] = ()
    left_hand_effort_limits_nm: tuple[float, ...] = ()
    right_hand_joint_names: tuple[str, ...] = ()
    right_hand_effort_limits_nm: tuple[float, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)


class ControllerSource(Protocol):
    """What the simulator needs from any controller."""

    kind: str

    def publish_state(self, state: RobotStateSample) -> None: ...

    def poll(self, tick: int) -> Any: ...

    def status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
