"""Deterministic controller used to exercise the simulator's command path.

Its default pose, gains and effort limits are the ones the pinned SONIC
deployment sends, so a failure here separates the command plumbing from the
policy itself: if this controller cannot move the robot, the deployment's
identical commands will not either.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts.commands import (
    BODY_COMMAND_SCHEMA,
    CommandError,
    CompleteRobotCommand,
    JointCommand,
)
from .base import ControllerInterface, RobotStateSample
from .sonic import BODY_JOINT_ORDER, deploy_gains, standing_pose

POSES = ("sonic_standing",)
GAIN_SETS = ("sonic_deploy",)
LIMIT_SETS = ("sonic_body",)


def _names(config: Mapping[str, Any]) -> tuple[str, ...]:
    pose = config.get("pose")
    if pose is not None:
        if pose not in POSES:
            raise CommandError(f"scripted controller pose must be one of {POSES}")
        return BODY_JOINT_ORDER
    targets = config.get("target_pose_rad")
    if not targets:
        raise CommandError("scripted controller needs either pose or target_pose_rad")
    return tuple(str(name) for name in targets)


def _pose(config: Mapping[str, Any]) -> dict[str, float]:
    if config.get("pose") == "sonic_standing":
        pose = standing_pose()
    else:
        pose = {str(name): float(value) for name, value in config["target_pose_rad"].items()}
    # An offset makes the command a target the robot is not already standing in,
    # which is what makes the tracking measurement meaningful.
    for name, offset in config.get("pose_offsets_rad", {}).items():
        if name not in pose:
            raise CommandError(f"scripted controller offset names unknown joint {name!r}")
        pose[name] += float(offset)
    return pose


def _gains(config: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    gain_set = config.get("gains", "sonic_deploy")
    if gain_set not in GAIN_SETS:
        raise CommandError(f"scripted controller gains must be one of {GAIN_SETS}")
    return deploy_gains()


def _limits(config: Mapping[str, Any]) -> tuple[float, ...]:
    from .sonic import BODY_EFFORT_LIMIT_NM

    limit_set = config.get("effort_limits", "sonic_body")
    if limit_set not in LIMIT_SETS:
        raise CommandError(f"scripted controller effort_limits must be one of {LIMIT_SETS}")
    return BODY_EFFORT_LIMIT_NM


def interface(config: Mapping[str, Any]) -> ControllerInterface:
    names = _names(config)
    pose = _pose(config)
    kp, kd = _gains(config)
    limits = _limits(config)
    return ControllerInterface(
        body_joint_names=names,
        body_effort_limits_nm=limits,
        notes={
            "hold_seconds": float(config["hold_seconds"]),
            "pose_source": config.get("pose", "target_pose_rad"),
            "kp_range_rad": [min(kp), max(kp)],
            "kd_range_rad": [min(kd), max(kd)],
            "pose_target_rad": [pose[name] for name in names],
        },
    )


class ScriptedController:
    """Hold a declared pose, then go silent so the timeout can be observed."""

    kind = "scripted"

    def __init__(self, config: Mapping[str, Any], *, physics_dt: float, ttl_s: float) -> None:
        self._dt = float(physics_dt)
        self._ttl_ticks = max(1, int(round(float(ttl_s) / self._dt)))
        self._hold_ticks = max(1, int(round(float(config["hold_seconds"]) / self._dt)))
        self._joint_names = _names(config)
        pose = _pose(config)
        self._target = tuple(float(pose[name]) for name in self._joint_names)
        kp, kd = _gains(config)
        self._kp = _per_joint(kp, len(self._joint_names), "kp")
        self._kd = _per_joint(kd, len(self._joint_names), "kd")
        self._sequence = 0
        self._emitted = 0
        self._stopped = False

    def publish_state(self, state: RobotStateSample) -> None:
        """The scripted controller ignores observations by construction."""

    def poll(self, tick: int) -> CompleteRobotCommand | None:
        if tick > self._hold_ticks:
            self._stopped = True
            return None
        self._sequence += 1
        self._emitted += 1
        body = JointCommand.build(
            schema=BODY_COMMAND_SCHEMA,
            sequence=self._sequence,
            joint_names=self._joint_names,
            q=self._target,
            kp=self._kp,
            kd=self._kd,
            valid_until_tick=tick + self._ttl_ticks,
        )
        return CompleteRobotCommand(episode_id=0, body=body)

    def status(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "state": "stopped" if self._stopped else "emitting",
            "commands_emitted": self._emitted,
            "hold_ticks": self._hold_ticks,
            "ttl_ticks": self._ttl_ticks,
        }

    def close(self) -> None:
        self._stopped = True


def _per_joint(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if len(resolved) != size:
        raise CommandError(f"scripted controller {name} has {len(resolved)} values, expected {size}")
    return resolved
