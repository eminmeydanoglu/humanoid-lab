"""Diagnostic direct replay of a prepared G1 joint trajectory.

This provider deliberately bypasses SONIC.  It exists as the visual oracle for
dataset joint semantics: the fixed-base robot receives the exact body and Dex3
targets stored in a canonical reference, with only name-based reordering at the
simulator boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..contracts.commands import (
    BODY_COMMAND_SCHEMA,
    DEX3_COMMAND_SCHEMA,
    CompleteRobotCommand,
    JointCommand,
)
from ..datasets.sonic.joints import reorder
from ..datasets.sonic.schema import CanonicalEpisode
from .base import ControllerInterface, RobotStateSample
from .sonic import (
    BODY_EFFORT_LIMIT_NM,
    BODY_JOINT_ORDER,
    HAND_EFFORT_LIMIT_NM,
    SONIC_REFERENCE_JOINT_ORDER,
    deploy_gains,
    hand_joint_names,
)


def _load(path: str | Path) -> CanonicalEpisode:
    data = np.load(Path(path))
    episode = CanonicalEpisode(
        **{name: data[name] for name in CanonicalEpisode.__dataclass_fields__}
    )
    episode.validate()
    return episode


def interface(config: Mapping[str, Any]) -> ControllerInterface:
    if not config.get("reference_path"):
        raise ValueError("trajectory controller requires reference_path")
    return ControllerInterface(
        body_joint_names=BODY_JOINT_ORDER,
        body_effort_limits_nm=BODY_EFFORT_LIMIT_NM,
        hand_kind="dex3",
        left_hand_joint_names=hand_joint_names("left"),
        left_hand_effort_limits_nm=HAND_EFFORT_LIMIT_NM,
        right_hand_joint_names=hand_joint_names("right"),
        right_hand_effort_limits_nm=HAND_EFFORT_LIMIT_NM,
        notes={
            "reference_path": str(config["reference_path"]),
            "reference_semantics": "direct fixed-base visual oracle; SONIC bypassed",
        },
    )


class TrajectoryController:
    kind = "trajectory"

    def __init__(self, config: Mapping[str, Any], *, physics_dt: float, ttl_s: float) -> None:
        self._episode = _load(config["reference_path"])
        self._dt = float(physics_dt)
        self._ttl_ticks = max(1, int(round(float(ttl_s) / self._dt)))
        self._pre_roll_ticks = max(0, int(round(float(config.get("pre_roll_s", 1.0)) / self._dt)))
        self._body_q = reorder(
            self._episode.joint_pos, SONIC_REFERENCE_JOINT_ORDER, BODY_JOINT_ORDER
        )
        self._body_dq = reorder(
            self._episode.joint_vel, SONIC_REFERENCE_JOINT_ORDER, BODY_JOINT_ORDER
        )
        self._kp, self._kd = deploy_gains()
        self._sequence = 0
        self._last_frame = 0
        self._closed = False

    def publish_state(self, state: RobotStateSample) -> None:
        pass

    def poll(self, tick: int) -> CompleteRobotCommand | None:
        if self._closed:
            return None
        elapsed = max(0, tick - self._pre_roll_ticks) * self._dt
        frame = min(int(np.floor(elapsed * 50.0 + 1e-9)), len(self._body_q) - 1)
        self._last_frame = frame
        self._sequence += 1
        valid = tick + self._ttl_ticks
        body = JointCommand.build(
            schema=BODY_COMMAND_SCHEMA,
            sequence=self._sequence,
            joint_names=BODY_JOINT_ORDER,
            q=self._body_q[frame],
            dq=self._body_dq[frame],
            kp=self._kp,
            kd=self._kd,
            valid_until_tick=valid,
        )
        hands = []
        for side, values in (
            ("left", self._episode.left_hand_joints[frame]),
            ("right", self._episode.right_hand_joints[frame]),
        ):
            hands.append(JointCommand.build(
                schema=DEX3_COMMAND_SCHEMA,
                sequence=self._sequence,
                joint_names=hand_joint_names(side),
                q=values,
                kp=(1.5,) * 7,
                kd=(0.1,) * 7,
                valid_until_tick=valid,
            ))
        return CompleteRobotCommand(episode_id=0, body=body, left_hand=hands[0], right_hand=hands[1])

    def status(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "state": "closed" if self._closed else "replaying",
            "frame": self._last_frame,
            "frames": len(self._body_q),
        }

    def close(self) -> None:
        self._closed = True
