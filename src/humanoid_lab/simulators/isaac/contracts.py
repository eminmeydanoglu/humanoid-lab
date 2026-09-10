"""Typed, dependency-free contracts for the Isaac G1 simulator."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class ContractError(ValueError):
    """A declarative profile or runtime value violates the Isaac G1 contract."""


class TimelineState(str, Enum):
    STARTING = "STARTING"
    PAUSED = "PAUSED"
    PLAYING = "PLAYING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class HandBehavior(str, Enum):
    PASSIVE = "passive"


def _tuple(values: Sequence[Any], length: int, name: str) -> tuple[float, ...]:
    if len(values) != length:
        raise ContractError(f"{name} must contain exactly {length} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ContractError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class CameraSpec:
    name: str
    parent_link: str
    width: int
    height: int
    focal_length_mm: float
    horizontal_aperture_mm: float
    position_m: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraSpec":
        parent = str(data["parent_link"])
        if not parent or "/" in parent:
            raise ContractError("camera parent_link must be one link name")
        width, height = (int(value) for value in data["resolution"])
        if width <= 0 or height <= 0:
            raise ContractError("camera resolution must be positive")
        rotation = _tuple(data["rotation_wxyz"], 4, "camera rotation_wxyz")
        if abs(sum(value * value for value in rotation) - 1.0) > 1e-5:
            raise ContractError("camera rotation_wxyz must be normalized")
        return cls(
            name=str(data.get("name", "head_camera")),
            parent_link=parent,
            width=width,
            height=height,
            focal_length_mm=float(data["focal_length_mm"]),
            horizontal_aperture_mm=float(data["horizontal_aperture_mm"]),
            position_m=_tuple(data["position_m"], 3, "camera position_m"),
            rotation_wxyz=rotation,
        )


@dataclass(frozen=True)
class HandSpec:
    kind: str
    dofs: int
    behavior: HandBehavior
    joint_name_patterns: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HandSpec":
        behavior = HandBehavior(str(data["behavior"]))
        dofs = int(data["dofs"])
        if dofs < 0:
            raise ContractError("hand dofs cannot be negative")
        patterns = tuple(str(value) for value in data.get("joint_name_patterns", ()))
        if dofs and not patterns:
            raise ContractError("a hand with joints needs joint_name_patterns")
        return cls(str(data["kind"]), dofs, behavior, patterns)


@dataclass(frozen=True)
class RobotSpec:
    name: str
    body_dofs: int
    asset_kind: str
    asset_reference: str
    asset_provenance: str
    initial_position_m: tuple[float, float, float]
    hand: HandSpec

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RobotSpec":
        body_dofs = int(data["body_dofs"])
        if body_dofs != 29:
            raise ContractError("the Isaac G1 platform requires the G1 29DoF body")
        kind = str(data["asset_kind"])
        if kind not in {"isaaclab_config", "usd"}:
            raise ContractError("asset_kind must be isaaclab_config or usd")
        return cls(
            name=str(data["name"]),
            body_dofs=body_dofs,
            asset_kind=kind,
            asset_reference=str(data["asset_reference"]),
            asset_provenance=str(data["asset_provenance"]),
            initial_position_m=_tuple(data["initial_position_m"], 3, "initial_position_m"),
            hand=HandSpec.from_dict(data["hand"]),
        )


@dataclass(frozen=True)
class RunProfile:
    schema_version: int
    profile_id: str
    robot: RobotSpec
    camera: CameraSpec
    physics_dt: float

    @classmethod
    def load(cls, path: Path) -> "RunProfile":
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot load profile {path}: {exc}") from exc
        if int(data.get("schema_version", 0)) != 1:
            raise ContractError("unsupported Isaac G1 profile schema_version")
        dt = float(data["simulation"]["physics_dt"])
        if not math.isfinite(dt) or dt <= 0.0:
            raise ContractError("physics_dt must be positive and finite")
        return cls(
            schema_version=1,
            profile_id=str(data["profile_id"]),
            robot=RobotSpec.from_dict(data["robot"]),
            camera=CameraSpec.from_dict(data["camera"]),
            physics_dt=dt,
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "robot": {
                "name": self.robot.name,
                "body_dofs": self.robot.body_dofs,
                "asset_kind": self.robot.asset_kind,
                "asset_reference": self.robot.asset_reference,
                "asset_provenance": self.robot.asset_provenance,
                "initial_position_m": list(self.robot.initial_position_m),
                "hand": {
                    "kind": self.robot.hand.kind,
                    "dofs": self.robot.hand.dofs,
                    "behavior": self.robot.hand.behavior.value,
                    "joint_name_patterns": list(self.robot.hand.joint_name_patterns),
                },
            },
            "camera": {
                "name": self.camera.name,
                "parent_link": self.camera.parent_link,
                "resolution": [self.camera.width, self.camera.height],
                "focal_length_mm": self.camera.focal_length_mm,
                "horizontal_aperture_mm": self.camera.horizontal_aperture_mm,
                "position_m": list(self.camera.position_m),
                "rotation_wxyz": list(self.camera.rotation_wxyz),
            },
            "simulation": {"physics_dt": self.physics_dt},
        }


@dataclass(frozen=True)
class RobotState:
    run_id: str
    episode_id: int
    physics_tick: int
    simulated_time: float
    root_position: tuple[float, float, float]
    root_rotation_wxyz: tuple[float, float, float, float]
    body_position: tuple[float, ...]
    body_velocity: tuple[float, ...]
    timeline_state: TimelineState
    control_mode: str = "passive"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "physics_tick": self.physics_tick,
            "simulated_time": self.simulated_time,
            "root_position": list(self.root_position),
            "root_rotation_wxyz": list(self.root_rotation_wxyz),
            "body_position": list(self.body_position),
            "body_velocity": list(self.body_velocity),
            "timeline_state": self.timeline_state.value,
            "control_mode": self.control_mode,
        }


def quaternion_up_z(rotation_wxyz: Sequence[float]) -> float:
    _, x, y, _ = _tuple(rotation_wxyz, 4, "root rotation")
    return 1.0 - 2.0 * (x * x + y * y)
