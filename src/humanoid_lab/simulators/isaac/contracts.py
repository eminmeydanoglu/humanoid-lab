"""Typed, dependency-free contracts for the Isaac G1 simulator."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?)$")
DEFAULT_DEVICE = "cpu"
DEFAULT_RENDER_INTERVAL = 8
# Long enough to ride out a slow control tick, short enough that a vanished
# controller leaves the robot passive well before it could look controlled.
DEFAULT_COMMAND_TTL_S = 0.25


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


SUPPORT_KINDS = ("pelvis_band",)
SUPPORT_RELEASES = ("controller_hold_then_motion",)


@dataclass(frozen=True)
class SupportSpec:
    """Declared start-up support, mirroring the official simulator's holder.

    The reference MuJoCo loop hangs the robot from a virtual spring band at the
    waist and the operator releases it only after the controller is running.
    Without that, a position-controlled humanoid is on the floor before the
    controller starts. This is declared, logged, time-bounded, and absent
    whenever no controller is attached.
    """

    kind: str
    release: str
    max_seconds: float
    point_m: tuple[float, float, float]
    linear_stiffness: float
    linear_damping: float
    angular_stiffness: float
    angular_damping: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SupportSpec":
        kind = str(data["kind"])
        if kind not in SUPPORT_KINDS:
            raise ContractError(f"support.kind must be one of {SUPPORT_KINDS}")
        release = str(data["release"])
        if release not in SUPPORT_RELEASES:
            raise ContractError(f"support.release must be one of {SUPPORT_RELEASES}")
        max_seconds = float(data["max_seconds"])
        if not math.isfinite(max_seconds) or max_seconds <= 0.0:
            raise ContractError("support.max_seconds must be positive and finite")
        return cls(
            kind=kind,
            release=release,
            max_seconds=max_seconds,
            point_m=_tuple(data["point_m"], 3, "support.point_m"),
            linear_stiffness=float(data["linear_stiffness"]),
            linear_damping=float(data["linear_damping"]),
            angular_stiffness=float(data["angular_stiffness"]),
            angular_damping=float(data["angular_damping"]),
        )


NAMED_POSES = ("sonic_standing",)


@dataclass(frozen=True)
class RunProfile:
    schema_version: int
    profile_id: str
    robot: RobotSpec
    camera: CameraSpec
    physics_dt: float
    device: str = DEFAULT_DEVICE
    render_interval: int = DEFAULT_RENDER_INTERVAL
    controller: Mapping[str, Any] | None = None
    initial_pose: str | None = None
    support: SupportSpec | None = None

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
        controller = data.get("controller")
        if controller is not None:
            if not isinstance(controller, dict) or "provider" not in controller:
                raise ContractError("controller must be an object with a provider")
            controller = dict(controller)
            ttl = float(controller.get("command_ttl_s", DEFAULT_COMMAND_TTL_S))
            if not math.isfinite(ttl) or ttl <= 0.0:
                raise ContractError("controller.command_ttl_s must be positive and finite")
            controller["command_ttl_s"] = ttl
            if "torso_link" not in controller:
                raise ContractError("controller.torso_link is required when a controller is selected")
        initial_pose = data.get("initial_pose")
        if initial_pose is not None and str(initial_pose) not in NAMED_POSES:
            raise ContractError(f"initial_pose must be one of {NAMED_POSES}")
        support = data.get("support")
        if support is not None and controller is None:
            raise ContractError("a support band only means something with a controller attached")
        return cls(
            schema_version=1,
            profile_id=str(data["profile_id"]),
            robot=RobotSpec.from_dict(data["robot"]),
            camera=CameraSpec.from_dict(data["camera"]),
            physics_dt=dt,
            controller=controller,
            initial_pose=str(initial_pose) if initial_pose is not None else None,
            support=SupportSpec.from_dict(support) if support is not None else None,
        )

    @property
    def command_ttl_s(self) -> float:
        """How long a commanded drive stays valid without a refresh."""
        if not self.controller:
            return DEFAULT_COMMAND_TTL_S
        return float(self.controller["command_ttl_s"])

    def with_device(self, device: str | None) -> "RunProfile":
        """Return the profile with an explicit CLI device override applied."""
        if device is None:
            return self
        if not DEVICE_PATTERN.match(device):
            raise ContractError("simulation.device must be cpu or cuda[:N]")
        return replace(self, device=device)

    @property
    def camera_update_period(self) -> float:
        """Sensor period that keeps the head camera aligned with the render cadence."""
        return self.physics_dt * self.render_interval

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
            "simulation": {
                "physics_dt": self.physics_dt,
                "device": self.device,
                "render_interval": self.render_interval,
            },
            "controller": dict(self.controller) if self.controller else None,
            "initial_pose": self.initial_pose,
            "support": (
                {"kind": self.support.kind, "release": self.support.release} if self.support else None
            ),
        }


def quaternion_up_z(rotation_wxyz: Sequence[float]) -> float:
    _, x, y, _ = _tuple(rotation_wxyz, 4, "root rotation")
    return 1.0 - 2.0 * (x * x + y * y)
