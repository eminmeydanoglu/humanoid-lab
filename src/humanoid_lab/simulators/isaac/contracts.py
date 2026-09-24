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


#: The three dynamic cube colors the block-stacking scene contract requires.
SCENE_CUBE_COLORS = ("red", "yellow", "blue")
SCENE_TARGET_KINDS = ("tape", "plate")


@dataclass(frozen=True)
class ObjectSpec:
    """A single primitive prop for the two plate-placement evaluations."""

    name: str
    shape: str
    size_m: tuple[float, float, float]
    mass_kg: float
    position_m: tuple[float, float, float]
    diffuse_rgb: tuple[float, float, float]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], surface_height_m: float) -> "ObjectSpec":
        name = str(data["name"])
        shape = str(data["shape"])
        if (name, shape) not in {("apple", "sphere"), ("gum", "cuboid")}:
            raise ContractError("scene.object requires an apple sphere or gum cuboid")
        size = _tuple(data["size_m"], 3, "object.size_m")
        position = _tuple(data["position_m"], 3, "object.position_m")
        rgb = _tuple(data["diffuse_rgb"], 3, "object.diffuse_rgb")
        mass = float(data["mass_kg"])
        if not all(value > 0 for value in size) or not all(0 <= value <= 1 for value in rgb) or not math.isfinite(mass) or mass <= 0:
            raise ContractError("scene.object size, mass or color is invalid")
        if abs(position[2] - surface_height_m - size[2] / 2) > 1e-3:
            raise ContractError("scene.object must rest on the declared table surface")
        return cls(name, shape, size, mass, position, rgb)


@dataclass(frozen=True)
class TableSpec:
    """One Unitree table asset with an explicitly declared worktop height.

    The surface height is a required field rather than a default: placing cubes
    relative to a guessed height is exactly the silent failure this contract
    exists to prevent.  Its provenance records how the value was derived.
    """

    asset_reference: str
    asset_provenance: str
    position_m: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    surface_height_m: float
    surface_height_provenance: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableSpec":
        asset_reference = str(data["asset_reference"])
        if not asset_reference.startswith("/"):
            raise ContractError("table asset_reference must be an absolute container path")
        rotation = _tuple(data.get("rotation_wxyz", (1.0, 0.0, 0.0, 0.0)), 4, "table.rotation_wxyz")
        if abs(sum(value * value for value in rotation) - 1.0) > 1e-5:
            raise ContractError("table.rotation_wxyz must be normalized")
        surface_height = float(data["surface_height_m"])
        if not math.isfinite(surface_height) or surface_height <= 0.0:
            raise ContractError("table.surface_height_m must be positive and finite")
        provenance = str(data.get("surface_height_provenance", "")).strip()
        if not provenance:
            raise ContractError("table.surface_height_provenance must state how the height was derived")
        return cls(
            asset_reference=asset_reference,
            asset_provenance=str(data["asset_provenance"]),
            position_m=_tuple(data["position_m"], 3, "table.position_m"),
            rotation_wxyz=rotation,
            surface_height_m=surface_height,
            surface_height_provenance=provenance,
        )


@dataclass(frozen=True)
class CubeSpec:
    """One dynamic cube resting on the declared table surface.

    ``color`` is the cube's identity: it names the prim, the profile entry and
    every telemetry key.  ``diffuse_rgb`` is an opt-in override of the *rendered*
    material only, for a scene variant that must change how a cube looks without
    changing what it is -- its pose, size, mass, physics material, prim name and
    telemetry key all stay exactly as declared.  Absent means the shipped
    colour-keyed material.
    """

    color: str
    size_m: tuple[float, float, float]
    mass_kg: float
    position_m: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    diffuse_rgb: tuple[float, float, float] | None = None
    diffuse_rgb_provenance: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], surface_height_m: float) -> "CubeSpec":
        color = str(data["color"])
        if color not in SCENE_CUBE_COLORS:
            raise ContractError(f"cube.color must be one of {SCENE_CUBE_COLORS}")
        size = _tuple(data["size_m"], 3, "cube.size_m")
        if not all(value > 0.0 for value in size):
            raise ContractError("cube.size_m must be positive")
        mass = float(data["mass_kg"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ContractError("cube.mass_kg must be positive and finite")
        position = _tuple(data["position_m"], 3, "cube.position_m")
        rotation = _tuple(data.get("rotation_wxyz", (1.0, 0.0, 0.0, 0.0)), 4, "cube.rotation_wxyz")
        if abs(sum(value * value for value in rotation) - 1.0) > 1e-5:
            raise ContractError("cube.rotation_wxyz must be normalized")
        expected_z = surface_height_m + size[2] / 2.0
        if abs(position[2] - expected_z) > 1e-3:
            raise ContractError(
                f"{color} cube rests at z={position[2]} but the declared table surface "
                f"{surface_height_m} puts a {size[2]} m cube center at {expected_z}"
            )
        override = data.get("diffuse_rgb")
        diffuse_rgb = None
        if override is not None:
            diffuse_rgb = _tuple(override, 3, "cube.diffuse_rgb")
            if not all(0.0 <= value <= 1.0 for value in diffuse_rgb):
                raise ContractError("cube.diffuse_rgb values must be in [0, 1]")
            provenance = str(data.get("diffuse_rgb_provenance", "")).strip()
            if not provenance:
                raise ContractError("cube.diffuse_rgb_provenance must state how the value was derived")
        return cls(color=color, size_m=size, mass_kg=mass, position_m=position, rotation_wxyz=rotation,
                   diffuse_rgb=diffuse_rgb,
                   diffuse_rgb_provenance=(str(data["diffuse_rgb_provenance"]).strip()
                                           if diffuse_rgb is not None else None))


@dataclass(frozen=True)
class TargetSpec:
    """A flat black tape marker on the declared table surface."""

    kind: str
    color: str
    size_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], surface_height_m: float) -> "TargetSpec":
        kind = str(data["kind"])
        if kind not in SCENE_TARGET_KINDS:
            raise ContractError(f"target.kind must be one of {SCENE_TARGET_KINDS}")
        color = str(data["color"])
        if (kind == "tape" and color != "black") or (kind == "plate" and color not in {"pink", "teal"}):
            raise ContractError("target color does not match its kind")
        size = _tuple(data["size_m"], 3, "target.size_m")
        if not all(value > 0.0 for value in size):
            raise ContractError("target.size_m must be positive")
        if kind == "plate" and abs(size[0] - size[1]) > 1e-6:
            raise ContractError("plate target diameter must be equal in x and y")
        position = _tuple(data["position_m"], 3, "target.position_m")
        rotation = _tuple(data.get("rotation_wxyz", (1.0, 0.0, 0.0, 0.0)), 4, "target.rotation_wxyz")
        if abs(sum(value * value for value in rotation) - 1.0) > 1e-5:
            raise ContractError("target.rotation_wxyz must be normalized")
        expected_z = surface_height_m + size[2] / 2.0
        if abs(position[2] - expected_z) > 1e-3:
            raise ContractError(
                f"target rests at z={position[2]} but the declared table surface "
                f"{surface_height_m} puts a {size[2]} m marker center at {expected_z}"
            )
        return cls(kind=kind, color=color, size_m=size, position_m=position, rotation_wxyz=rotation)


@dataclass(frozen=True)
class SceneSpec:
    """Optional declarative scene: one Unitree table, cubes, a tape target.

    The scene is data only.  It enables the camera service but never adds a
    controller or an RL loop: how the scene is driven stays a profile choice.
    """

    table: TableSpec
    cubes: tuple[CubeSpec, ...]
    object: ObjectSpec | None
    target: TargetSpec
    camera_enabled: bool
    ground_plane: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneSpec":
        table = TableSpec.from_dict(data["table"])
        raw_cubes = data.get("cubes", [])
        if not isinstance(raw_cubes, list) or len(raw_cubes) not in {0, len(SCENE_CUBE_COLORS)}:
            raise ContractError("scene.cubes must list zero or exactly three cubes")
        cubes = tuple(CubeSpec.from_dict(entry, table.surface_height_m) for entry in raw_cubes)
        colors = [cube.color for cube in cubes]
        if colors and sorted(colors) != sorted(SCENE_CUBE_COLORS):
            raise ContractError(f"scene.cubes colors must be exactly {SCENE_CUBE_COLORS}")
        if len(set(colors)) != len(colors):
            raise ContractError("scene.cubes colors must be unique")
        target = TargetSpec.from_dict(data["target"], table.surface_height_m)
        obj = ObjectSpec.from_dict(data["object"], table.surface_height_m) if "object" in data else None
        if (bool(cubes), obj is not None, target.kind) not in {(True, False, "tape"), (False, True, "plate")}:
            raise ContractError("scene requires either three cubes and tape or one object and plate")
        return cls(
            table=table,
            cubes=cubes,
            object=obj,
            target=target,
            camera_enabled=bool(data.get("camera_enabled", False)),
            ground_plane=bool(data.get("ground_plane", True)),
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
    fixed_base: bool = False
    #: Explicit gravity switch; defaults to the fixed-base behaviour.  A
    #: kinematic replay writes every joint each tick, so gravity would otherwise
    #: make the joints sag between the write and the measurement.
    disable_gravity: bool | None = None
    #: PhysX self-collision for the articulation.  ``None`` leaves the USD
    #: asset's own authored value in place, which is what every historic profile
    #: did.  It is declared because the training rig sets it explicitly
    #: (``enabled_self_collisions=True`` in ``G1_CYLINDER_MODEL_12_DEX_CFG``)
    #: and because ``UsdFileCfg`` here is built from ``G1_29DOF_CFG``, whose own
    #: value is ``False`` -- so an undeclared evaluation runs the opposite of the
    #: training setting rather than an unset one.
    self_collisions: bool | None = None

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
            fixed_base=bool(data.get("fixed_base", False)),
            disable_gravity=(
                None if data.get("disable_gravity") is None else bool(data["disable_gravity"])
            ),
            self_collisions=(
                None if data.get("self_collisions") is None else bool(data["self_collisions"])
            ),
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
    scene: SceneSpec | None = None

    @classmethod
    def load(cls, path: Path) -> "RunProfile":
        data = cls._load_data(path, set())
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
            scene=SceneSpec.from_dict(data["scene"]) if data.get("scene") is not None else None,
        )

    @classmethod
    def _load_data(cls, path: Path, seen: set[Path]) -> dict[str, Any]:
        """Inherit shared plant and camera fields; task scenes replace whole scenes."""
        path = path.resolve()
        if path in seen:
            raise ContractError(f"cyclic Isaac G1 profile inheritance at {path}")
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot load profile {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ContractError(f"profile {path} must be a JSON object")
        base_name = data.pop("base_profile", None)
        if base_name is None:
            return data
        if not isinstance(base_name, str) or Path(base_name).name != base_name:
            raise ContractError("base_profile must be a filename beside the profile")
        base = cls._load_data(path.parent / base_name, seen | {path})
        return {**base, **data}

    @property
    def camera_service_enabled(self) -> bool:
        """Whether the profile asks for the head-camera service to run."""
        return self.scene is not None and self.scene.camera_enabled

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
                "fixed_base": self.robot.fixed_base,
                "disable_gravity": self.robot.disable_gravity,
                "self_collisions": self.robot.self_collisions,
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
            "scene": (
                {
                    "camera_enabled": self.scene.camera_enabled,
                    "ground_plane": self.scene.ground_plane,
                    "table": {
                        "asset_reference": self.scene.table.asset_reference,
                        "position_m": list(self.scene.table.position_m),
                        "rotation_wxyz": list(self.scene.table.rotation_wxyz),
                        "surface_height_m": self.scene.table.surface_height_m,
                        "surface_height_provenance": self.scene.table.surface_height_provenance,
                    },
                    "cubes": [
                        {"color": cube.color, "size_m": list(cube.size_m), "position_m": list(cube.position_m)}
                        for cube in self.scene.cubes
                    ],
                    "object": (
                        {"name": self.scene.object.name, "shape": self.scene.object.shape,
                         "size_m": list(self.scene.object.size_m), "position_m": list(self.scene.object.position_m)}
                        if self.scene.object is not None else None
                    ),
                    "target": {
                        "kind": self.scene.target.kind,
                        "color": self.scene.target.color,
                        "size_m": list(self.scene.target.size_m),
                        "position_m": list(self.scene.target.position_m),
                    },
                }
                if self.scene is not None
                else None
            ),
        }


def quaternion_up_z(rotation_wxyz: Sequence[float]) -> float:
    _, x, y, _ = _tuple(rotation_wxyz, 4, "root rotation")
    return 1.0 - 2.0 * (x * x + y * y)


#: PhysX prefixes the joint DOFs of a floating-base articulation with the six
#: DOFs of its free base in every generalized-force vector.
FLOATING_BASE_DOF_COUNT = 6


def body_gravity_columns(
    values_width: int, joint_count: int, body_indices: Sequence[int]
) -> tuple[int, list[int]]:
    """Columns of a PhysX generalized-force row that belong to the body joints.

    The vector holds one entry per DOF in the articulation's own order: the six
    free-base DOFs first when the root is not fixed, then the joints.  The width
    therefore says which of the two layouts is in front of us, and anything else
    is refused instead of indexed on an assumption -- a wrong offset would apply
    another joint's gravity torque with a perfectly plausible magnitude.
    """
    if values_width == joint_count:
        offset = 0
    elif values_width == joint_count + FLOATING_BASE_DOF_COUNT:
        offset = FLOATING_BASE_DOF_COUNT
    else:
        raise ContractError(
            f"generalized force width {values_width} matches neither {joint_count} joint DOFs "
            f"(fixed base) nor {joint_count + FLOATING_BASE_DOF_COUNT} (floating base)"
        )
    return offset, [offset + int(index) for index in body_indices]


#: The upper-limb joints an opt-in reset pose may name.  The pose never touches
#: the lower body or the waist: those channels are a single synthetic constant
#: across every demonstration, so changing them would change more than the one
#: variable such an experiment is about.
UPPER_LIMB_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
)


def load_reset_pose(path: Path) -> dict[str, float]:
    """Read an opt-in upper-limb reset pose from a JSON joint->radians file.

    Flat ``{joint_name: radians}`` and the experiment's grouped
    ``{"left_arm": [...], "right_arm": [...], ...}`` spelling are both accepted.
    Every named joint must be an upper-limb joint this contract knows; a pose
    that names a leg or waist joint is refused rather than silently applied.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ContractError(f"reset pose file does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ContractError(f"reset pose file {path} is not JSON: {error}") from None
    if not isinstance(raw, dict):
        raise ContractError(f"reset pose file {path} must hold a JSON object")

    groups = {
        "left_arm": ("left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                     "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
                     "left_wrist_pitch_joint", "left_wrist_yaw_joint"),
        "right_arm": ("right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                      "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
                      "right_wrist_pitch_joint", "right_wrist_yaw_joint"),
        "left_hand": ("left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
                      "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
                      "left_hand_middle_1_joint", "left_hand_index_0_joint",
                      "left_hand_index_1_joint"),
        "right_hand": ("right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
                       "right_hand_thumb_2_joint", "right_hand_index_0_joint",
                       "right_hand_index_1_joint", "right_hand_middle_0_joint",
                       "right_hand_middle_1_joint"),
    }
    pose: dict[str, float] = {}
    for key, value in raw.items():
        if key in groups:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ContractError(f"reset pose group {key!r} must be a list of radians")
            if len(value) != len(groups[key]):
                raise ContractError(
                    f"reset pose group {key!r} must hold exactly {len(groups[key])} values"
                )
            for name, angle in zip(groups[key], value):
                pose[name] = _angle(angle, name)
            continue
        if key not in UPPER_LIMB_JOINT_NAMES:
            raise ContractError(
                f"reset pose names {key!r}, which is not an upper-limb joint; the "
                "opt-in reset pose only ever moves the arms and the hands"
            )
        pose[key] = _angle(value, key)
    if not pose:
        raise ContractError(f"reset pose file {path} names no joints")
    return pose


def _angle(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"reset pose value for {name} must be a number, got {value!r}")
    angle = float(value)
    if not math.isfinite(angle):
        raise ContractError(f"reset pose value for {name} is not finite: {value!r}")
    return angle
