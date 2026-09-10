"""Dataset-calibrated CloudWalk Isaac scene construction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HEAD_CAMERA_NAME = "head_camera"


def head_camera_prim_path(config: dict[str, Any], env_namespace: str = "{ENV_REGEX_NS}") -> str:
    parent_link = config["camera"]["parent_link"]
    if not parent_link or "/" in parent_link:
        raise ValueError("camera parent_link must be a single robot link name")
    return f"{env_namespace}/Robot/{parent_link}/{HEAD_CAMERA_NAME}"


def load_scene_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported CloudWalk scene config schema")
    camera = config["camera"]
    if len(camera["resolution"]) != 2 or any(int(value) <= 0 for value in camera["resolution"]):
        raise ValueError("camera resolution must contain two positive dimensions")
    if len(camera["position_rel_parent_m"]) != 3 or len(camera["rotation_wxyz_parent"]) != 4:
        raise ValueError("camera mount pose must contain a 3D position and quaternion")
    quaternion_norm = sum(float(value) ** 2 for value in camera["rotation_wxyz_parent"])
    if abs(quaternion_norm - 1.0) > 1e-6:
        raise ValueError("camera mount quaternion must be normalized")
    head_camera_prim_path(config)
    robot = config["robot"]
    table = config["table"]
    if len(robot["position_m"]) != 3 or len(robot["rotation_wxyz"]) != 4:
        raise ValueError("robot pose must contain a 3D position and quaternion")
    robot_quaternion_norm = sum(float(value) ** 2 for value in robot["rotation_wxyz"])
    if abs(robot_quaternion_norm - 1.0) > 1e-6:
        raise ValueError("robot quaternion must be normalized")
    if not robot["require_asset_collisions"] or not table["collision_enabled"]:
        raise ValueError("CloudWalk robot and table collisions must be enabled")
    table_min_x = float(table["position_m"][0]) - float(table["size_m"][0]) / 2.0
    robot_max_x = float(robot["position_m"][0]) + float(robot["clearance_radius_m"])
    if robot_max_x > table_min_x:
        raise ValueError("robot must start outside the table collision footprint")
    if table_min_x - robot_max_x > 0.15:
        raise ValueError("robot must start close to the table")
    color_paths = (
        ("table", "wood_color_srgb"), ("bottle", "shell_color_srgb"), ("bottle", "water_color_srgb"),
        ("bottle", "cap_color_srgb"), ("room", "floor_color_srgb"), ("room", "wall_color_srgb"),
        ("lighting", "key", "color_srgb"), ("lighting", "fill", "color_srgb"),
        ("lighting", "ambient", "color_srgb"),
    )
    for path_parts in color_paths:
        value: Any = config
        for part in path_parts:
            value = value[part]
        if len(value) != 3 or any(not 0.0 <= float(channel) <= 1.0 for channel in value):
            raise ValueError(f"{'/'.join(path_parts)} must be an RGB triplet in [0, 1]")
    for name in ("shell_opacity", "water_opacity"):
        if not 0.0 < float(config["bottle"][name]) < 1.0:
            raise ValueError(f"bottle/{name} must be strictly between 0 and 1")
    if len(config["table"]["wood_texture_scale_rgba"]) != 4:
        raise ValueError("table/wood_texture_scale_rgba must contain four channels")
    if config["render"].get("realtime_translucency") is not True:
        raise ValueError("render/realtime_translucency must be enabled")
    return config


def configure_rtx(config: dict[str, Any]) -> None:
    import carb

    settings = carb.settings.get_settings()
    render = config["render"]
    settings.set("/rtx/rendermode", render["renderer"])
    settings.set_bool("/rtx/translucency/enabled", bool(render["realtime_translucency"]))
    settings.set_int("/rtx/pathtracing/spp", int(render["samples_per_pixel"]))
    settings.set_float("/rtx/post/tonemap/exposureBias", float(render["exposure"]))
    settings.set_bool("/rtx/post/tonemap/enabled", True)



def configure_g1_free_base_articulation() -> None:
    """Move the Inspire USD's articulation root from its disabled world joint to pelvis."""
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_joint_path = "/World/envs/env_0/Robot/root_joint"
    root_joint_prim = stage.GetPrimAtPath(root_joint_path)
    root_joint = UsdPhysics.Joint(root_joint_prim)
    robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    pelvis = stage.GetPrimAtPath("/World/envs/env_0/Robot/pelvis")
    if not root_joint or not robot or not pelvis:
        raise RuntimeError("CloudWalk free-base USD override cannot find Robot, root_joint, and pelvis")
    collision_apis = [
        UsdPhysics.CollisionAPI(prim)
        for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
        if UsdPhysics.CollisionAPI(prim)
    ]
    if not collision_apis or any(api.GetCollisionEnabledAttr().Get() is False for api in collision_apis):
        raise RuntimeError("CloudWalk robot asset must provide enabled collision prims")
    root_joint.GetJointEnabledAttr().Set(False)
    source_api = PhysxSchema.PhysxArticulationAPI(root_joint_prim)
    source_attributes = {
        name: root_joint_prim.GetAttribute(name).Get()
        for name in source_api.GetSchemaAttributeNames()
        if root_joint_prim.GetAttribute(name)
    }
    root_joint_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    root_joint_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    UsdPhysics.ArticulationRootAPI.Apply(pelvis)
    pelvis_api = PhysxSchema.PhysxArticulationAPI.Apply(pelvis)
    for name, value in source_attributes.items():
        pelvis.GetAttribute(name).Set(value)
    if not UsdPhysics.ArticulationRootAPI(pelvis) or PhysxSchema.PhysxArticulationAPI(root_joint_prim):
        raise RuntimeError("CloudWalk free-base USD override did not move the articulation root to pelvis")


def make_scene_cfg(config: dict[str, Any], robot_cfg: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    robot_data = config["robot"]
    table_data = config["table"]
    bottle_data = config["bottle"]
    camera_data = config["camera"]
    lighting_data = config["lighting"]
    table_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=table_data["static_friction"], dynamic_friction=table_data["dynamic_friction"], restitution=table_data["restitution"]
    )
    bottle_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=bottle_data["static_friction"], dynamic_friction=bottle_data["dynamic_friction"], restitution=bottle_data["restitution"]
    )

    @configclass
    class CloudWalkSceneCfg(InteractiveSceneCfg):
        floor = AssetBaseCfg(
            prim_path="/World/Floor",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.025)),
            spawn=sim_utils.CuboidCfg(
                size=(6.0, 6.0, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(config["room"]["floor_color_srgb"]), roughness=config["room"]["floor_roughness"]
                ),
            ),
        )
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(table_data["position_m"])),
            spawn=sim_utils.CuboidCfg(
                size=tuple(table_data["size_m"]),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=table_data["collision_enabled"]),
                physics_material=table_material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(table_data["wood_color_srgb"]), roughness=table_data["roughness"], metallic=0.0),
            ),
        )
        bottle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bottle",
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(bottle_data["position_m"])),
            spawn=sim_utils.CylinderCfg(
                radius=bottle_data["collision_radius_m"], height=bottle_data["height_m"], axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=bottle_data["mass_kg"]), physics_material=bottle_material,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(bottle_data["shell_color_srgb"]), opacity=bottle_data["shell_opacity"],
                    roughness=bottle_data["shell_roughness"],
                ),
            ),
        )
        key = AssetBaseCfg(
            prim_path="/World/KeyLight",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tuple(lighting_data["key"]["position_m"]), rot=tuple(lighting_data["key"]["rotation_wxyz"])
            ),
            spawn=sim_utils.DiskLightCfg(
                intensity=lighting_data["key"]["intensity"], color=tuple(lighting_data["key"]["color_srgb"]),
                radius=lighting_data["key"]["radius_m"], enable_color_temperature=True,
                color_temperature=lighting_data["key"]["color_temperature_k"],
            ),
        )
        fill = AssetBaseCfg(
            prim_path="/World/FillLight", init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(lighting_data["fill"]["position_m"])),
            spawn=sim_utils.SphereLightCfg(
                intensity=lighting_data["fill"]["intensity"], color=tuple(lighting_data["fill"]["color_srgb"]),
                radius=lighting_data["fill"]["radius_m"],
            ),
        )
        ambient = AssetBaseCfg(
            prim_path="/World/Ambient",
            spawn=sim_utils.DomeLightCfg(
                intensity=lighting_data["ambient"]["intensity"], color=tuple(lighting_data["ambient"]["color_srgb"])
            ),
        )
        robot: ArticulationCfg = robot_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=robot_cfg.init_state.replace(
                pos=tuple(robot_data["position_m"]), rot=tuple(robot_data["rotation_wxyz"]),
            ),
        )
        head_camera = CameraCfg(
            prim_path=head_camera_prim_path(config), update_period=0.0,
            height=camera_data["resolution"][1], width=camera_data["resolution"][0], data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=camera_data["focal_length_mm"], horizontal_aperture=camera_data["horizontal_aperture_mm"], clipping_range=(0.05, 20.0)
            ),
            offset=CameraCfg.OffsetCfg(pos=tuple(camera_data["position_rel_parent_m"]), rot=tuple(camera_data["rotation_wxyz_parent"]), convention="world"),
        )

    return CloudWalkSceneCfg


def _preview_material(
    stage: Any, path: str, color: tuple[float, float, float], roughness: float,
    opacity: float = 1.0, ior: float = 1.5, texture: Path | None = None,
    texture_scale: tuple[float, float, float, float] | None = None,
) -> Any:
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse.Set(color)
    if texture is not None:
        reader = UsdShade.Shader.Define(stage, f"{path}/Primvar")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        sampler = UsdShade.Shader.Define(stage, f"{path}/Texture")
        sampler.CreateIdAttr("UsdUVTexture")
        sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        sampler.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        if texture_scale is not None:
            sampler.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(texture_scale)
        diffuse.ConnectToSource(sampler.ConnectableAPI(), "rgb")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _cylinder(stage: Any, path: str, radius: float, height: float, z: float, material: Any) -> None:
    from pxr import Gf, UsdGeom, UsdShade

    prim = UsdGeom.Cylinder.Define(stage, path)
    prim.CreateAxisAttr("Z"); prim.CreateRadiusAttr(radius); prim.CreateHeightAttr(height)
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)


def decorate_scene(config: dict[str, Any]) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    room = config["room"]
    wall_mat = _preview_material(
        stage, "/World/Looks/NeutralWall", tuple(room["wall_color_srgb"]), room["wall_roughness"]
    )
    table_data = config["table"]
    wood_texture = Path(__file__).resolve().parents[1] / "configs" / "cloudwalk_wood.png"
    wood = _preview_material(
        stage, "/World/Looks/WarmGlossWood", tuple(table_data["wood_color_srgb"]),
        table_data["roughness"], texture=wood_texture, texture_scale=tuple(table_data["wood_texture_scale_rgba"]),
    )
    bottle_data = config["bottle"]
    glass = _preview_material(
        stage, "/World/Looks/BottleGlass", tuple(bottle_data["shell_color_srgb"]), bottle_data["shell_roughness"],
        bottle_data["shell_opacity"], bottle_data["shell_ior"],
    )
    water = _preview_material(
        stage, "/World/Looks/BlueWater", tuple(bottle_data["water_color_srgb"]), bottle_data["water_roughness"],
        bottle_data["water_opacity"], bottle_data["water_ior"],
    )
    black = _preview_material(
        stage, "/World/Looks/CapPlastic", tuple(bottle_data["cap_color_srgb"]), bottle_data["cap_roughness"]
    )
    metal = _preview_material(stage, "/World/Looks/TableLegMetal", (0.055, 0.065, 0.07), 0.22)

    def cube(path: str, scale: tuple[float, float, float], pos: tuple[float, float, float], material: Any) -> None:
        prim = UsdGeom.Cube.Define(stage, path); prim.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(prim); xform.AddTranslateOp().Set(Gf.Vec3d(*pos)); xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)

    cube("/World/Room/BackWall", (0.08, 3.0, 1.5), (2.45, 0.0, 1.5), wall_mat)
    cube("/World/Room/LeftWall", (2.5, 0.08, 1.5), (0.0, 2.4, 1.5), wall_mat)
    cube("/World/Room/Baseboard", (0.035, 2.4, 0.07), (2.34, 0.0, 0.07), metal)

    top_z = config["table"]["size_m"][2] / 2.0
    for index, y in enumerate((-0.285, -0.1425, 0.0, 0.1425, 0.285)):
        cube(f"/World/envs/env_0/Table/Visual/Plank{index}", (0.895, 0.137, 0.018), (0.0, y, top_z + 0.018), wood)
    for index, (x, y) in enumerate(((0.36, 0.27), (0.36, -0.27), (-0.36, 0.27), (-0.36, -0.27))):
        cube(f"/World/envs/env_0/Table/Visual/Leg{index}", (0.045, 0.045, 0.69), (x, y, -0.03), metal)

    h = bottle_data["height_m"]
    base = -h / 2.0
    _cylinder(stage, "/World/envs/env_0/Bottle/Visual/Water", bottle_data["body_radius_m"] * 0.82, h * 0.66, base + h * 0.36, water)
    for index, (radius, height, z) in enumerate(((0.048, 0.145, base + 0.075), (0.041, 0.028, base + 0.158), (0.031, 0.022, base + 0.181))):
        _cylinder(stage, f"/World/envs/env_0/Bottle/Visual/Shell{index}", radius, height, z, glass)
    for index in range(6):
        _cylinder(stage, f"/World/envs/env_0/Bottle/Visual/Rib{index}", 0.0495, 0.004, base + 0.038 + index * 0.018, glass)
    _cylinder(stage, "/World/envs/env_0/Bottle/Visual/Cap", bottle_data["cap_radius_m"], bottle_data["cap_height_m"], base + h + bottle_data["cap_height_m"] / 2.0, black)
    for index in range(7):
        _cylinder(stage, f"/World/envs/env_0/Bottle/Visual/CapRib{index}", bottle_data["cap_radius_m"] + 0.0012, 0.0015, base + h - 0.009 + index * 0.003, black)
