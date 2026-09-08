"""Dataset-calibrated CloudWalk Isaac scene construction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_scene_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported CloudWalk scene config schema")
    return config


def configure_rtx(config: dict[str, Any]) -> None:
    import carb

    settings = carb.settings.get_settings()
    render = config["render"]
    settings.set("/rtx/rendermode", render["renderer"])
    settings.set_int("/rtx/pathtracing/spp", int(render["samples_per_pixel"]))
    settings.set_float("/rtx/post/tonemap/exposureBias", float(render["exposure"]))
    settings.set_bool("/rtx/post/tonemap/enabled", True)


def discover_nucleus_assets() -> dict[str, Any]:
    try:
        import omni.client
    except ImportError:
        return {"available": False, "reason": "omni.client extension unavailable"}
    roots = (
        "omniverse://localhost/NVIDIA/Assets/Isaac/5.1",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
    )
    discovered = []
    for root in roots:
        result, entries = omni.client.list(root)
        if result == omni.client.Result.OK:
            discovered.append({"root": root, "entries": [entry.relative_path for entry in entries[:12]]})
    return {"available": bool(discovered), "roots": discovered}


def configure_g1_free_base_articulation() -> None:
    """Move the Inspire USD's articulation root from its disabled world joint to pelvis."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_joint_path = "/World/envs/env_0/Robot/root_joint"
    root_joint_prim = stage.GetPrimAtPath(root_joint_path)
    root_joint = UsdPhysics.Joint(root_joint_prim)
    pelvis = stage.GetPrimAtPath("/World/envs/env_0/Robot/pelvis")
    if not root_joint or not pelvis:
        raise RuntimeError("CloudWalk free-base USD override cannot find root_joint and pelvis")
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

    table_data = config["table"]
    bottle_data = config["bottle"]
    camera_data = config["camera"]
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
                size=(6.0, 6.0, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(config["room"]["floor_color_srgb"]), roughness=0.58),
            ),
        )
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(table_data["position_m"])),
            spawn=sim_utils.CuboidCfg(
                size=tuple(table_data["size_m"]), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=table_material,
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
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.09, 0.12), opacity=0.02, roughness=0.18),
            ),
        )
        key = AssetBaseCfg(
            prim_path="/World/KeyLight", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, -1.3, 2.8), rot=(0.9239, 0.3827, 0.0, 0.0)),
            spawn=sim_utils.DiskLightCfg(intensity=1250.0, color=(1.0, 0.86, 0.72), radius=1.25, enable_color_temperature=True, color_temperature=4300.0),
        )
        fill = AssetBaseCfg(
            prim_path="/World/FillLight", init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, 1.5, 2.2)),
            spawn=sim_utils.SphereLightCfg(intensity=650.0, color=(0.72, 0.84, 1.0), radius=0.7),
        )
        ambient = AssetBaseCfg(prim_path="/World/Ambient", spawn=sim_utils.DomeLightCfg(intensity=420.0, color=(0.68, 0.72, 0.74)))
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis/head_camera", update_period=0.0,
            height=camera_data["resolution"][1], width=camera_data["resolution"][0], data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=camera_data["focal_length_mm"], horizontal_aperture=camera_data["horizontal_aperture_mm"], clipping_range=(0.05, 20.0)
            ),
            offset=CameraCfg.OffsetCfg(pos=tuple(camera_data["position_rel_pelvis_m"]), rot=tuple(camera_data["rotation_wxyz_ros"]), convention="ros"),
        )

    return CloudWalkSceneCfg


def _preview_material(
    stage: Any, path: str, color: tuple[float, float, float], roughness: float,
    opacity: float = 1.0, ior: float = 1.5, texture: Path | None = None,
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
    wall_mat = _preview_material(stage, "/World/Looks/NeutralWall", tuple(room["wall_color_srgb"]), 0.72)
    wood_texture = Path(__file__).resolve().parents[1] / "configs" / "cloudwalk_wood.png"
    wood = _preview_material(
        stage, "/World/Looks/WarmGlossWood", tuple(config["table"]["wood_color_srgb"]),
        config["table"]["roughness"], texture=wood_texture,
    )
    glass = _preview_material(stage, "/World/Looks/BottleGlass", (0.10, 0.34, 0.48), 0.08, 0.36, 1.49)
    water = _preview_material(stage, "/World/Looks/BlueWater", tuple(config["bottle"]["water_color_srgb"]), 0.05, 0.48, 1.333)
    black = _preview_material(stage, "/World/Looks/CapPlastic", (0.006, 0.008, 0.009), 0.18)
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

    bottle_data = config["bottle"]
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
