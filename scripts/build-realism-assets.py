"""Build the static assets of the realistic PickApple scene.

Writes, under ``<data root>/models/realism/pickapple-v1/``:

* ``desk.usda``        white office desk (static colliders), origin on the floor
* ``plate.usda``       revolved ceramic plate (static triangle-mesh collider),
                       origin at the centre of its bounding box
* ``apple.usda``       Poly Haven ``food_apple_01`` (CC0) scan, scaled to the
                       dataset apple, convex-hull collider, rigid body; origin
                       at the centre of its bounding box
* ``environment.usda`` office floor projected from a dataset head-camera frame
                       plus the lights that replace the default dome/key rig

The floor is "projection mapped": every vertex is a head-camera ray hitting
the floor and its texture coordinate is the pixel the ray came from, so from
the (fixed-base) head camera the floor reproduces the dataset background with
the correct perspective.  Pixels that belong to the real table and the robot
are inpainted first; the simulated table and robot draw over them anyway.

Needs ``usd-core numpy opencv-python-headless trimesh`` (host side, no Isaac):

    uv venv /tmp/usdenv && uv pip install -p /tmp/usdenv/bin/python usd-core numpy opencv-python-headless trimesh
    /tmp/usdenv/bin/python scripts/build-realism-assets.py

The scene constants below are the values the profile
``configs/profiles/pick-apple-askida-real.json`` declares; change both together.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from humanoid_lab.simulators.isaac.lens import LensModel  # noqa: E402

DATA_ROOT = REPO / "data"
CONTAINER_DATA_ROOT = Path("/data")
OUT_REL = Path("models/realism/pickapple-v1")

# --- scene geometry (world frame, metres) -----------------------------------
SURFACE_Z = 0.8357609                      # declared table surface height
DESK_X = (0.18, 0.80)                      # near / far edge
DESK_Y = (-0.57, 0.51)                     # right / left edge
DESK_TOP_THICKNESS = 0.025
PLATE_DIAMETER = 0.19
PLATE_HEIGHT = 0.02
APPLE_WIDTH = 0.062                        # dataset apple, measured in the head camera
APPLE_SATURATION, APPLE_VALUE = 0.6, 1.5

# --- head camera (must equal the profile's camera block) ---------------------
ROBOT_ROOT = np.array([0.08, 0.0, 0.792563])       # pick-apple-askida initial_position_m
TORSO_IN_PELVIS = np.array([-0.0039635, 0.0, 0.044])  # asset torso_link origin at zero waist
CAMERA_IN_TORSO = np.array([0.0576235, 0.01753, 0.41987])
CAMERA_PITCH_DEG = 55.0
FOCAL_MM, APERTURE_MM = 8.79, 20.955
K1, K2, RENDER_SCALE = 0.15, 0.0, 1.5
WIDTH, HEIGHT = 640, 480

BACKDROP_FRAME = REPO / "realizm/apple-unifolm/episode_000052_t2.87s.png"


def camera_pose() -> tuple[np.ndarray, np.ndarray]:
    """World position and rotation (columns: forward, left, up) of the head camera."""
    p = math.radians(CAMERA_PITCH_DEG)
    rotation = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    return ROBOT_ROOT + TORSO_IN_PELVIS + CAMERA_IN_TORSO, rotation


def lens() -> LensModel:
    return LensModel.build(WIDTH, HEIGHT, round(WIDTH * RENDER_SCALE), round(HEIGHT * RENDER_SCALE),
                           FOCAL_MM, APERTURE_MM, K1, K2)


def project(points: np.ndarray) -> np.ndarray:
    position, rotation = camera_pose()
    model = lens()
    local = (points - position) @ rotation
    return np.array([model.project(-y / x, -z / x) if x > 1e-3 else (np.nan, np.nan) for x, y, z in local])


def pixel_ray(u: float, v: float) -> np.ndarray:
    model = lens()
    xd = (u - WIDTH / 2.0) / model.output_focal_px
    yd = (v - HEIGHT / 2.0) / model.output_focal_px
    r2 = xd * xd + yd * yd
    scale = 1.0 + K1 * r2 + K2 * r2 * r2
    _, rotation = camera_pose()
    return rotation @ np.array([1.0, -xd * scale, -yd * scale])


# --- small USD helpers -------------------------------------------------------
def new_stage(path: Path, root: str) -> tuple[Usd.Stage, Usd.Prim]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, f"/{root}").GetPrim()
    stage.SetDefaultPrim(prim)
    return stage, prim


def preview_material(stage: Usd.Stage, path: str, diffuse=(0.8, 0.8, 0.8), roughness=0.5, specular=0.5,
                     metallic=0.0, clearcoat=0.0, diffuse_texture: str | None = None,
                     emissive_texture: str | None = None, emissive_scale: float = 1.0) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.1)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.04 * specular / 0.5))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    textures = [("diffuseColor", diffuse_texture, 1.0), ("emissiveColor", emissive_texture, emissive_scale)]
    if diffuse_texture or emissive_texture:
        reader = UsdShade.Shader.Define(stage, f"{path}/StReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    for name, texture, scale in textures:
        if texture is None:
            continue
        tex = UsdShade.Shader.Define(stage, f"{path}/{name}Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(scale, scale, scale, 1.0))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    if diffuse_texture is None:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    return material


def bind(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def box(stage: Usd.Stage, path: str, lo, hi, material, collider: bool) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*((lo + hi) / 2.0)))
    xform.SetScale(Gf.Vec3f(*(hi - lo)))
    bind(cube.GetPrim(), material)
    if collider:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


# --- assets --------------------------------------------------------------------
def build_desk(out: Path) -> dict:
    """Desk in world coordinates with its origin at the world origin (profile position 0)."""
    stage, root = new_stage(out / "desk.usda", "OfficeDesk")
    laminate = preview_material(stage, "/OfficeDesk/Looks/WhiteLaminate", diffuse=(0.78, 0.79, 0.78),
                                roughness=0.55, specular=0.35)
    frame = preview_material(stage, "/OfficeDesk/Looks/GreyFrame", diffuse=(0.55, 0.56, 0.57),
                             roughness=0.4, metallic=0.6)
    top_lo = (DESK_X[0], DESK_Y[0], SURFACE_Z - DESK_TOP_THICKNESS)
    top_hi = (DESK_X[1], DESK_Y[1], SURFACE_Z)
    box(stage, "/OfficeDesk/Top", top_lo, top_hi, laminate, collider=True)
    leg, inset = 0.04, 0.05
    leg_top = SURFACE_Z - DESK_TOP_THICKNESS
    for index, (x, y) in enumerate([(DESK_X[0] + inset, DESK_Y[0] + inset), (DESK_X[0] + inset, DESK_Y[1] - inset),
                                    (DESK_X[1] - inset, DESK_Y[0] + inset), (DESK_X[1] - inset, DESK_Y[1] - inset)]):
        box(stage, f"/OfficeDesk/Leg{index}", (x - leg / 2, y - leg / 2, 0.0), (x + leg / 2, y + leg / 2, leg_top),
            frame, collider=False)
    # Side rails under the top, as on the real desk frame.
    for side, y in (("Left", DESK_Y[1] - inset), ("Right", DESK_Y[0] + inset)):
        box(stage, f"/OfficeDesk/Rail{side}", (DESK_X[0] + inset, y - 0.015, leg_top - 0.06),
            (DESK_X[1] - inset, y + 0.015, leg_top), frame, collider=False)
    stage.GetRootLayer().Save()
    return {"near_edge_x": DESK_X[0], "far_edge_x": DESK_X[1], "y": DESK_Y, "surface_z": SURFACE_Z}


def revolve(profile: list[tuple[float, float]], segments: int = 96):
    """Mesh of a closed (r, z) profile revolved about +z."""
    points, counts, indices = [], [], []
    n = len(profile)
    for s in range(segments):
        a = 2 * math.pi * s / segments
        for r, z in profile:
            points.append((r * math.cos(a), r * math.sin(a), z))
    for s in range(segments):
        t = (s + 1) % segments
        for i in range(n - 1):
            a0, a1 = s * n + i, s * n + i + 1
            b0, b1 = t * n + i, t * n + i + 1
            counts.append(4)
            indices.extend([a0, b0, b1, a1])
    return points, counts, indices


def build_plate(out: Path) -> dict:
    """Glazed pastel-pink plate: flat well, rising rim with a rounded lip, foot ring."""
    stage, root = new_stage(out / "plate.usda", "Plate")
    R, H = PLATE_DIAMETER / 2.0, PLATE_HEIGHT
    # Closed profile, top surface from the centre outwards then the underside back.
    top = [(0.0, 0.006), (0.060, 0.006), (0.070, 0.0065), (0.078, 0.0085), (0.085, 0.0125),
           (0.090, 0.0165), (R - 0.004, H - 0.0005), (R - 0.0015, H), (R, H - 0.002)]
    bottom = [(R - 0.0005, H - 0.005), (0.086, 0.0085), (0.075, 0.0045), (0.062, 0.0025),
              (0.060, 0.0), (0.054, 0.0), (0.052, 0.002), (0.0, 0.002)]
    profile = [(r, z - H / 2.0) for r, z in top + bottom]
    points, counts, indices = revolve(profile)
    mesh = UsdGeom.Mesh.Define(stage, "/Plate/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    # Faces wind so that the top surface faces +z when revolved counter-clockwise.
    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    mesh.CreateExtentAttr([Gf.Vec3f(-R, -R, -H / 2), Gf.Vec3f(R, R, H / 2)])
    glaze = preview_material(stage, "/Plate/Looks/PinkGlaze", diffuse=(0.55, 0.22, 0.18), roughness=0.3,
                             specular=0.5, clearcoat=0.25)
    bind(mesh.GetPrim(), glaze)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    stage.GetRootLayer().Save()
    return {"size_m": [PLATE_DIAMETER, PLATE_DIAMETER, PLATE_HEIGHT]}


def fetch_apple(source_dir: Path) -> None:
    """Download Poly Haven food_apple_01 (CC0) at 2k once into the data root."""
    import urllib.request

    if (source_dir / "food_apple_01_2k.usdc").exists():
        return
    (source_dir / "textures").mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen("https://api.polyhaven.com/files/food_apple_01", timeout=30) as response:
        files = json.load(response)
    (source_dir / "api.json").write_text(json.dumps(files))
    entry = files["usd"]["2k"]["usd"]
    for relative, item in [("food_apple_01_2k.usdc", entry), *entry["include"].items()]:
        urllib.request.urlretrieve(item["url"], source_dir / relative)


def build_apple(out: Path, source_dir: Path) -> dict:
    import trimesh

    fetch_apple(source_dir)

    source = Usd.Stage.Open(str(source_dir / "food_apple_01_2k.usdc"))
    mesh_prim = next(p for p in source.Traverse() if p.IsA(UsdGeom.Mesh))
    src = UsdGeom.Mesh(mesh_prim)
    xform = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim))
    points = np.array(src.GetPointsAttr().Get(), dtype=np.float64)
    points = (np.c_[points, np.ones(len(points))] @ xform)[:, :3]
    lo, hi = points.min(0), points.max(0)
    scale = APPLE_WIDTH / max(hi[0] - lo[0], hi[1] - lo[1])
    points = (points - (lo + hi) / 2.0) * scale
    size = (hi - lo) * scale
    counts = np.array(src.GetFaceVertexCountsAttr().Get())
    indices = np.array(src.GetFaceVertexIndicesAttr().Get())
    st = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st")

    triangles = []
    offset = 0
    for count in counts:
        face = indices[offset:offset + count]
        triangles.extend([[face[0], face[k], face[k + 1]] for k in range(1, count - 1)])
        offset += count
    body = trimesh.Trimesh(points, np.array(triangles), process=True)
    volume_m3 = float(abs(body.volume)) if body.is_watertight else float(body.convex_hull.volume)
    hull_volume = float(body.convex_hull.volume)

    textures = out / "textures"
    textures.mkdir(parents=True, exist_ok=True)
    # The dataset apple reads as a pale red-yellow apple under the office light;
    # the scan is a deep red one, so its albedo is desaturated and lifted.
    skin_image = cv2.imread(str(source_dir / "textures/food_apple_01_diff_2k.jpg"))
    hsv = cv2.cvtColor(skin_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= APPLE_SATURATION
    hsv[..., 2] = np.clip(hsv[..., 2] * APPLE_VALUE, 0, 255)
    cv2.imwrite(str(textures / "apple_diffuse.jpg"), cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    stage, root = new_stage(out / "apple.usda", "Apple")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root)
    mesh = UsdGeom.Mesh.Define(stage, "/Apple/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts.tolist()))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices.tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(*(-size / 2)), Gf.Vec3f(*(size / 2))])
    normals = src.GetNormalsAttr()
    if normals.HasValue():
        mesh.CreateNormalsAttr(normals.Get())
        mesh.SetNormalsInterpolation(src.GetNormalsInterpolation())
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, st.GetInterpolation())
    primvar.Set(st.Get())
    if st.GetIndicesAttr().HasValue():
        primvar.SetIndices(st.GetIndices())
    skin = preview_material(stage, "/Apple/Looks/Skin", roughness=0.35, specular=0.5, clearcoat=0.3,
                            diffuse_texture="./textures/apple_diffuse.jpg")
    bind(mesh.GetPrim(), skin)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    stage.GetRootLayer().Save()
    return {"size_m": [round(float(v), 4) for v in size], "mesh_volume_cm3": round(volume_m3 * 1e6, 1),
            "hull_volume_cm3": round(hull_volume * 1e6, 1), "watertight": bool(body.is_watertight),
            "scale": scale}


def backdrop_texture(out: Path) -> Path:
    """Dataset frame with the real table and robot inpainted to carpet."""
    image = cv2.imread(str(BACKDROP_FRAME))
    mask = np.zeros(image.shape[:2], np.uint8)
    edge = np.linspace(0.0, 1.0, 200)[:, None]
    corners = [np.array(c) for c in ((DESK_X[0], DESK_Y[0], SURFACE_Z), (DESK_X[1], DESK_Y[0], SURFACE_Z),
                                     (DESK_X[1], DESK_Y[1], SURFACE_Z), (DESK_X[0], DESK_Y[1], SURFACE_Z))]
    outline = np.concatenate([a + edge * (b - a) for a, b in zip(corners, corners[1:] + corners[:1])])
    polygon = project(outline)
    polygon = polygon[np.isfinite(polygon).all(1)].astype(np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    # The real table is slightly larger than its projected outline in places
    # and the robot fills the bottom of the frame below the table edge.
    mask = cv2.dilate(mask, np.ones((25, 25), np.uint8))
    mask[int(HEIGHT * 0.80):, :] = 255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = ((hsv[..., 2] > 150) & (hsv[..., 1] < 60)).astype(np.uint8) * 255
    bright[:int(HEIGHT * 0.3)] = 0  # keep far-field furniture
    bright[:, int(WIDTH * 0.88):] = 0  # keep the neighbouring desk on the right
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_or(mask, cv2.dilate(bright, np.ones((9, 9), np.uint8)))
    filled = cv2.inpaint(image, mask, 21, cv2.INPAINT_TELEA)
    path = out / "textures/backdrop.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), filled)
    cv2.imwrite(str(out / "textures/backdrop_mask.png"), mask)
    return path


def build_environment(out: Path) -> dict:
    texture = backdrop_texture(out)
    carpet = cv2.imread(str(texture))[: int(HEIGHT * 0.25)].reshape(-1, 3).mean(0)[::-1] / 255.0
    stage, root = new_stage(out / "environment.usda", "Environment")

    # Projected floor: a grid over the (slightly overscanned) image plane.
    nu, nv, overscan = 97, 73, 0.08
    us = np.linspace(-overscan, 1 + overscan, nu) * WIDTH
    vs = np.linspace(-overscan, 1 + overscan, nv) * HEIGHT
    position, _ = camera_pose()
    floor_z = 0.001
    points, uvs = [], []
    for v in vs:
        for u in us:
            ray = pixel_ray(u, v)
            t = (floor_z - position[2]) / ray[2] if ray[2] < -1e-3 else 1e9
            hit = position + ray * min(t, 25.0 / max(np.linalg.norm(ray[:2]), 1e-6))
            hit[2] = floor_z
            points.append(hit)
            uvs.append((np.clip(u / WIDTH, 0.0, 1.0), 1.0 - np.clip(v / HEIGHT, 0.0, 1.0)))
    counts, indices = [], []
    for j in range(nv - 1):
        for i in range(nu - 1):
            a = j * nu + i
            counts.append(4)
            indices.extend([a, a + nu, a + nu + 1, a + 1])
    mesh = UsdGeom.Mesh.Define(stage, "/Environment/ProjectedFloor")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                      UsdGeom.Tokens.vertex)
    primvar.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uvs]))
    backdrop = preview_material(stage, "/Environment/Looks/Backdrop", diffuse=(0, 0, 0), roughness=1.0,
                                specular=0.0, emissive_texture="./textures/backdrop.png",
                                emissive_scale=EMISSIVE_SCALE)
    # Emission only: the texture already contains the real lighting.
    backdrop_shader = UsdShade.Shader(stage.GetPrimAtPath("/Environment/Looks/Backdrop/Surface"))
    backdrop_shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.0, 0.0, 0.0))
    bind(mesh.GetPrim(), backdrop)

    # Plain carpet everywhere else (below the projected patch).
    base = preview_material(stage, "/Environment/Looks/Carpet", diffuse=tuple(float(c) ** 2.2 for c in carpet),
                            roughness=0.95, specular=0.1)
    floor = UsdGeom.Mesh.Define(stage, "/Environment/BaseFloor")
    floor.CreatePointsAttr([(-30, -30, 0), (30, -30, 0), (30, 30, 0), (-30, 30, 0)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    bind(floor.GetPrim(), base)

    # Office lighting: a soft sky-like fill and a large ceiling panel over the desk.
    dome = UsdLux.DomeLight.Define(stage, "/Environment/Lights/Fill")
    dome.CreateIntensityAttr(DOME_INTENSITY)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
    panel = UsdLux.RectLight.Define(stage, "/Environment/Lights/CeilingPanel")
    panel.CreateWidthAttr(1.2)
    panel.CreateHeightAttr(1.2)
    panel.CreateIntensityAttr(PANEL_INTENSITY)
    panel.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    UsdGeom.XformCommonAPI(panel).SetTranslate(Gf.Vec3d(0.9, 0.3, 2.8))  # rect lights emit along -z
    stage.GetRootLayer().Save()
    return {"floor_vertices": len(points), "carpet_rgb": carpet.round(3).tolist()}


DOME_INTENSITY = 190.0
PANEL_INTENSITY = 2400.0
EMISSIVE_SCALE = 1.0


def main() -> int:
    global DOME_INTENSITY, PANEL_INTENSITY, EMISSIVE_SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--dome", type=float, default=DOME_INTENSITY)
    parser.add_argument("--panel", type=float, default=PANEL_INTENSITY)
    parser.add_argument("--emissive", type=float, default=EMISSIVE_SCALE)
    args = parser.parse_args()
    DOME_INTENSITY, PANEL_INTENSITY, EMISSIVE_SCALE = args.dome, args.panel, args.emissive
    out = DATA_ROOT / OUT_REL
    report = {
        "container_dir": str(CONTAINER_DATA_ROOT / OUT_REL),
        "desk": build_desk(out),
        "plate": build_plate(out),
        "apple": build_apple(out, DATA_ROOT / "models/polyhaven/food_apple_01"),
        "environment": build_environment(out),
        "lights": {"dome": DOME_INTENSITY, "panel": PANEL_INTENSITY, "emissive_scale": EMISSIVE_SCALE},
    }
    (out / "build-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
