"""Measure a declared elbow-flexion reach pose on the live Isaac stage.

The BlockStacking table height is not a free parameter: it is fixed by the
geometry of the G1/Dex3 asset at the pose the robot is expected to reach with.
This tool answers that question with a reading instead of an estimate.  It
loads the profile's robot, applies the declared start-up pose with the elbow
joints set to the angle that bends the arm to ninety degrees, and reports the
lowest world-z of each Dex3 hand -- palm, fingers and thumb included -- from
the live stage bounding boxes, next to the table and cube heights the profile
declares.

The ninety-degree elbow angle is solved, not tuned: the asset's own joint
frames give the upper-arm vector (the elbow joint origin in its parent frame)
and the forearm vector (the wrist-roll joint origin in the elbow frame), and
the hinge is the elbow's revolute axis.  The pose is a static geometry probe --
gravity is disabled so the measured pose is exactly the declared one -- and the
probe never drives the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import RunProfile

#: Links of one Dex3 hand.  The asset keeps every link as a sibling of the
#: root, so the hand is not one subtree to bound: the palm and the three finger
#: chains have to be bounded together, or the fingertips are left out.
HAND_LINK_NAMES: dict[str, tuple[str, ...]] = {
    "left": (
        "left_hand_palm_link",
        "left_hand_index_0_link",
        "left_hand_index_1_link",
        "left_hand_middle_0_link",
        "left_hand_middle_1_link",
        "left_hand_thumb_0_link",
        "left_hand_thumb_1_link",
        "left_hand_thumb_2_link",
    ),
    "right": (
        "right_hand_palm_link",
        "right_hand_index_0_link",
        "right_hand_index_1_link",
        "right_hand_middle_0_link",
        "right_hand_middle_1_link",
        "right_hand_thumb_0_link",
        "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
    ),
}

#: The two arm links whose world positions define the elbow bend.
UPPER_ARM_LINK = "{side}_shoulder_yaw_link"
ELBOW_LINK = "{side}_elbow_link"
FOREARM_LINK = "{side}_wrist_roll_link"

#: How far the measured bend angle may sit from the requested one.  The solve
#: is analytic, so a larger gap means the applied pose is not the requested one.
BEND_ANGLE_TOLERANCE_DEG = 1.0


def hand_link_names(side: str) -> tuple[str, ...]:
    """Links whose union is one Dex3 hand, fingers included."""
    try:
        return HAND_LINK_NAMES[side]
    except KeyError as error:
        raise ValueError(f"unknown hand side {side!r}") from error


def perpendicular_elbow_angle(
    upper_arm_offset: Sequence[float], forearm_offset: Sequence[float]
) -> float:
    """Elbow joint angle (rad) that puts the forearm perpendicular to the upper arm.

    ``upper_arm_offset`` is the elbow joint's origin in its parent frame, i.e.
    the upper-arm vector; ``forearm_offset`` is the wrist-roll joint's origin in
    the elbow frame, i.e. the forearm vector.  The elbow hinge turns about the
    frame's Y axis, so the forearm direction at joint value ``q`` is the offset
    rotated by ``q`` about Y, and the condition ``u . R_y(q) f == 0`` is solved
    for the solution closest to zero -- the one that bends the hand forward
    instead of backwards.
    """
    if len(upper_arm_offset) != 3 or len(forearm_offset) != 3:
        raise ValueError("joint offsets must be three-component vectors")
    ux, uy, uz = (float(value) for value in upper_arm_offset)
    fx, fy, fz = (float(value) for value in forearm_offset)
    # u . R_y(q) f = uy*fy + (ux*fx + uz*fz) * cos q + (ux*fz - uz*fx) * sin q
    constant = uy * fy
    cosine = ux * fx + uz * fz
    sine = ux * fz - uz * fx
    radius = math.hypot(cosine, sine)
    if radius < 1e-9:
        raise ValueError("elbow hinge axis is parallel to an arm segment; no bend angle exists")
    if abs(constant) > radius + 1e-9:
        raise ValueError("no elbow angle makes the arm segments perpendicular")
    phase = math.atan2(sine, cosine)
    offset = math.acos(max(-1.0, min(1.0, -constant / radius)))
    candidates = (phase + offset, phase - offset)
    return min(candidates, key=lambda angle: abs(math.atan2(math.sin(angle), math.cos(angle))))


def with_elbow_flexion(
    pose: Mapping[str, float], elbow_angle_rad: float, sides: Sequence[str] = ("left", "right")
) -> dict[str, float]:
    """The declared pose with the elbow joints set to one explicit angle."""
    flexion = dict(pose)
    for side in sides:
        joint = f"{side}_elbow_joint"
        if joint not in flexion:
            raise ValueError(f"pose does not name {joint}")
        flexion[joint] = float(elbow_angle_rad)
    return flexion


def arm_bend_angle_deg(shoulder: Sequence[float], elbow: Sequence[float], wrist: Sequence[float]) -> float:
    """Angle between the upper-arm and forearm segments, from world positions."""
    upper = [float(b) - float(a) for a, b in zip(shoulder, elbow)]
    forearm = [float(b) - float(a) for a, b in zip(elbow, wrist)]
    upper_norm = math.sqrt(sum(value * value for value in upper))
    forearm_norm = math.sqrt(sum(value * value for value in forearm))
    if upper_norm < 1e-9 or forearm_norm < 1e-9:
        raise ValueError("arm segments have zero length; positions are degenerate")
    cosine = sum(a * b for a, b in zip(upper, forearm)) / (upper_norm * forearm_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def lowest_link(link_bounds: Mapping[str, Sequence[Sequence[float]]]) -> tuple[str, float]:
    """Link with the lowest world-z in a set of (min, max) bounding boxes."""
    if not link_bounds:
        raise ValueError("no link bounds to compare")
    name = min(link_bounds, key=lambda link: float(link_bounds[link][0][2]))
    return name, float(link_bounds[name][0][2])


def build_report(
    *,
    profile_id: str,
    asset_reference: str,
    base_pose: str,
    elbow_joint_value_rad: float,
    elbow_joint_limit_rad: tuple[float, float],
    elbow_axis: str,
    joint_values: Mapping[str, float],
    hands: Mapping[str, Mapping[str, Any]],
    scene: Mapping[str, Any] | None,
    frames: Sequence[str],
) -> dict[str, Any]:
    """The measurement record: the pose, both hands, and the scene it was read on."""
    lowest = {side: float(hand["min_z_m"]) for side, hand in hands.items()}
    report: dict[str, Any] = {
        "profile_id": profile_id,
        "robot_asset_reference": asset_reference,
        "base_pose": base_pose,
        "pose": {
            "elbow_joint_value_rad": elbow_joint_value_rad,
            "elbow_joint_axis": elbow_axis,
            "elbow_joint_limits_rad": list(elbow_joint_limit_rad),
            "joint_values_rad": dict(sorted(joint_values.items())),
        },
        "hands": {side: dict(hand) for side, hand in hands.items()},
        "lowest_hand_z_m": lowest,
        "common_lowest_hand_z_m": min(lowest.values()),
        "cube_size_m": None,
        "table_surface_m": None,
        "surface_from_cube_top_m": None,
        "scene": dict(scene) if scene is not None else None,
        "frames": list(frames),
    }
    if scene is not None:
        cube_size = float(scene["cube_size_m"])
        surface = float(scene["declared_surface_height_m"])
        report["cube_size_m"] = cube_size
        report["table_surface_m"] = surface
        report["surface_from_cube_top_m"] = min(lowest.values()) - cube_size
        report["hand_minus_cube_top_m"] = {
            side: low - (surface + cube_size) for side, low in lowest.items()
        }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="write the measurement record here as JSON")
    parser.add_argument(
        "--frame",
        type=Path,
        help="write a validation-camera JPEG of the measured pose here (.jpg/.png by suffix)",
    )
    parser.add_argument("--head-camera-frame", type=Path, help="write the head-camera JPEG here")
    parser.add_argument(
        "--include-scene",
        action="store_true",
        help="build the profile's table, cubes and target too, so the hands can be read against them",
    )
    parser.add_argument(
        "--no-flexion",
        action="store_true",
        help="keep the declared start-up pose unchanged instead of bending the elbows to 90 degrees",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True
    return args


def _scene_cfg(profile: RunProfile, include_scene: bool) -> Any:
    """The probe scene: the profile's robot, optionally with the profile's table and cubes."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab_assets.robots.unitree import G1_29DOF_CFG

    from .service import _CUBE_DIFFUSE_RGB, _TARGET_DIFFUSE_RGB

    robot_spec = profile.robot
    robot_cfg: ArticulationCfg = G1_29DOF_CFG.copy()
    if robot_spec.hand.dofs == 0:
        robot_cfg.actuators.pop("hands", None)
    robot_cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=robot_spec.asset_reference,
        activate_contact_sensors=True,
        # The probe reads geometry, not balance: without gravity the written
        # pose is the measured pose, and no controller is needed to hold it.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=robot_spec.fixed_base),
    )
    robot_cfg = robot_cfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=robot_cfg.init_state.replace(pos=robot_spec.initial_position_m),
    )

    camera = profile.camera
    scene_spec = profile.scene if include_scene else None
    cube_cfgs = (
        {}
        if scene_spec is None
        else {
            cube.color: RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cube_{cube.color}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=cube.position_m, rot=cube.rotation_wxyz),
                spawn=sim_utils.CuboidCfg(
                    size=cube.size_m,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=cube.mass_kg),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CUBE_DIFFUSE_RGB[cube.color]),
                ),
            )
            for cube in scene_spec.cubes
        }
    )

    @configclass
    class PoseProbeSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=1400.0, color=(0.82, 0.86, 0.92)),
        )
        key_light = AssetBaseCfg(
            prim_path="/World/KeyLight",
            spawn=sim_utils.DistantLightCfg(intensity=2800.0, color=(1.0, 0.95, 0.88)),
        )
        if scene_spec is not None:
            table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=scene_spec.table.position_m, rot=scene_spec.table.rotation_wxyz
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=scene_spec.table.asset_reference,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
            )
            cube_red = cube_cfgs["red"]
            cube_yellow = cube_cfgs["yellow"]
            cube_blue = cube_cfgs["blue"]
            target = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/TargetTape",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=scene_spec.target.position_m, rot=scene_spec.target.rotation_wxyz
                ),
                spawn=sim_utils.CuboidCfg(
                    size=scene_spec.target.size_m,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_TARGET_DIFFUSE_RGB),
                ),
            )
        robot: ArticulationCfg = robot_cfg
        head_camera = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{camera.parent_link}/{camera.name}",
            update_period=profile.camera_update_period,
            width=camera.width,
            height=camera.height,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=camera.focal_length_mm,
                horizontal_aperture=camera.horizontal_aperture_mm,
                clipping_range=(0.05, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=camera.position_m, rot=camera.rotation_wxyz, convention="world"
            ),
        )
        validation_camera = CameraCfg(
            prim_path="/World/ValidationCamera",
            update_period=profile.camera_update_period,
            width=960,
            height=720,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(2.6, 2.4, 1.6),
                rot=(-0.3610133, -0.1217959, -0.0476202, 0.9233458),
                convention="world",
            ),
        )

    return PoseProbeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    app = launcher.app
    exit_code = 2
    try:
        exit_code = _run(args, app)
    except Exception as exc:  # noqa: BLE001 - the probe reports, it does not crash
        traceback.print_exc()
        print(json.dumps({"event": "pose_probe_failed", "reason": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        print(json.dumps({"event": "pose_probe_app_close", "exit_code": exit_code}), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


def _run(args: argparse.Namespace, app: Any) -> int:
    import isaaclab.sim as sim_utils
    import numpy as np
    from isaaclab.scene import InteractiveScene
    from pxr import Usd, UsdGeom, UsdPhysics

    from ...controllers.sonic import named_pose

    profile = RunProfile.load(args.profile)
    robot_path = "/World/envs/env_0/Robot"
    scene_path = "/World/envs/env_0"

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=profile.physics_dt,
            device=profile.device,
            use_fabric=profile.device.startswith("cuda"),
            render_interval=1,
            render=sim_utils.RenderCfg(rendering_mode="balanced", enable_dl_denoiser=True),
        )
    )
    scene = InteractiveScene(_scene_cfg(profile, args.include_scene))
    sim.reset()
    robot = scene["robot"]

    # A pose probe must not be driven: the same zeroed drives the service uses
    # keep the declared joint values the ones that are measured.
    import torch

    joints = list(range(len(robot.joint_names)))
    zeros = torch.zeros((1, len(joints)), device=robot.device)
    robot.write_joint_stiffness_to_sim(zeros, joint_ids=joints)
    robot.write_joint_damping_to_sim(zeros, joint_ids=joints)
    robot.set_joint_effort_target(zeros, joint_ids=joints)

    base_pose, _root_height = named_pose(profile.initial_pose)
    stage = sim.stage
    # The joint frames, axis and limits come from the asset file itself: the
    # angle is only meaningful next to the frames it was solved from.
    asset_stage = Usd.Stage.Open(profile.robot.asset_reference)
    asset_usd = _asset_joint_frames(asset_stage, UsdPhysics)
    elbow_offsets = {side: asset_usd[side]["offsets"] for side in ("left", "right")}
    elbow_axis = asset_usd["left"]["axis"]
    elbow_limits_rad = asset_usd["left"]["limits_rad"]
    if asset_usd["right"]["axis"] != elbow_axis or asset_usd["right"]["limits_rad"] != elbow_limits_rad:
        raise RuntimeError("left and right elbow axes/limits differ; the asset is not mirrored")
    live_offsets = {
        side: _live_joint_offsets(stage, UsdPhysics, robot_path, side) for side in ("left", "right")
    }
    live_match = all(
        all(abs(a - b) < 1e-6 for a, b in zip(value, elbow_offsets[side][0]))
        for side, value in live_offsets.items()
        if value is not None
    )
    solved = perpendicular_elbow_angle(*elbow_offsets["left"])
    if abs(solved - perpendicular_elbow_angle(*elbow_offsets["right"])) > 1e-6:
        raise RuntimeError("left and right elbow frames disagree; the asset is not mirrored as expected")
    if not elbow_limits_rad[0] <= solved <= elbow_limits_rad[1]:
        raise RuntimeError(f"solved elbow angle {solved} rad sits outside the asset limits {elbow_limits_rad}")

    applied_pose = base_pose if args.no_flexion else with_elbow_flexion(base_pose, solved)
    joint_values = dict(applied_pose)
    names = list(robot.joint_names)
    positions = robot.data.default_joint_pos.clone()
    for name, value in applied_pose.items():
        if name not in names:
            raise RuntimeError(f"the asset has no joint {name!r}")
        positions[0, names.index(name)] = float(value)
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions))

    from omni.timeline import get_timeline_interface

    timeline = get_timeline_interface()
    timeline.play()
    for _ in range(3):
        scene.write_data_to_sim()
        sim.step(render=False)
    robot.update(0.0)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    positions_w = robot.data.body_pos_w[0]
    joint_readback = robot.data.joint_pos[0]
    body_index = {name: index for index, name in enumerate(robot.body_names)}
    joint_index = {name: index for index, name in enumerate(robot.joint_names)}
    hands: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        bounds = {
            link: _world_bounds(stage, cache, f"{robot_path}/{link}") for link in hand_link_names(side)
        }
        lowest_name, lowest_z = lowest_link(bounds)
        shoulder = positions_w[body_index[UPPER_ARM_LINK.format(side=side)]].tolist()
        elbow = positions_w[body_index[ELBOW_LINK.format(side=side)]].tolist()
        wrist = positions_w[body_index[FOREARM_LINK.format(side=side)]].tolist()
        upper = [b - a for a, b in zip(shoulder, elbow)]
        forearm = [b - a for a, b in zip(elbow, wrist)]
        hands[side] = {
            "elbow_bend_angle_deg": arm_bend_angle_deg(shoulder, elbow, wrist),
            "min_z_m": lowest_z,
            "lowest_link": lowest_name,
            "lowest_link_bounds_m": list(bounds[lowest_name]),
            "link_min_z_m": {link: float(bound[0][2]) for link, bound in sorted(bounds.items())},
            # The realized arm geometry, so a measured bend can be checked
            # against the frames the angle was solved from instead of assumed.
            "joint_readback_rad": float(joint_readback[joint_index[f"{side}_elbow_joint"]]),
            "shoulder_yaw_pos_m": shoulder,
            "elbow_pos_m": elbow,
            "wrist_roll_pos_m": wrist,
            "upper_arm_vector_m": upper,
            "forearm_vector_m": forearm,
            "upper_arm_length_m": math.sqrt(sum(value * value for value in upper)),
            "forearm_length_m": math.sqrt(sum(value * value for value in forearm)),
        }

    scene_report: dict[str, Any] | None = None
    if profile.scene is not None and args.include_scene:
        table = profile.scene.table
        cube = profile.scene.cubes[0]
        worktop = _world_bounds(stage, cache, f"{scene_path}/Table")
        cube_top = _world_bounds(stage, cache, f"{scene_path}/Cube_{cube.color}")
        robot_bounds = _world_bounds(stage, cache, robot_path)
        table_bounds = worktop
        scene_report = {
            "declared_surface_height_m": table.surface_height_m,
            "live_worktop_top_z_m": float(worktop[1][2]),
            "live_worktop_minus_declared_m": float(worktop[1][2]) - table.surface_height_m,
            "cube_color": cube.color,
            "cube_size_m": float(cube.size_m[2]),
            "declared_cube_top_z_m": cube.position_m[2] + cube.size_m[2] / 2.0,
            "live_cube_top_z_m": float(cube_top[1][2]),
            "robot_bounds_m": [list(robot_bounds[0]), list(robot_bounds[1])],
            "table_bounds_m": [list(table_bounds[0]), list(table_bounds[1])],
            "robot_table_gap_x_m": float(table_bounds[0][0] - robot_bounds[1][0]),
        }

    frames: list[str] = []
    sim.render()
    scene.update(profile.camera_update_period)
    frames.extend(_write_frame(scene["validation_camera"], args.frame, np))
    frames.extend(_write_frame(scene["head_camera"], args.head_camera_frame, np))

    # The world placement of the robot and its head camera: the pose numbers
    # above only mean something next to the frame they were taken in.
    placement: dict[str, Any] = {
        "default_root_state": robot.data.default_root_state[0].tolist(),
        "root_pos_w_m": robot.data.root_pos_w[0].tolist(),
        "root_quat_wxyz": robot.data.root_quat_w[0].tolist(),
        "pelvis_pos_m": positions_w[body_index["pelvis"]].tolist(),
        "torso_pos_m": positions_w[body_index["torso_link"]].tolist(),
        "head_camera": _camera_world_pose(
            stage, robot_path, profile.camera.parent_link, profile.camera.name
        ),
    }

    report = build_report(
        profile_id=profile.profile_id,
        asset_reference=profile.robot.asset_reference,
        base_pose=profile.initial_pose,
        elbow_joint_value_rad=solved,
        elbow_joint_limit_rad=elbow_limits_rad,
        elbow_axis=elbow_axis,
        joint_values=joint_values,
        hands=hands,
        scene=scene_report,
        frames=frames,
    )
    report["elbow_offsets_m"] = {
        side: {"upper_arm_m": list(offsets[0]), "forearm_m": list(offsets[1])}
        for side, offsets in elbow_offsets.items()
    }
    report["live_stage_joint_offsets_m"] = {
        side: None if value is None else list(value) for side, value in live_offsets.items()
    }
    report["live_stage_matches_asset_frames"] = live_match
    report["flexion_applied"] = not args.no_flexion
    report["requested_bend_angle_deg"] = None if args.no_flexion else 90.0
    violations = [
        f"{side} elbow measures {hand['elbow_bend_angle_deg']:.2f} deg, "
        f"expected 90 +/- {BEND_ANGLE_TOLERANCE_DEG}"
        for side, hand in report["hands"].items()
        if not args.no_flexion
        and abs(float(hand["elbow_bend_angle_deg"]) - 90.0) > BEND_ANGLE_TOLERANCE_DEG
    ]
    report["pose_validation"] = "pass" if not violations else "fail"
    report["pose_validation_notes"] = violations
    # The measurement is written before its validation decides the exit code: a
    # reading that misses the requested angle is still the reading.
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    timeline.pause()
    return 0 if not violations else 1


def _asset_joint_frames(stage: Any, UsdPhysics: Any) -> dict[str, dict[str, Any]]:
    """Per-side elbow frame data read from the asset USD.

    ``offsets`` holds the upper-arm vector and the forearm vector in the frames
    their joint is authored in, which is the geometry the perpendicularity
    solve needs.  USD physics stores revolute limits in degrees; joint state is
    radians, so the limits are converted here once.
    """
    frames: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        # Joint prims are authored under their parent link, not under the asset
        # root, so the joint is located by name.
        elbow = _find_prim(stage.GetDefaultPrim(), f"{side}_elbow_joint")
        wrist = _find_prim(stage.GetDefaultPrim(), f"{side}_wrist_roll_joint")
        if elbow is None or wrist is None:
            raise RuntimeError(f"asset {stage.GetRootLayer().identifier} is missing the {side} arm joints")
        elbow_joint = UsdPhysics.RevoluteJoint(elbow)
        frames[side] = {
            "offsets": (
                tuple(float(value) for value in elbow_joint.GetLocalPos0Attr().Get()),
                tuple(float(value) for value in UsdPhysics.RevoluteJoint(wrist).GetLocalPos0Attr().Get()),
            ),
            "axis": str(elbow_joint.GetAxisAttr().Get()),
            "limits_rad": (
                math.radians(float(elbow_joint.GetLowerLimitAttr().Get())),
                math.radians(float(elbow_joint.GetUpperLimitAttr().Get())),
            ),
        }
    return frames


def _find_prim(root: Any, name: str) -> Any | None:
    """First prim named ``name`` in the subtree, or None."""
    from pxr import Usd

    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    return None


def _live_joint_offsets(
    stage: Any, UsdPhysics: Any, robot_path: str, side: str
) -> tuple[float, float, float] | None:
    """Elbow joint offset as the spawned stage carries it, when it is exposed."""
    robot = stage.GetPrimAtPath(robot_path)
    prim = _find_prim(robot, f"{side}_elbow_joint") if robot and robot.IsValid() else None
    if prim is None:
        return None
    joint = UsdPhysics.RevoluteJoint(prim)
    attribute = joint.GetLocalPos0Attr() if joint else None
    value = attribute.Get() if attribute and attribute.IsValid() else None
    return None if value is None else tuple(float(component) for component in value)


def _world_bounds(
    stage: Any, cache: Any, path: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"live stage is missing {path}")
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if aligned.IsEmpty():
        raise RuntimeError(f"live stage prim {path} has no geometry bounds")
    return (
        tuple(float(value) for value in aligned.GetMin()),
        tuple(float(value) for value in aligned.GetMax()),
    )


def _write_frame(camera: Any, path: Path | None, np: Any) -> list[str]:
    if path is None:
        return []
    image = camera.data.output.get("rgb")
    if image is None:
        return []
    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    array = np.ascontiguousarray(array[..., :3].astype(np.uint8, copy=False))
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    # The camera output is RGB; OpenCV writes BGR.
    if not cv2.imwrite(str(path), np.ascontiguousarray(array[..., ::-1])):
        raise RuntimeError(f"could not write the probe frame to {path}")
    return [str(path)]


if __name__ == "__main__":
    raise SystemExit(main())
