"""Controller-free Isaac Lab G1 simulator."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import RobotState, RunProfile, TimelineState, quaternion_up_z
from .evidence import EvidenceSink, StreamingVideoWriter


@dataclass(frozen=True)
class AcceptanceThresholds:
    minimum_drop_m: float = 0.12
    minimum_up_change: float = 0.15
    minimum_frames: int = 2
    minimum_changed_pixels: int = 1
    minimum_root_z: float = -0.50


class SimulatorService:
    """Own the one scene, one reset, one physics loop, and its evidence."""

    def __init__(
        self,
        profile: RunProfile,
        simulation_app: Any,
        *,
        run_id: str,
        output_root: Path,
        duration: float,
        record: bool,
        show_ui: bool,
        test_mode: str | None,
        capture_every: int = 4,
    ) -> None:
        self.profile = profile
        self.app = simulation_app
        self.run_id = run_id
        self.duration = duration
        self.show_ui = show_ui
        self.test_mode = test_mode
        self.capture_every = max(1, capture_every)
        fps = max(1, round(1.0 / profile.physics_dt / self.capture_every))
        self.evidence = EvidenceSink(output_root / "runs" / run_id, record=record, fps=fps)
        self.state = TimelineState.STARTING
        self.tick = 0
        self.episode_id = 1
        self._sim: Any = None
        self._scene: Any = None
        self._timeline: Any = None
        self._robot: Any = None
        self._body_ids: list[int] = []
        self._hand_ids: list[int] = []
        self._viewport_capture: Any = None
        self._control_window: Any = None
        self._head_camera_panel: Any = None
        self._head_camera_provider: Any = None
        self._ui_state_label: Any = None
        self._ui_tick = 0
        self._pending_ui_reset = False
        self._simulation_output = "fabric"
        self._root_z: list[float] = []
        self._root_up: list[float] = []
        self._simulated_times: list[float] = []

    def _transition(self, state: TimelineState, **details: Any) -> None:
        self.state = state
        self.evidence.status.write(
            {"run_id": self.run_id, "component": "simulator", "state": state.value, **details}
        )

    def start(self) -> None:
        import carb
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim.utils.stage import attach_stage_to_usd_context
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.timeline import get_timeline_interface

        print('{"event":"isaac_g1_start","stage":"simulation_context"}', flush=True)
        settings = carb.settings.get_settings()
        use_fabric = True
        settings.set_bool("/physics/fabricEnabled", use_fabric)
        settings.set_bool("/physics/updateToUsd", not use_fabric)
        SimulationManager.enable_fabric(use_fabric)
        self._sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=self.profile.physics_dt,
                device="cuda:0",
                use_fabric=use_fabric,
                render_interval=4,
            )
        )
        print('{"event":"isaac_g1_start","stage":"interactive_scene"}', flush=True)
        self._scene = InteractiveScene(self._make_scene_cfg())
        print('{"event":"isaac_g1_start","stage":"free_base"}', flush=True)
        self._configure_free_base_if_needed()
        attach_stage_to_usd_context()
        print('{"event":"isaac_g1_start","stage":"hard_reset"}', flush=True)
        self._sim.reset()  # The sole hard reset, after all topology exists.
        self._robot = self._scene["robot"]
        print('{"event":"isaac_g1_start","stage":"passive_actuators"}', flush=True)
        self._resolve_and_disable_actuators()
        self._timeline = get_timeline_interface()
        self._timeline.pause()
        self._set_debug_camera()
        self._viewport_capture = self._make_viewport_capture()
        if self.show_ui:
            self._open_ui()
        print('{"event":"isaac_g1_start","stage":"evidence"}', flush=True)
        self._write_manifest_and_provenance()
        self._transition(
            TimelineState.PAUSED,
            controller="absent",
            control_mode="passive",
            simulation_output=self._simulation_output,
        )

    def play(self) -> None:
        self._timeline.play()
        self._transition(TimelineState.PLAYING)

    def pause(self) -> None:
        self._timeline.pause()
        self._transition(TimelineState.PAUSED)

    def stop(self) -> None:
        if self._timeline is not None:
            self._timeline.stop()
        self._transition(TimelineState.STOPPING)

    def reset_episode(self, *, resume: bool | None = None) -> None:
        """Soft-reset tensors without rebuilding topology or hard-resetting Kit."""
        was_playing = bool(self._timeline.is_playing())
        should_resume = was_playing if resume is None else resume
        default_root = self._write_initial_state_to_sim()
        self._scene.reset()
        self._resolve_and_disable_actuators()
        self.episode_id += 1
        self.tick = 0
        self._root_z.clear()
        self._root_up.clear()
        self._simulated_times.clear()
        self._transition(
            TimelineState.PLAYING if should_resume else TimelineState.PAUSED,
            event="soft_reset",
            episode_id=self.episode_id,
            root_position=[float(value) for value in default_root[0, :3].tolist()],
        )
        if should_resume and not self._timeline.is_playing():
            self.play()
        elif not should_resume and self._timeline.is_playing():
            self.pause()

    def _write_initial_state_to_sim(self) -> Any:
        """Write the profile's initial root, velocity, and joint state."""
        import torch

        default_root = self._robot.data.default_root_state.clone()
        default_root[:, :3] += self._scene.env_origins
        self._robot.write_root_pose_to_sim(default_root[:, :7])
        self._robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=self._robot.device))
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos, self._robot.data.default_joint_vel.clone().zero_()
        )
        return default_root

    def _request_reset_from_ui(self) -> None:
        """Queue reset work for the simulation-loop boundary."""
        self._pending_ui_reset = True
        print('{"event":"isaac_g1_ui_reset_requested"}', flush=True)

    def _open_ui(self) -> None:
        from omni import ui

        self._head_camera_panel = ui.Window(
            "G1 Head Camera",
            width=430,
            height=330,
            position_x=980,
            position_y=500,
            raster_policy=ui.RasterPolicy.NEVER,
        )
        with self._head_camera_panel.frame:
            self._head_camera_provider = ui.ByteImageProvider()
            ui.ImageWithProvider(
                self._head_camera_provider,
                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
            )
        self._control_window = ui.Window(
            "G1 Simulator", width=300, height=110, position_x=1090, position_y=120
        )
        with self._control_window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Passive robot: motors disabled")
                ui.Button("Reset Robot", height=36, clicked_fn=self._request_reset_from_ui)
                ui.Label("Reset restores the initial pose and pauses physics.")

    def _make_scene_cfg(self) -> Any:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils import configclass
        from isaaclab_assets.robots.unitree import G1_29DOF_CFG, G1_INSPIRE_FTP_CFG

        robot_spec = self.profile.robot
        if robot_spec.asset_kind == "isaaclab_config":
            configs = {"G1_29DOF_CFG": G1_29DOF_CFG, "G1_INSPIRE_FTP_CFG": G1_INSPIRE_FTP_CFG}
            robot_cfg: ArticulationCfg = configs[robot_spec.asset_reference].copy()
        else:
            robot_cfg = G1_29DOF_CFG.copy()
            if robot_spec.hand.dofs == 0:
                robot_cfg.actuators.pop("hands", None)
            robot_cfg.spawn = sim_utils.UsdFileCfg(
                usd_path=robot_spec.asset_reference,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
            )
        robot_cfg.spawn.articulation_props.fix_root_link = False
        robot_cfg.spawn.rigid_props.disable_gravity = False
        robot_cfg = robot_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=robot_cfg.init_state.replace(pos=robot_spec.initial_position_m),
        )
        camera = self.profile.camera
        camera_path = f"{{ENV_REGEX_NS}}/Robot/{camera.parent_link}/{camera.name}"

        @configclass
        class IsaacG1SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            dome_light = AssetBaseCfg(
                prim_path="/World/DomeLight",
                spawn=sim_utils.DomeLightCfg(intensity=1400.0, color=(0.82, 0.86, 0.92)),
            )
            key_light = AssetBaseCfg(
                prim_path="/World/KeyLight",
                spawn=sim_utils.DistantLightCfg(intensity=2800.0, color=(1.0, 0.95, 0.88)),
            )
            robot: ArticulationCfg = robot_cfg
            head_camera = CameraCfg(
                prim_path=camera_path,
                update_period=0.0,
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

        return IsaacG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    def _configure_free_base_if_needed(self) -> None:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        robot_path = "/World/envs/env_0/Robot"
        pelvis = stage.GetPrimAtPath(f"{robot_path}/pelvis")
        if not pelvis or not pelvis.IsValid():
            raise RuntimeError("G1 asset is missing its pelvis prim")
        moved_root = False
        for name in ("root_joint", "floating_base_joint"):
            joint_prim = stage.GetPrimAtPath(f"{robot_path}/{name}")
            if not joint_prim or not joint_prim.IsValid():
                continue
            joint = UsdPhysics.Joint(joint_prim)
            if joint:
                joint.GetJointEnabledAttr().Set(False)
            joint_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            joint_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            moved_root = True
        if moved_root and not UsdPhysics.ArticulationRootAPI(pelvis):
            UsdPhysics.ArticulationRootAPI.Apply(pelvis)
        if moved_root and not PhysxSchema.PhysxArticulationAPI(pelvis):
            PhysxSchema.PhysxArticulationAPI.Apply(pelvis)

    def _resolve_and_disable_actuators(self) -> None:
        import torch

        names = list(self._robot.joint_names)
        hand_patterns = self.profile.robot.hand.joint_name_patterns
        self._hand_ids = [
            index for index, name in enumerate(names) if any(pattern in name for pattern in hand_patterns)
        ]
        self._body_ids = [index for index in range(len(names)) if index not in set(self._hand_ids)]
        if len(self._body_ids) != self.profile.robot.body_dofs:
            raise RuntimeError(f"asset exposes {len(self._body_ids)} body joints, expected 29")
        if len(self._hand_ids) != self.profile.robot.hand.dofs:
            raise RuntimeError(
                f"asset exposes {len(self._hand_ids)} {self.profile.robot.hand.kind} joints, "
                f"expected {self.profile.robot.hand.dofs}"
            )
        all_ids = list(range(len(names)))
        zeros = torch.zeros((1, len(all_ids)), device=self._robot.device)
        self._robot.write_joint_stiffness_to_sim(zeros, joint_ids=all_ids)
        self._robot.write_joint_damping_to_sim(zeros, joint_ids=all_ids)
        self._robot.set_joint_effort_target(zeros, joint_ids=all_ids)
        stiffness = self._robot.data.joint_stiffness[0, all_ids]
        damping = self._robot.data.joint_damping[0, all_ids]
        if bool(torch.any(stiffness != 0.0)) or bool(torch.any(damping != 0.0)):
            raise RuntimeError("passive actuator verification failed: non-zero drive remains")

    def _set_debug_camera(self) -> None:
        if hasattr(self._sim, "set_camera_view"):
            self._sim.set_camera_view(eye=(2.6, 2.4, 1.6), target=(0.0, 0.0, 0.65))

    @staticmethod
    def _make_viewport_capture() -> Any:
        try:
            import omni.replicator.core as rep
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            product = getattr(viewport, "render_product_path", None)
            if not product:
                return None
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            return lambda: annotator.get_data()
        except Exception:  # noqa: BLE001
            return None

    def _write_manifest_and_provenance(self) -> None:
        manifest = self.profile.as_manifest()
        manifest.update(
            {
                "run_id": self.run_id,
                "controller": "absent",
                "root": "free",
                "gravity": "enabled",
                "ground_collision": "enabled",
                "body_actuator": "passive",
                "test_mode": self.test_mode,
                "simulation_output": self._simulation_output,
            }
        )
        self.evidence.write_manifest(manifest)
        self.evidence.policy_metrics.write({"run_id": self.run_id, "state": "not_applicable"})
        self.evidence.controller_metrics.write(
            {"run_id": self.run_id, "state": "absent", "control_mode": "passive"}
        )
        self.evidence.write_json("model-provenance.json", {"state": "not_applicable"})
        self.evidence.write_json("task-metrics.json", {"state": "not_applicable"})
        self.evidence.write_json(
            "asset-provenance.json",
            {
                "asset_reference": self.profile.robot.asset_reference,
                "declared_provenance": self.profile.robot.asset_provenance,
                "sonic_commit": self._git_revision("/opt/src/sonic"),
                "isaaclab_commit": self._git_revision("/opt/src/isaaclab"),
            },
        )
        self.evidence.write_json("asset-report.json", self._asset_report())
        (self.evidence.run_dir / "graph.mmd").write_text(
            "graph LR\n  CLI --> SimulatorService\n  SimulatorService --> IsaacLab\n"
            "  IsaacLab --> G1\n  G1 --> HeadCamera\n  IsaacLab --> DebugViewport\n"
            "  HeadCamera --> EvidenceSink\n  DebugViewport --> EvidenceSink\n"
            "  G1 --> EvidenceSink\n",
            encoding="utf-8",
        )

    @staticmethod
    def _git_revision(path: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def _asset_report(self) -> dict[str, Any]:
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        prims = list(Usd.PrimRange(robot, Usd.TraverseInstanceProxies()))
        collisions = []
        masses = []
        for prim in prims:
            collision = UsdPhysics.CollisionAPI(prim)
            if collision:
                enabled = collision.GetCollisionEnabledAttr().Get()
                collisions.append({"prim": str(prim.GetPath()), "enabled": enabled is not False})
            mass_api = UsdPhysics.MassAPI(prim)
            if mass_api:
                diagonal = mass_api.GetDiagonalInertiaAttr().Get()
                center = mass_api.GetCenterOfMassAttr().Get()
                masses.append(
                    {
                        "prim": str(prim.GetPath()),
                        "mass_kg": mass_api.GetMassAttr().Get(),
                        "diagonal_inertia": None if diagonal is None else list(diagonal),
                        "center_of_mass": None if center is None else list(center),
                    }
                )
        collision_count = len(collisions)
        mass_count = len(masses)
        return {
            "profile_id": self.profile.profile_id,
            "prim_count": len(prims),
            "collision_prim_count": collision_count,
            "mass_or_inertia_prim_count": mass_count,
            "joint_count": len(self._robot.joint_names),
            "body_dofs": len(self._body_ids),
            "hand_dofs": len(self._hand_ids),
            "joint_names": list(self._robot.joint_names),
            "body_names": list(self._robot.body_names),
            "collisions": collisions,
            "mass_and_inertia": masses,
            "mesh_paths_resolved": bool(prims and collision_count and mass_count),
            "mesh_path_note": "runtime-loaded USD traversal found populated collision and mass prims",
        }

    def _robot_state(self) -> RobotState:
        robot = self._robot
        root_pos = tuple(float(v) for v in robot.data.root_pos_w[0].tolist())
        root_quat = tuple(float(v) for v in robot.data.root_quat_w[0].tolist())
        body_pos = tuple(float(robot.data.joint_pos[0, i]) for i in self._body_ids)
        body_vel = tuple(float(robot.data.joint_vel[0, i]) for i in self._body_ids)
        return RobotState(
            run_id=self.run_id,
            episode_id=self.episode_id,
            physics_tick=self.tick,
            simulated_time=float(self._sim.current_time),
            root_position=root_pos,
            root_rotation_wxyz=root_quat,
            body_position=body_pos,
            body_velocity=body_vel,
            timeline_state=self.state,
        )

    def _record_tick(self, state: RobotState) -> None:
        metric = state.as_dict()
        camera_metric = None
        viewport_metric = None
        if self.tick % self.capture_every == 0:
            camera = self._scene["head_camera"].data.output.get("rgb")
            if self.evidence.head_video is not None and camera is not None:
                camera_metric = self.evidence.head_video.append(camera)
            if self.evidence.viewport_video is not None and self._viewport_capture is not None:
                viewport_metric = self.evidence.viewport_video.append(self._viewport_capture())
        metric["camera"] = camera_metric
        metric["viewport"] = viewport_metric
        self.evidence.metrics.write(metric)

    def _update_head_camera_panel(self) -> None:
        if self._head_camera_provider is None:
            return
        camera = self._scene["head_camera"].data.output.get("rgb")
        if camera is None:
            return
        try:
            import numpy as np

            image = StreamingVideoWriter.normalize(camera)
            rgba = np.empty((*image.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = image
            rgba[..., 3] = 255
            self._head_camera_provider.set_data_array(
                rgba, [int(image.shape[1]), int(image.shape[0])]
            )
        except (TypeError, ValueError):
            return

    def _step_physics(self, *, record: bool) -> None:
        """Advance the sole physics owner once, optionally recording the tick."""
        self._robot.set_joint_effort_target(0.0)
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self.profile.physics_dt)
        self._ui_tick += 1
        if self._ui_tick % self.capture_every == 0:
            self._update_head_camera_panel()
        if not record:
            return
        self.tick += 1
        state = self._robot_state()
        self._root_z.append(state.root_position[2])
        self._root_up.append(quaternion_up_z(state.root_rotation_wxyz))
        self._simulated_times.append(state.simulated_time)
        self._record_tick(state)

    def run(self) -> dict[str, Any]:
        self.start()
        self.play()
        started = time.monotonic()
        try:
            while self.app.is_running() and time.monotonic() - started < self.duration:
                if self._pending_ui_reset:
                    self._pending_ui_reset = False
                    # Match the official Isaac Lab reset order: write state
                    # while playing, publish it through the sole physics step,
                    # then pause on the freshly rendered initial pose.
                    self.reset_episode(resume=True)
                    self._step_physics(record=False)
                    self.pause()
                    print('{"event":"isaac_g1_ui_reset","timeline":"paused"}', flush=True)
                    continue
                if self._timeline.is_playing():
                    if self.state is not TimelineState.PLAYING:
                        self._transition(TimelineState.PLAYING)
                    self._step_physics(record=True)
                else:
                    if self.state is not TimelineState.PAUSED:
                        self._transition(TimelineState.PAUSED)
                    self._sim.render()  # Paused loop renders and never advances physics.
        except Exception as exc:
            self._transition(TimelineState.FAILED, error=str(exc))
            raise
        finally:
            self.stop()
        self._transition(TimelineState.STOPPED)
        videos = self.evidence.close()
        summary = (
            self._passive_fall_acceptance(videos)
            if self.test_mode == "passive-fall"
            else self._runtime_summary(videos)
        )
        self.evidence.write_json("summary.json", summary)
        return summary

    def _runtime_summary(self, videos: dict[str, Any]) -> dict[str, Any]:
        return {
            "result": "COMPLETED",
            "run_id": self.run_id,
            "profile_id": self.profile.profile_id,
            "physics_ticks": self.tick,
            "episodes": self.episode_id,
            "videos": videos,
        }

    def _passive_fall_acceptance(self, videos: dict[str, Any]) -> dict[str, Any]:
        thresholds = AcceptanceThresholds()
        initial_z = self._root_z[0] if self._root_z else None
        minimum_z = min(self._root_z) if self._root_z else None
        drop = 0.0 if initial_z is None or minimum_z is None else initial_z - minimum_z
        up_change = 0.0 if not self._root_up else abs(self._root_up[-1] - self._root_up[0])
        fell = drop >= thresholds.minimum_drop_m or up_change >= thresholds.minimum_up_change
        head = videos["head_camera"]
        viewport = videos["debug_viewport"]
        physics_time_monotonic = len(self._simulated_times) > 1 and all(
            later > earlier for earlier, later in zip(self._simulated_times, self._simulated_times[1:])
        )
        expected_shape = [self.profile.camera.height, self.profile.camera.width, 3]
        checks = {
            "controller_absent": True,
            "physics_tick_monotonic": self.tick > 1 and physics_time_monotonic,
            "robot_fell": fell,
            "floor_bounded": bool(self._root_z and minimum_z >= thresholds.minimum_root_z),
            "head_camera_flowing": head.get("frames", 0) >= thresholds.minimum_frames
            and head.get("changed_pixels", 0) >= thresholds.minimum_changed_pixels
            and head.get("shape") == expected_shape,
            "clean_stop": True,
        }
        if self.show_ui:
            checks["debug_viewport_flowing"] = (
                viewport.get("frames", 0) >= thresholds.minimum_frames
                and viewport.get("changed_pixels", 0) >= thresholds.minimum_changed_pixels
            )
        result = "PASS" if all(checks.values()) else "FAIL"
        return {
            "result": result,
            "run_id": self.run_id,
            "profile_id": self.profile.profile_id,
            "checks": checks,
            "physics_ticks": self.tick,
            "initial_root_z": initial_z,
            "minimum_root_z": minimum_z,
            "root_drop_m": drop,
            "pelvis_up_change": up_change,
            "videos": videos,
        }
