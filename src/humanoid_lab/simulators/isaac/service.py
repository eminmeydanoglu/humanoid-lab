"""Controller-free Isaac Lab G1 simulator."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from .contracts import RunProfile, TimelineState, quaternion_up_z


@dataclass(frozen=True)
class AcceptanceThresholds:
    minimum_drop_m: float = 0.12
    minimum_up_change: float = 0.15
    minimum_root_z: float = -0.50


class SimulatorService:
    """Own the one scene, one reset, and one physics loop."""

    PAUSED_RENDER_HZ = 30.0
    PHYSX_NUM_THREADS = 4

    def __init__(
        self,
        profile: RunProfile,
        simulation_app: Any,
        *,
        duration: float,
        show_ui: bool,
        show_head_camera: bool,
        test_mode: str | None,
    ) -> None:
        self.profile = profile
        self.app = simulation_app
        self.duration = duration
        self.show_ui = show_ui
        self.show_head_camera = show_head_camera
        self.test_mode = test_mode
        self.state = TimelineState.STARTING
        self.tick = 0
        self._sim: Any = None
        self._scene: Any = None
        self._timeline: Any = None
        self._robot: Any = None
        self._body_ids: list[int] = []
        self._hand_ids: list[int] = []
        self._control_window: Any = None
        self._head_camera_panel: Any = None
        self._pending_ui_reset = False
        self._device = profile.device
        self._is_rendering = False
        self._run_started = time.monotonic()
        self._wall_physics_elapsed = 0.0
        self._last_perf_report = self._run_started
        self._last_perf_tick = 0
        self._perf_step_seconds = 0.0
        self._perf_render_seconds = 0.0
        self._perf_render_count = 0
        self._last_paused_render = 0.0
        self._root_z: list[float] = []
        self._root_up: list[float] = []
        self._camera_frames = 0
        self._camera_shape: list[int] | None = None
        self._camera_changed_frames = 0
        self._previous_camera_digest: bytes | None = None

    def _transition(self, state: TimelineState) -> None:
        self.state = state

    def start(self) -> None:
        import carb
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim.utils.stage import attach_stage_to_usd_context
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.timeline import get_timeline_interface

        print('{"event":"isaac_g1_start","stage":"simulation_context"}', flush=True)
        settings = carb.settings.get_settings()
        # Isaac Lab's Fabric/Warp view path does not support a CPU device.
        use_fabric = self._device.startswith("cuda")
        # PhysX reads the thread count when the physics scene is created.
        settings.set_int("/persistent/physics/numThreads", self.PHYSX_NUM_THREADS)
        settings.set_bool("/physics/fabricEnabled", use_fabric)
        settings.set_bool("/physics/updateToUsd", not use_fabric)
        SimulationManager.enable_fabric(use_fabric)
        self._sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=self.profile.physics_dt,
                device=self._device,
                use_fabric=use_fabric,
                # This service owns the outer render cadence. Passing the same
                # interval here makes Kit wait that full period again inside
                # app.update(), effectively charging it twice.
                render_interval=1,
                render=sim_utils.RenderCfg(
                    rendering_mode="balanced",
                    enable_dlssg=self.show_ui
                    and self.test_mode is None
                    and not self.show_head_camera,
                    enable_dl_denoiser=True,
                    dlss_mode=1,
                ),
            )
        )
        print('{"event":"isaac_g1_start","stage":"interactive_scene"}', flush=True)
        self._scene = InteractiveScene(self._make_scene_cfg())
        # Mirror DirectRLEnv: rendering is only needed for a GUI or an RTX sensor.
        self._is_rendering = self._sim.has_gui() or self._sim.has_rtx_sensors()
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
        if self.show_ui:
            self._open_ui()
        self._transition(TimelineState.PAUSED)

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
        self.tick = 0
        self._wall_physics_elapsed = 0.0
        self._last_perf_report = time.monotonic()
        self._last_perf_tick = 0
        self._perf_step_seconds = 0.0
        self._perf_render_seconds = 0.0
        self._perf_render_count = 0
        self._root_z.clear()
        self._root_up.clear()
        self._camera_frames = 0
        self._camera_shape = None
        self._camera_changed_frames = 0
        self._previous_camera_digest = None
        self._transition(TimelineState.PLAYING if should_resume else TimelineState.PAUSED)
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
        from omni.kit.viewport.utility import create_viewport_window
        from pxr import Sdf

        if self.show_head_camera:
            camera = self.profile.camera
            camera_path = Sdf.Path(
                f"/World/envs/env_0/Robot/{camera.parent_link}/{camera.name}"
            )
            # Keep the camera texture on the GPU. ByteImageProvider required a
            # synchronous texture-to-host copy and a second RGB-to-RGBA copy on
            # every rendered frame, stalling both physics and the main viewport.
            self._head_camera_panel = create_viewport_window(
                name="G1 Head Camera",
                width=430,
                height=330,
                position_x=980,
                position_y=500,
                camera_path=camera_path,
            )
            if self._head_camera_panel is None:
                raise RuntimeError("failed to create the GPU-backed head-camera viewport")
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
        class IsaacG1BaseSceneCfg(InteractiveSceneCfg):
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

        @configclass
        class IsaacG1CameraSceneCfg(IsaacG1BaseSceneCfg):
            head_camera = CameraCfg(
                prim_path=camera_path,
                update_period=self.profile.camera_update_period,
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

        scene_cfg = (
            IsaacG1CameraSceneCfg
            if self.test_mode == "passive-fall" or self.show_head_camera
            else IsaacG1BaseSceneCfg
        )
        return scene_cfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

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

    def _head_camera_rgb(self) -> Any | None:
        camera = self._scene["head_camera"].data.output.get("rgb")
        if camera is None:
            return None
        try:
            import numpy as np

            image = camera.detach().cpu().numpy() if hasattr(camera, "detach") else np.asarray(camera)
            while image.ndim > 3 and image.shape[0] == 1:
                image = image[0]
            if image.ndim == 4:
                image = image[0]
            if image.ndim != 3 or image.shape[-1] not in (3, 4):
                return None
            return np.ascontiguousarray(image[..., :3].astype(np.uint8, copy=False))
        except (TypeError, ValueError):
            return None

    def _consume_head_camera_frame(self) -> None:
        """Read a frame only when the explicit acceptance test needs evidence."""
        image = self._head_camera_rgb()
        if image is None:
            return
        digest = hashlib.blake2b(image, digest_size=16).digest()
        if self._previous_camera_digest is not None and digest != self._previous_camera_digest:
            self._camera_changed_frames += 1
        self._previous_camera_digest = digest
        self._camera_frames += 1
        self._camera_shape = [int(value) for value in image.shape]

    def _step_physics(self) -> None:
        """Advance physics once, refreshing render and sensors on their own cadence."""
        step_started = time.perf_counter()
        self._scene.write_data_to_sim()
        self._sim.step(render=False)
        self.tick += 1
        rendered = self._is_rendering and self.tick % self.profile.render_interval == 0
        if rendered:
            self._perf_step_seconds += time.perf_counter() - step_started
            render_started = time.perf_counter()
            self._sim.render()
            self._perf_render_seconds += time.perf_counter() - render_started
            self._perf_render_count += 1
            step_started = time.perf_counter()
        self._scene.update(self.profile.physics_dt)
        self._perf_step_seconds += time.perf_counter() - step_started
        if rendered and self.test_mode == "passive-fall":
            self._consume_head_camera_frame()
        if self.test_mode == "passive-fall":
            self._root_z.append(float(self._robot.data.root_pos_w[0, 2]))
            root_quat = tuple(float(value) for value in self._robot.data.root_quat_w[0].tolist())
            self._root_up.append(quaternion_up_z(root_quat))

    def _render_paused(self) -> None:
        """Refresh UI and viewport at a bounded rate while physics is paused."""
        interval = 1.0 / self.PAUSED_RENDER_HZ
        delay = self._last_paused_render + interval - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        self._last_paused_render = time.monotonic()
        self._sim.render()

    def _report_performance(self) -> None:
        """Print one compact performance sample per wall-clock second."""
        now = time.monotonic()
        elapsed = now - self._last_perf_report
        if elapsed < 1.0:
            return
        physics_hz = (self.tick - self._last_perf_tick) / elapsed
        render_fps = physics_hz / self.profile.render_interval if self._is_rendering else 0.0
        ticks = self.tick - self._last_perf_tick
        step_ms = 1000.0 * self._perf_step_seconds / ticks if ticks else 0.0
        render_ms = (
            1000.0 * self._perf_render_seconds / self._perf_render_count
            if self._perf_render_count
            else 0.0
        )
        print(
            f"[isaac-g1] physics={physics_hz:.1f} Hz  render={render_fps:.1f} FPS  "
            f"RTF={physics_hz * self.profile.physics_dt:.2f}x  "
            f"step={step_ms:.2f} ms  render_call={render_ms:.2f} ms  device={self._device}",
            flush=True,
        )
        self._last_perf_report = now
        self._last_perf_tick = self.tick
        self._perf_step_seconds = 0.0
        self._perf_render_seconds = 0.0
        self._perf_render_count = 0

    def run(self) -> dict[str, Any]:
        self.start()
        self.play()
        self._run_started = time.monotonic()
        self._last_perf_report = self._run_started
        try:
            while self.app.is_running() and time.monotonic() - self._run_started < self.duration:
                if self._pending_ui_reset:
                    self._pending_ui_reset = False
                    # Match the official Isaac Lab reset order: write state
                    # while playing, publish it through the sole physics step,
                    # then pause on the freshly rendered initial pose.
                    self.reset_episode(resume=True)
                    self._step_physics()
                    self.pause()
                    print('{"event":"isaac_g1_ui_reset","timeline":"paused"}', flush=True)
                    continue
                if self._timeline.is_playing():
                    if self.state is not TimelineState.PLAYING:
                        self._transition(TimelineState.PLAYING)
                    iteration_started = time.monotonic()
                    self._step_physics()
                    self._wall_physics_elapsed += time.monotonic() - iteration_started
                    self._report_performance()
                else:
                    if self.state is not TimelineState.PAUSED:
                        self._transition(TimelineState.PAUSED)
                    self._render_paused()
        except Exception as exc:
            self._transition(TimelineState.FAILED, error=str(exc))
            raise
        finally:
            self.stop()
        self._transition(TimelineState.STOPPED)
        summary = (
            self._passive_fall_acceptance()
            if self.test_mode == "passive-fall"
            else self._runtime_summary()
        )
        return summary

    def _accounting(self) -> dict[str, Any]:
        simulated_time = self.tick * self.profile.physics_dt
        return {
            "physics_ticks": self.tick,
            "real_time_factor": (
                simulated_time / self._wall_physics_elapsed if self._wall_physics_elapsed else 0.0
            ),
            "device": self._device,
            "render_interval": self.profile.render_interval,
        }

    def _runtime_summary(self) -> dict[str, Any]:
        return {
            "result": "COMPLETED",
            "profile_id": self.profile.profile_id,
            **self._accounting(),
        }

    def _passive_fall_acceptance(self) -> dict[str, Any]:
        thresholds = AcceptanceThresholds()
        initial_z = self._root_z[0] if self._root_z else None
        minimum_z = min(self._root_z) if self._root_z else None
        drop = 0.0 if initial_z is None or minimum_z is None else initial_z - minimum_z
        up_change = 0.0 if not self._root_up else abs(self._root_up[-1] - self._root_up[0])
        fell = drop >= thresholds.minimum_drop_m or up_change >= thresholds.minimum_up_change
        checks = {
            "controller_absent": True,
            "physics_tick_monotonic": self.tick > 1,
            "robot_fell": fell,
            "floor_bounded": bool(self._root_z and minimum_z >= thresholds.minimum_root_z),
            "head_camera_shape": self._camera_shape
            == [self.profile.camera.height, self.profile.camera.width, 3],
            "head_camera_flowing": self._camera_frames >= 2
            and self._camera_changed_frames >= 1,
        }
        result = "PASS" if all(checks.values()) else "FAIL"
        return {
            "result": result,
            "profile_id": self.profile.profile_id,
            "checks": checks,
            **self._accounting(),
            "initial_root_z": initial_z,
            "minimum_root_z": minimum_z,
            "root_drop_m": drop,
            "pelvis_up_change": up_change,
            "head_camera_frames": self._camera_frames,
            "head_camera_changed_frames": self._camera_changed_frames,
            "head_camera_shape": self._camera_shape,
        }
