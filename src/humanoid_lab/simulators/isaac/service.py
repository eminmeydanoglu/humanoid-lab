"""Isaac Lab G1 simulator: the sole owner of the physics loop.

A controller may feed it joint commands, but the loop itself decides what is
applied on every tick.  With no controller, or with a controller whose commands
have gone stale, the robot is driven passively and falls.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

from ...contracts.commands import CommandError, JointCommand, JointLayout, resolve_layouts
from ...controllers.base import ControllerSource, RobotStateSample
from ...controllers.factory import build_controller
from .contracts import RunProfile, TimelineState, quaternion_up_z
from .maths import quat_to_rotation_vector

PASSIVE = "passive"
CONTROLLED = "controlled"
HAND_FALLBACKS = ("passive",)


def _scalar_or_none(value: Any) -> float | None:
    """Largest finite value of a gain/limit that may be a tensor or None."""
    if value is None:
        return None
    try:
        if hasattr(value, "numel") and value.numel() > 1:
            return float(value.max())
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class AcceptanceThresholds:
    minimum_drop_m: float = 0.12
    minimum_up_change: float = 0.15
    minimum_root_z: float = -0.50
    minimum_tracking_progress: float = 0.5
    minimum_responsive_joints: int = 3
    minimum_responsive_travel_rad: float = 0.15
    # A standing G1 root sits near 0.75 m; a fallen one is well below 0.5 m.
    standing_minimum_root_z: float = 0.5


class SimulatorService:
    """Own the one scene, one reset, and one physics loop."""

    PAUSED_RENDER_HZ = 30.0
    PHYSX_NUM_THREADS = 4
    # A setpoint change below this size is the deployment holding a pose, not
    # the controller moving; above it, control has taken over.
    SUPPORT_STATIC_EPSILON_RAD = 1e-3
    # How long the controller must hold a constant pose before the support is
    # considered armed for release (the deployment's wait phase).
    SUPPORT_ARM_SECONDS = 0.5
    # Roughly one command trace sample per second at 200 Hz.
    TRACE_EVERY_TICKS = 200

    def __init__(
        self,
        profile: RunProfile,
        simulation_app: Any,
        *,
        duration: float | None,
        show_ui: bool,
        show_head_camera: bool,
        test_mode: str | None,
        controller_provider: str | None = None,
    ) -> None:
        self.profile = profile
        self.app = simulation_app
        self.duration = duration
        self.show_ui = show_ui
        self.show_head_camera = show_head_camera
        self.test_mode = test_mode
        self.controller_provider = controller_provider
        self.state = TimelineState.STARTING
        self.tick = 0
        self.episode_id = 0
        self.error: str | None = None
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
        self._controller: ControllerSource | None = None
        self._controller_config: Mapping[str, Any] = profile.controller or {}
        self._body_layout: JointLayout | None = None
        self._hand_layouts: dict[str, JointLayout | None] = {"left": None, "right": None}
        self._body_effort_limits: tuple[float, ...] | None = None
        self._hand_effort_limits: dict[str, tuple[float, ...] | None] = {"left": None, "right": None}
        self._hand_fallback = str(self._controller_config.get("hand_fallback", PASSIVE))
        self._mass_alignment: dict[str, Any] | None = None
        self._joint_dynamics_alignment: dict[str, Any] | None = None
        self._initial_joint_pos: Any = None
        self._support_active = False
        self._support_armed = False
        self._support_started_tick: int | None = None
        self._support_release_tick: int | None = None
        self._support_release_reason: str | None = None
        self._support_static_ticks = 0
        self._support_previous_q: tuple[float, ...] | None = None
        self._band_body_id = 0
        self._control_mode = PASSIVE
        self._passive_groups: set[str] = set()
        self._effort_target: Any = None
        self._applied_ticks = 0
        self._stale_ticks = 0
        self._last_applied_sequence: int | None = None
        self._last_sequence_seen: int | None = None
        self._sequence_regressions = 0
        self._rejected_commands = 0
        self._last_rejection: str | None = None
        self._tracking_progress = 0.0
        self._responsive_joints = 0
        self._body_tracking_error_max = 0.0
        self._body_tracking_error_sum = 0.0
        self._body_tracking_samples = 0
        self._command_trace: list[dict[str, Any]] = []
        self._pacing_overruns = 0
        self._hold_initial_q: Any = None
        self._hold_target_q: Any = None
        self._last_controlled_tick: int | None = None
        self._passive_since_tick: int | None = None
        self._controlled_trace: list[bool] = []
        self._effort_limit_checks: dict[str, Any] = {}

    def _transition(self, state: TimelineState, error: str | None = None) -> None:
        self.state = state
        if error is not None:
            self.error = error

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
        self._resolve_initial_pose()
        # The hard reset above wrote the asset's own defaults; the profile's
        # declared start-up pose has to replace them before the first step.
        self._write_initial_state_to_sim()
        self._start_controller()
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
        self._resolve_initial_pose()
        self.episode_id += 1
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
            self._initial_joint_pos, self._robot.data.default_joint_vel.clone().zero_()
        )
        return default_root

    def _resolve_initial_pose(self) -> None:
        """Apply the profile's declared start-up pose, if it names one.

        Spawning from the asset defaults leaves the robot in a pose that does
        not match the ground: a controller would then have to catch a falling
        robot rather than take over a standing one. A declared pose puts the
        robot exactly where the controller expects to find it, and nothing is
        held in place afterwards -- an unpowered robot still falls.
        """
        import torch

        self._initial_joint_pos = self._robot.data.default_joint_pos.clone()
        if self.profile.initial_pose is None:
            return
        from ...controllers.sonic import named_pose

        pose, root_height = named_pose(self.profile.initial_pose)
        names = list(self._robot.joint_names)
        unknown = sorted(set(pose) - set(names))
        if unknown:
            raise ContractError(f"initial pose names joints this asset does not have: {unknown}")
        values = self._initial_joint_pos.clone()
        for name, angle in pose.items():
            values[0, names.index(name)] = float(angle)
        self._initial_joint_pos = values
        # The pose and the height it was authored for travel together: the same
        # pose dropped from a different height is a different, unsupported state.
        robot_spec = self.profile.robot
        if abs(robot_spec.initial_position_m[2] - root_height) > 1e-3:
            raise ContractError(
                f"initial_pose {self.profile.initial_pose!r} is authored for root height "
                f"{root_height} m but the profile spawns at {robot_spec.initial_position_m[2]} m"
            )
        print(
            json.dumps(
                {
                    "event": "isaac_g1_initial_pose",
                    "pose": self.profile.initial_pose,
                    "root_height_m": root_height,
                    "joints": len(pose),
                }
            ),
            flush=True,
        )

    def _request_reset_from_ui(self) -> None:
        """Queue reset work for the simulation-loop boundary."""
        if self._control_mode == CONTROLLED:
            # A reset teleports the articulation while the controller still
            # drives it from its own state history. Refusing the request is the
            # only outcome that cannot mix two episodes.
            print(
                '{"event":"isaac_g1_ui_reset_refused","reason":"controller_active"}',
                flush=True,
            )
            return
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
            if self.test_mode is not None or self.show_head_camera
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
        self._effort_target = torch.zeros((1, len(names)), device=self._robot.device)
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

    # ----------------------------------------------------------------- support

    def _start_support(self) -> None:
        """Declare the start-up support the profile asks for, if any."""
        self._support_active = self.profile.support is not None and self._controller is not None
        if not self._support_active:
            return
        assert self.profile.support is not None
        # The reference MuJoCo band has no autonomous timeout. Our safety
        # timeout starts only once a controller command exists; charging it
        # while SONIC is still loading can drop an otherwise untouched robot.
        self._support_started_tick = None
        self._support_release_tick: int | None = None
        self._support_release_reason: str | None = None
        self._support_static_ticks = 0
        self._support_previous_q: tuple[float, ...] | None = None
        self._support_armed = False
        print(
            json.dumps(
                {
                    "event": "isaac_g1_support_engaged",
                    "kind": self.profile.support.kind,
                    "release": self.profile.support.release,
                    "max_seconds": self.profile.support.max_seconds,
                }
            ),
            flush=True,
        )

    def _update_support(self, command: JointCommand | None) -> None:
        """Release the start-up support at the moment the controller takes over.

        The reference simulator releases its band by operator keypress once the
        controller is running. The same moment is visible here without a control
        channel: the deployment holds a constant pose while waiting, and starts
        moving its setpoints the instant control begins.
        """
        support = self.profile.support
        if support is None or not self._support_active or self._support_release_tick is not None:
            return
        if command is None:
            return
        if self._support_started_tick is None:
            self._support_started_tick = self.tick
        elapsed = (self.tick - self._support_started_tick) * self.profile.physics_dt
        if elapsed >= support.max_seconds:
            self._release_support("timeout")
            return
        q = command.q
        if self._support_previous_q is not None:
            delta = max(abs(a - b) for a, b in zip(q, self._support_previous_q))
            if delta > self.SUPPORT_STATIC_EPSILON_RAD:
                self._support_static_ticks = 0
                if self._support_armed:
                    self._release_support("controller_motion_after_hold")
            else:
                self._support_static_ticks += 1
                if self._support_static_ticks * self.profile.physics_dt >= self.SUPPORT_ARM_SECONDS:
                    self._support_armed = True
        self._support_previous_q = tuple(q)

    def _release_support(self, reason: str) -> None:
        self._support_release_tick = self.tick
        self._support_release_reason = reason
        print(
            json.dumps(
                {
                    "event": "isaac_g1_support_released",
                    "reason": reason,
                    "physics_tick": self.tick,
                    "simulated_time_s": round(self.tick * self.profile.physics_dt, 3),
                }
            ),
            flush=True,
        )

    def _apply_support(self) -> None:
        """Hold the pelvis with the reference loop's spring band, while engaged."""
        import torch
        from isaaclab.utils.math import quat_rotate_inverse

        support = self.profile.support
        if support is None or not self._support_active:
            return
        body_ids = [self._band_body_id]
        if self._support_release_tick is not None:
            zeros = torch.zeros((1, 1, 3), device=self._robot.device)
            self._robot.set_external_force_and_torque(zeros, zeros, body_ids=body_ids)
            return

        position = self._robot.data.body_pos_w[0, self._band_body_id]
        point = torch.tensor(support.point_m, device=self._robot.device)
        offset = point - position
        linear_velocity = self._robot.data.body_lin_vel_w[0, self._band_body_id]
        force = support.linear_stiffness * offset - support.linear_damping * linear_velocity

        quaternion = self._robot.data.body_quat_w[0, self._band_body_id].unsqueeze(0)
        rotation_vector = quat_to_rotation_vector(quaternion)[0]
        angular_velocity = quat_rotate_inverse(
            quaternion, self._robot.data.body_ang_vel_w[0, self._band_body_id].unsqueeze(0)
        )[0]
        torque = -support.angular_stiffness * rotation_vector - support.angular_damping * angular_velocity

        self._robot.set_external_force_and_torque(
            force.reshape(1, 1, 3), torque.reshape(1, 1, 3), body_ids=body_ids
        )

    def _resolve_band_body_id(self) -> None:
        support = self.profile.support
        assert support is not None
        name = str(self._controller_config.get("support_link", "pelvis"))
        names = list(self._robot.body_names)
        if name not in names:
            raise CommandError(f"controller.support_link {name!r} is not a body of this asset")
        self._band_body_id = names.index(name)

    def _support_status(self) -> dict[str, Any] | None:
        if self.profile.support is None:
            return None
        return {
            "kind": self.profile.support.kind,
            "active": self._support_active and self._support_release_tick is None,
            "armed": self._support_armed,
            "release_reason": self._support_release_reason,
            "release_tick": self._support_release_tick,
            "timeout_started_tick": self._support_started_tick,
            "max_seconds": self.profile.support.max_seconds,
        }

    # ------------------------------------------------------------------ control

    def _start_controller(self) -> None:
        """Resolve the controller's joint order against the asset and start it."""
        import torch

        config = dict(self._controller_config)
        provider = str(config.get("provider", "none"))
        if self.controller_provider is not None:
            provider = self.controller_provider
        if provider == "none":
            return
        if self._hand_fallback not in HAND_FALLBACKS:
            raise CommandError(
                f"controller.hand_fallback must be one of {HAND_FALLBACKS}, got {self._hand_fallback!r}"
            )
        config["provider"] = provider
        controller, interface = build_controller(
            config,
            provider_override=provider,
            physics_dt=self.profile.physics_dt,
            ttl_s=self.profile.command_ttl_s,
        )
        assert controller is not None and interface is not None

        available = list(self._robot.joint_names)
        bind_hands = bool(interface.hand_kind) and interface.hand_kind == self.profile.robot.hand.kind
        body, left, right = resolve_layouts(
            available_joints=available,
            body_names=interface.body_joint_names,
            left_hand_names=interface.left_hand_joint_names if bind_hands else (),
            right_hand_names=interface.right_hand_joint_names if bind_hands else (),
        )
        if set(body.indices) != set(self._body_ids):
            raise CommandError(
                "the controller's body joints do not match the body joints the profile declares"
            )
        if bind_hands and set(left.indices) | set(right.indices) != set(self._hand_ids):
            raise CommandError(
                "the controller's hand joints do not match the hand joints the profile declares"
            )

        self._body_layout = body
        self._hand_layouts = {"left": left if bind_hands else None, "right": right if bind_hands else None}
        self._body_effort_limits = tuple(interface.body_effort_limits_nm)
        self._hand_effort_limits = {
            "left": tuple(interface.left_hand_effort_limits_nm) if bind_hands else None,
            "right": tuple(interface.right_hand_effort_limits_nm) if bind_hands else None,
        }
        self._torso_body_id = self._resolve_torso_body_id()
        self._validate_effort_limits(interface)
        self._body_index = torch.tensor(body.indices, device=self._robot.device, dtype=torch.long)
        self._body_limit_tensor = torch.tensor(
            interface.body_effort_limits_nm, device=self._robot.device, dtype=torch.float32
        )
        self._controller = controller
        self._align_joint_dynamics()
        self._align_body_masses()
        self._report_actuators()
        if self.profile.support is not None:
            self._resolve_band_body_id()
            self._start_support()
        print(
            json.dumps(
                {
                    "event": "isaac_g1_controller",
                    "provider": provider,
                    "body_joints": len(body.names),
                    "hand_binding": self.profile.robot.hand.kind if bind_hands else "passive",
                    "hand_reason": None if bind_hands else "provider hand vocabulary is not this hand",
                    "command_ttl_s": self.profile.command_ttl_s,
                    **interface.notes,
                }
            ),
            flush=True,
        )

    def _align_joint_dynamics(self) -> None:
        """Match the passive joint dynamics of the compiled SONIC MJCF."""
        mode = self._controller_config.get("joint_dynamics_alignment")
        if mode is None:
            return
        if mode != "sonic_mujoco":
            raise CommandError(
                f"controller.joint_dynamics_alignment must be 'sonic_mujoco', got {mode!r}"
            )

        import torch

        from ...controllers.sonic import (
            MUJOCO_BODY_FRICTION_LOSS_NM,
            MUJOCO_HAND_FRICTION_LOSS_NM,
            MUJOCO_JOINT_ARMATURE,
            MUJOCO_JOINT_VISCOUS_FRICTION,
        )

        names = list(self._robot.joint_names)
        friction_by_name = dict(zip(self._body_layout.names, MUJOCO_BODY_FRICTION_LOSS_NM))
        for side in ("left", "right"):
            layout = self._hand_layouts[side]
            if layout is not None:
                friction_by_name.update(zip(layout.names, MUJOCO_HAND_FRICTION_LOSS_NM))
        missing = sorted(set(names) - set(friction_by_name))
        if missing:
            raise CommandError(f"MuJoCo joint dynamics do not cover asset joints: {missing}")

        joint_ids = list(range(len(names)))
        armature = torch.full(
            (1, len(names)), MUJOCO_JOINT_ARMATURE, device=self._robot.device
        )
        coulomb = torch.tensor(
            [[friction_by_name[name] for name in names]],
            device=self._robot.device,
            dtype=torch.float32,
        )
        viscous = torch.full(
            (1, len(names)), MUJOCO_JOINT_VISCOUS_FRICTION, device=self._robot.device
        )
        before = self._robot.data.joint_armature[0].clone()
        self._robot.write_joint_armature_to_sim(armature, joint_ids=joint_ids)
        self._robot.write_joint_friction_coefficient_to_sim(
            coulomb,
            joint_dynamic_friction_coeff=coulomb,
            joint_viscous_friction_coeff=viscous,
            joint_ids=joint_ids,
        )
        self._robot.data.default_joint_armature = self._robot.data.joint_armature.clone()
        self._robot.data.default_joint_friction_coeff = (
            self._robot.data.joint_friction_coeff.clone()
        )
        self._robot.data.default_joint_dynamic_friction_coeff = (
            self._robot.data.joint_dynamic_friction_coeff.clone()
        )
        self._robot.data.default_joint_viscous_friction_coeff = (
            self._robot.data.joint_viscous_friction_coeff.clone()
        )
        self._joint_dynamics_alignment = {
            "mode": mode,
            "joints": len(names),
            "armature_before_min": round(float(before.min()), 6),
            "armature_before_max": round(float(before.max()), 6),
            "armature_after": MUJOCO_JOINT_ARMATURE,
            "viscous_friction_after": MUJOCO_JOINT_VISCOUS_FRICTION,
            "coulomb_friction_min": min(friction_by_name.values()),
            "coulomb_friction_max": max(friction_by_name.values()),
        }
        print(
            json.dumps({"event": "isaac_g1_joint_dynamics_alignment", **self._joint_dynamics_alignment}),
            flush=True,
        )

    def _align_body_masses(self) -> None:
        """Give the asset the masses of the model the controller was tuned for.

        The profile declares this explicitly because it changes the robot's
        physics. The values come from the pinned MuJoCo model itself, so the
        two sides cannot drift apart by someone retyping a table.
        """
        mode = self._controller_config.get("mass_alignment")
        if mode is None:
            return
        if mode != "sonic_mujoco":
            raise CommandError(f"controller.mass_alignment must be 'sonic_mujoco', got {mode!r}")

        from .masses import (
            DEFAULT_MUJOCO_MODEL,
            WELDED_BODY_MASS_KG,
            load_mujoco_body_masses,
            plan_alignment,
        )

        path = str(self._controller_config.get("mujoco_model_path", DEFAULT_MUJOCO_MODEL))
        mujoco_masses = load_mujoco_body_masses(path)
        body_names = list(self._robot.body_names)
        indices, values, welded = plan_alignment(body_names, mujoco_masses)

        import torch

        current = self._robot.root_physx_view.get_masses().clone()
        before = float(current.sum())
        for index, value in zip(indices, values):
            current[:, index] = float(value)
        self._robot.root_physx_view.set_masses(current, torch.arange(current.shape[0]))
        after = float(current.sum())
        expected = sum(mujoco_masses[name] for name in body_names if name in mujoco_masses)
        self._robot.data.default_mass = current.clone()
        self._mass_alignment = {
            "mode": mode,
            "model": path,
            "bodies_from_model": len(indices) - len(welded),
            "bodies_welded_in_model": len(welded),
            "welded_names": welded,
            "total_mass_before_kg": round(before, 4),
            "total_mass_after_kg": round(after, 4),
            "model_total_kg": round(expected + len(welded) * WELDED_BODY_MASS_KG, 4),
        }
        print(json.dumps({"event": "isaac_g1_mass_alignment", **self._mass_alignment}), flush=True)

    def _report_actuators(self) -> None:
        """Report the actuator groups' model and effort limits once at start-up.

        The simulator drives joints through the effort path, so whether that
        path is implicit (PhysX drive) or explicit (torque) and what limit
        applies are properties worth stating rather than assuming.
        """
        groups = []
        for name, actuator in self._robot.actuators.items():
            groups.append(
                {
                    "name": name,
                    "model": type(actuator).__name__,
                    "joints": len(actuator.joint_indices),
                    "effort_limit_max": _scalar_or_none(getattr(actuator, "effort_limit", None)),
                    "effort_limit_sim_max": _scalar_or_none(getattr(actuator, "effort_limit_sim", None)),
                }
            )
        print(
            json.dumps(
                {
                    "event": "isaac_g1_actuators",
                    "has_implicit": self._robot._has_implicit_actuators,
                    "groups": groups,
                }
            ),
            flush=True,
        )

    def _resolve_torso_body_id(self) -> int:
        name = str(self._controller_config["torso_link"])
        names = list(self._robot.body_names)
        if name not in names:
            raise CommandError(f"controller.torso_link {name!r} is not a body of this asset")
        return names.index(name)

    def _validate_effort_limits(self, interface: Any) -> None:
        """The controller's limits must not exceed what the asset can deliver."""
        asset_limits = getattr(self._robot.data, "joint_effort_limits", None)
        if asset_limits is None:
            self._effort_limit_checks = {"result": "unverified", "reason": "asset exposes no effort limits"}
            return
        values = asset_limits[0].tolist()
        for group, indices, limits in (
            ("body", self._body_layout.indices, interface.body_effort_limits_nm),
            ("left_hand", self._hand_layouts["left"].indices if self._hand_layouts["left"] else (), self._hand_effort_limits["left"] or ()),
            ("right_hand", self._hand_layouts["right"].indices if self._hand_layouts["right"] else (), self._hand_effort_limits["right"] or ()),
        ):
            for index, limit in zip(indices, limits):
                if values[index] + 1e-6 < limit:
                    raise CommandError(
                        f"{group} joint {self._robot.joint_names[index]!r} accepts {values[index]} Nm "
                        f"but the controller commands up to {limit} Nm; the asset would silently clip"
                    )
        self._effort_limit_checks = {
            "result": "pass",
            "asset_effort_limits_nm": {
                name: values[index] for name, index in zip(self._body_layout.names, self._body_layout.indices)
            },
        }

    def _publish_robot_state(self) -> None:
        if self._controller is None:
            return
        from isaaclab.utils.math import quat_rotate_inverse

        robot = self._robot
        torso_index = self._torso_body_id
        # The torque this simulator staged for the tick, in the controller's own
        # joint order. The asset's actuator models report their own estimate,
        # which is not what the solver is given.
        body_tau = (
            tuple(self._effort_target[0].index_select(0, self._body_index).tolist())
            if self._effort_target is not None
            else (0.0,) * len(self._body_ids)
        )
        # The robot side of the Unitree protocol carries angular velocity in the
        # body frame: MuJoCo's free-joint velocity is body-frame, and the
        # official bridge publishes it unchanged. The torso IMU is likewise
        # published in the torso's own frame.
        torso_quat = robot.data.body_quat_w[0, torso_index]
        torso_ang_vel_b = quat_rotate_inverse(
            torso_quat.unsqueeze(0), robot.data.body_ang_vel_w[0, torso_index].unsqueeze(0)
        )[0]
        self._controller.publish_state(
            RobotStateSample(
                episode_id=self.episode_id,
                physics_tick=self.tick,
                simulated_time_s=self.tick * self.profile.physics_dt,
                root_position_m=tuple(robot.data.root_pos_w[0].tolist()),
                root_quaternion_wxyz=tuple(robot.data.root_quat_w[0].tolist()),
                root_linear_velocity_mps=tuple(robot.data.root_lin_vel_w[0].tolist()),
                root_angular_velocity_rps=tuple(robot.data.root_ang_vel_b[0].tolist()),
                # Body values are handed over in the controller's declared joint
                # order, not in the order the articulation happens to store them.
                body_q=tuple(robot.data.joint_pos[0].index_select(0, self._body_index).tolist()),
                body_dq=tuple(robot.data.joint_vel[0].index_select(0, self._body_index).tolist()),
                body_tau=body_tau,
                torso_quaternion_wxyz=tuple(torso_quat.tolist()),
                torso_angular_velocity_rps=tuple(torso_ang_vel_b.tolist()),
                left_hand_q=self._hand_positions("left"),
                left_hand_dq=self._hand_velocities("left"),
                right_hand_q=self._hand_positions("right"),
                right_hand_dq=self._hand_velocities("right"),
            )
        )

    def _hand_positions(self, side: str) -> tuple[float, ...]:
        indices = self._hand_layouts[side].indices if self._hand_layouts[side] else self._hand_ids
        return tuple(float(value) for value in self._robot.data.joint_pos[0, list(indices)].tolist())

    def _hand_velocities(self, side: str) -> tuple[float, ...]:
        indices = self._hand_layouts[side].indices if self._hand_layouts[side] else self._hand_ids
        return tuple(float(value) for value in self._robot.data.joint_vel[0, list(indices)].tolist())

    def _apply_control(self) -> None:
        """Apply the newest valid command, or fall back to passive drive.

        A controller is not trusted with the loop's life: a command that cannot
        be decoded, is not finite, or does not match the declared joint order is
        counted and treated exactly like a missing command, so the robot goes
        passive instead of the process dying.
        """
        if self._controller is None:
            return
        command = None
        try:
            command = self._controller.poll(self.tick)
            valid = (
                command is not None
                and command.episode_id == self.episode_id
                and command.is_valid_at(self.tick)
            )
            if valid and self._last_sequence_seen is not None and command.sequence < self._last_sequence_seen:
                self._sequence_regressions += 1
                valid = False
            if valid:
                self._apply_body_command(command.body)
                self._apply_hand_command(command.left_hand, "left")
                self._apply_hand_command(command.right_hand, "right")
        except CommandError as error:
            self._rejected_commands += 1
            self._last_rejection = f"{type(error).__name__}: {error}"
            command = None
            valid = False
        if not valid or command is None:
            self._stale_ticks += 1
            if self._control_mode == CONTROLLED:
                self._passive_since_tick = self.tick
            self._control_mode = PASSIVE
            self._go_passive()
            self._update_support(None)
            return

        self._last_sequence_seen = command.sequence
        self._update_support(command.body)
        self._control_mode = CONTROLLED
        self._applied_ticks += 1
        self._last_applied_sequence = command.sequence
        self._last_controlled_tick = self.tick
    def _apply_body_command(self, command: JointCommand) -> None:
        import torch

        layout = self._body_layout
        assert layout is not None
        command.validate(layout)
        measured_q = self._robot.data.joint_pos[0].index_select(0, self._body_index)
        measured_dq = self._robot.data.joint_vel[0].index_select(0, self._body_index)
        q_command = torch.tensor(command.q, device=self._robot.device, dtype=torch.float32)
        dq_command = torch.tensor(command.dq, device=self._robot.device, dtype=torch.float32)
        kp = torch.tensor(command.kp, device=self._robot.device, dtype=torch.float32)
        kd = torch.tensor(command.kd, device=self._robot.device, dtype=torch.float32)
        tau_ff = torch.tensor(command.tau, device=self._robot.device, dtype=torch.float32)
        # The parity formula the plan fixes, identical to the official MuJoCo loop.
        torque = tau_ff + kp * (q_command - measured_q) + kd * (dq_command - measured_dq)
        torque = torch.clamp(torque, -self._body_limit_tensor, self._body_limit_tensor)
        self._write_effort("body", layout.indices, torque)
        error = (q_command - measured_q).abs()
        self._body_tracking_error_max = max(self._body_tracking_error_max, float(error.max()))
        self._body_tracking_error_sum += float(error.mean())
        self._body_tracking_samples += 1
        if self._body_tracking_samples % self.TRACE_EVERY_TICKS == 0:
            knee = self._body_layout.names.index("left_knee_joint")
            # The joint with the largest load in the standing pose: its command,
            # its measured response, and the torque the simulator staged.
            self._command_trace.append(
                {
                    "tick": self.tick,
                    "sim_s": round(self.tick * self.profile.physics_dt, 2),
                    "error_rad": round(float(error.mean()), 4),
                    "knee_cmd": round(float(q_command[knee]), 3),
                    "knee_meas": round(float(measured_q[knee]), 3),
                    "knee_dq": round(float(measured_dq[knee]), 3),
                    "knee_tau": round(float(torque[knee]), 2),
                    "root_z": round(float(self._robot.data.root_pos_w[0, 2]), 3),
                    "root_x": round(float(self._robot.data.root_pos_w[0, 0]), 3),
                    "root_y": round(float(self._robot.data.root_pos_w[0, 1]), 3),
                }
            )
        if self.test_mode == "controlled-hold":
            self._record_tracking(measured_q, q_command)

    def _record_tracking(self, measured_q: Any, q_command: Any) -> None:
        if self._hold_target_q is None:
            self._hold_initial_q = measured_q.clone()
            self._hold_target_q = q_command.clone()
        initial = self._hold_initial_q
        target = self._hold_target_q
        travel = (target - initial).abs()
        responsive = travel >= AcceptanceThresholds().minimum_responsive_travel_rad
        count = int(responsive.sum())
        if count == 0:
            return
        remaining = (measured_q - target).abs()
        progress = ((travel - remaining) / travel)[responsive]
        self._responsive_joints = max(self._responsive_joints, count)
        self._tracking_progress = max(self._tracking_progress, float(progress.mean()))

    def _apply_hand_command(self, command: JointCommand | None, side: str) -> None:
        layout = self._hand_layouts[side]
        if layout is None:
            return
        if command is None:
            # The profile's declared policy for a missing hand command.
            self._write_passive(f"hand_{side}", layout.indices)
            return
        command.validate(layout)
        limits = self._hand_effort_limits[side]
        if limits is None:
            self._write_passive(f"hand_{side}", layout.indices)
            return
        measured_q = self._robot.data.joint_pos[0, list(layout.indices)]
        measured_dq = self._robot.data.joint_vel[0, list(layout.indices)]
        values = []
        for position, (q_target, dq_target, tau_ff, kp, kd, limit) in enumerate(
            zip(command.q, command.dq, command.tau, command.kp, command.kd, limits)
        ):
            torque = tau_ff + kp * (q_target - float(measured_q[position])) + kd * (
                dq_target - float(measured_dq[position])
            )
            values.append(max(-limit, min(limit, torque)))
        self._write_effort(f"hand_{side}", layout.indices, values)

    def _write_effort(self, group: str, indices: Any, values: Any) -> None:
        """Stage joint torques for this tick's single write to PhysX."""
        import torch

        tensor = (
            values
            if isinstance(values, torch.Tensor)
            else torch.tensor(list(values), device=self._robot.device, dtype=torch.float32)
        )
        self._effort_target[0, list(indices)] = tensor
        self._passive_groups.discard(group)

    def _write_passive(self, group: str, indices: Any) -> None:
        if group in self._passive_groups:
            return
        self._passive_groups.add(group)
        self._effort_target[0, list(indices)] = 0.0

    def _flush_effort(self) -> None:
        """Hand this tick's torques to the physics engine.

        This writes the actuation forces directly rather than going through the
        asset's actuator models. Those models (the pinned G1 asset uses explicit
        DCMotor models on the legs and implicit ones elsewhere) compute their
        own effort from their own gains and discard a supplied feed-forward
        torque, which is not the contract this simulator implements: the
        controller's command is the torque. Writing here, after the scene has
        written its own targets, makes the controller's torque the one that
        reaches the solver, clamped only by the controller's own limits.
        """
        self._robot.root_physx_view.set_dof_actuation_forces(
            self._effort_target, self._robot._ALL_INDICES
        )

    def _go_passive(self) -> None:
        self._write_passive("body", self._body_ids)
        for side in ("left", "right"):
            layout = self._hand_layouts[side]
            self._write_passive(f"hand_{side}", layout.indices if layout else self._hand_ids)

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
        self._apply_control()
        self._apply_support()
        self._scene.write_data_to_sim()
        self._flush_effort()
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
        self._publish_robot_state()
        self._perf_step_seconds += time.perf_counter() - step_started
        if rendered and self.test_mode is not None:
            self._consume_head_camera_frame()
        if self.test_mode is not None:
            root_z = float(self._robot.data.root_pos_w[0, 2])
            self._root_z.append(root_z)
            root_quat = tuple(float(value) for value in self._robot.data.root_quat_w[0].tolist())
            self._root_up.append(quaternion_up_z(root_quat))
            if self.test_mode == "controller-hold":
                self._controlled_trace.append(self._control_mode == CONTROLLED)

    def _render_paused(self) -> None:
        """Refresh UI and viewport at a bounded rate while physics is paused."""
        interval = 1.0 / self.PAUSED_RENDER_HZ
        delay = self._last_paused_render + interval - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        self._last_paused_render = time.monotonic()
        self._sim.render()

    def _pace_physics(self, iteration_started: float) -> None:
        """Hold the declared physics rate once a controller is attached.

        A controller's rates are wall-clock rates over DDS, so an unattended
        free-running loop would silently change their meaning and starve the
        threads that carry them. With no controller the loop stays unpaced,
        exactly as it was before controllers existed.
        """
        delay = self.profile.physics_dt - (time.monotonic() - iteration_started)
        if delay > 0.0:
            time.sleep(delay)
        else:
            self._pacing_overruns += 1

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
            while self.app.is_running() and (
                self.duration is None
                or time.monotonic() - self._run_started < self.duration
            ):
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
                    if self._controller is not None:
                        self._pace_physics(iteration_started)
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
            if self._controller is not None:
                self._controller.close()
        self._transition(TimelineState.STOPPED)
        summary = (
            self._passive_fall_acceptance()
            if self.test_mode == "passive-fall"
            else self._controlled_hold_acceptance()
            if self.test_mode == "controlled-hold"
            else self._controller_hold_acceptance()
            if self.test_mode == "controller-hold"
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
            "physics_pacing": "realtime" if self._controller is not None else "free",
            "pacing_overruns": self._pacing_overruns,
        }

    def _runtime_summary(self) -> dict[str, Any]:
        return {
            "result": "COMPLETED",
            "profile_id": self.profile.profile_id,
            **self._accounting(),
            "controller": self._controller_status(),
        }

    def _controller_status(self) -> dict[str, Any] | None:
        if self._controller is None:
            return None
        return {
            **self._controller.status(),
            "control_mode": self._control_mode,
            "applied_ticks": self._applied_ticks,
            "stale_ticks": self._stale_ticks,
            "last_applied_sequence": self._last_applied_sequence,
            "sequence_regressions": self._sequence_regressions,
            "rejected_commands": self._rejected_commands,
            "last_rejection": self._last_rejection,
            "body_tracking_error_rad": {
                "max": self._body_tracking_error_max,
                "mean": (
                    self._body_tracking_error_sum / self._body_tracking_samples
                    if self._body_tracking_samples
                    else None
                ),
                "samples": self._body_tracking_samples,
            },
            "command_trace": self._command_trace[:40],
            "hand_binding": {
                side: ("bound" if self._hand_layouts[side] else "passive") for side in ("left", "right")
            },
            "hand_fallback": self._hand_fallback,
            "mass_alignment": self._mass_alignment,
            "joint_dynamics_alignment": self._joint_dynamics_alignment,
            "support": self._support_status(),
            "effort_limits": self._effort_limit_checks,
        }

    def _controller_hold_acceptance(self) -> dict[str, Any]:
        """Acceptance for a live controller: the robot is held, then drops.

        This is the visible gate of the whole exercise, stated as a measurement:
        while commands arrive the robot stays up, and once they stop it goes
        passive and falls.
        """
        thresholds = AcceptanceThresholds()
        controlled_z = [
            z for z, controlled in zip(self._root_z, self._controlled_trace) if controlled
        ]
        tail_z = [
            z for z, controlled in zip(self._root_z, self._controlled_trace) if not controlled
        ]
        minimum_controlled_z = min(controlled_z) if controlled_z else None
        minimum_tail_z = min(tail_z) if tail_z else None
        drop_after_control = (
            minimum_controlled_z - minimum_tail_z
            if minimum_controlled_z is not None and minimum_tail_z is not None
            else 0.0
        )
        checks = {
            "controller_present": self._controller is not None,
            "commands_applied": self._applied_ticks > 0,
            "sequence_monotonic": self._sequence_regressions == 0,
            "held_while_controlled": (
                minimum_controlled_z is not None
                and minimum_controlled_z >= thresholds.standing_minimum_root_z
            ),
            "passive_after_commands_stopped": (
                self._passive_since_tick is not None
                and self._last_controlled_tick is not None
                and self._passive_since_tick > self._last_controlled_tick
                and self._control_mode == PASSIVE
            ),
            "fell_after_commands_stopped": drop_after_control >= thresholds.minimum_drop_m,
            "physics_tick_monotonic": self.tick > 1,
        }
        result = "PASS" if all(checks.values()) else "FAIL"
        return {
            "result": result,
            "profile_id": self.profile.profile_id,
            "checks": checks,
            **self._accounting(),
            "controller": self._controller_status(),
            "controlled_ticks": sum(1 for value in self._controlled_trace if value),
            "command_trace": self._command_trace[:40],
            "root_z_trace": self._root_z_trace(summary=True),
            "minimum_root_z_while_controlled": minimum_controlled_z,
            "minimum_root_z_after_control": minimum_tail_z,
            "drop_after_control_m": drop_after_control,
            "last_controlled_tick": self._last_controlled_tick,
            "passive_since_tick": self._passive_since_tick,
            "head_camera_frames": self._camera_frames,
            "head_camera_changed_frames": self._camera_changed_frames,
            "head_camera_shape": self._camera_shape,
        }

    def _root_z_trace(self, *, summary: bool) -> dict[str, Any]:
        """A compact view of the root height over the run, for acceptance reading."""
        if not self._root_z:
            return {"samples": 0}
        stride = max(1, len(self._root_z) // 24)
        return {
            "samples": len(self._root_z),
            "interval_ticks": stride,
            "minimum": min(self._root_z),
            "series": [round(value, 4) for value in self._root_z[::stride]],
        }

    def _controlled_hold_acceptance(self) -> dict[str, Any]:
        """Acceptance for the command path: drive arrives, and expires."""
        thresholds = AcceptanceThresholds()
        initial_z = self._root_z[0] if self._root_z else None
        minimum_z = min(self._root_z) if self._root_z else None
        drop = 0.0 if initial_z is None or minimum_z is None else initial_z - minimum_z
        up_change = 0.0 if not self._root_up else abs(self._root_up[-1] - self._root_up[0])
        checks = {
            "controller_present": self._controller is not None,
            "commands_applied": self._applied_ticks > 0,
            "joints_tracked_command": self._tracking_progress
            >= thresholds.minimum_tracking_progress
            and self._responsive_joints >= thresholds.minimum_responsive_joints,
            "sequence_monotonic": self._sequence_regressions == 0,
            "passive_after_command_stop": (
                self._passive_since_tick is not None
                and self._last_controlled_tick is not None
                and self._passive_since_tick > self._last_controlled_tick
                and self._control_mode == PASSIVE
            ),
            "passive_drive_final": self._control_mode == PASSIVE,
            "physics_tick_monotonic": self.tick > 1,
        }
        result = "PASS" if all(checks.values()) else "FAIL"
        return {
            "result": result,
            "profile_id": self.profile.profile_id,
            "checks": checks,
            **self._accounting(),
            "controller": self._controller_status(),
            "tracking_progress": self._tracking_progress,
            "responsive_joints": self._responsive_joints,
            "last_controlled_tick": self._last_controlled_tick,
            "passive_since_tick": self._passive_since_tick,
            "root_drop_m": drop,
            "pelvis_up_change": up_change,
            "head_camera_frames": self._camera_frames,
            "head_camera_changed_frames": self._camera_changed_frames,
        }

    def _passive_fall_acceptance(self) -> dict[str, Any]:
        thresholds = AcceptanceThresholds()
        initial_z = self._root_z[0] if self._root_z else None
        minimum_z = min(self._root_z) if self._root_z else None
        drop = 0.0 if initial_z is None or minimum_z is None else initial_z - minimum_z
        up_change = 0.0 if not self._root_up else abs(self._root_up[-1] - self._root_up[0])
        fell = drop >= thresholds.minimum_drop_m or up_change >= thresholds.minimum_up_change
        checks = {
            "controller_absent": self._controller is None,
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
