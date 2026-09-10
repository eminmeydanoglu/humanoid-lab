#!/usr/bin/env python3
"""Run an Isaac Lab G1 with the pinned upstream SONIC controller as its brain.

The runner owns no DDS stack: Isaac Sim bundles its own DDS implementation, so
the Unitree topics are served by ``scripts/sonic-isaac-dds-bridge.py`` in the
SONIC environment, reached over the loopback :class:`StateLink`.

Structure mirrors the existing interactive runner split so the timeline
lifecycle and the per-step effort path stay directly testable:

* :func:`reset_simulation_paused` -- ``sim.reset()`` then pause, so PhysX tensor
  views exist while nothing advances.
* :func:`step_frame` -- steps physics only while the timeline is playing, and
  writes SONIC's ``tau`` instead of a position target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sonic_isaac_actuation import (  # noqa: E402
    ActuationMode,
    BodyActuation,
    LowCmdSample,
    RateMeter,
)
from sonic_isaac_contract import (  # noqa: E402
    BODY_JOINT_COUNT,
    SONIC_DEFAULT_BODY_ANGLES,
    sonic_default_pose_by_name,
    ContractError,
    JointLimits,
    PHYSICS_DT,
    PHYSICS_HZ,
    SONIC_BODY_JOINT_NAMES,
    build_joint_mapping,
    mapping_evidence_summary,
    validate_limits,
)
from sonic_isaac_ipc import (  # noqa: E402
    DEFAULT_IPC_PORT,
    HandStateFrame,
    LowCmdFrame,
    StateFrame,
    StateLink,
)

EXIT_OK = 0
EXIT_CONTRACT = 2


@dataclass(frozen=True)
class Profile:
    name: str
    isaac_lab_config: str
    hand_joint_count: int


PROFILES = {
    "g1-29dof": Profile("g1-29dof", "G1_29DOF_CFG", 0),
    "g1-inspire": Profile("g1-inspire", "G1_INSPIRE_FTP_CFG", 24),
}

#: The Inspire fingers are held by their own drive, never by SONIC body output.
HANDS_CONTROLLED_BY_SONIC = False


# --------------------------------------------------------------------------- #
# Timeline lifecycle and the per-frame step
# --------------------------------------------------------------------------- #


def physx_timestamp() -> float:
    """PhysX's own simulation clock.

    ``SimulationContext.current_time`` is only advanced by ``step()``, so it
    cannot prove that an ``app.update()`` frame did not step physics. PhysX's
    timestamp can.
    """
    try:
        import omni.physx

        return float(omni.physx.get_physx_interface().get_simulation_timestamp())
    except Exception:  # noqa: BLE001
        return float("nan")


def reset_simulation_paused(sim: object, timeline: object | None) -> None:
    """``sim.reset()`` first, then pause; the ordering matters.

    ``reset()`` initialises the PhysX tensor views and leaves the timeline
    playing.  Pausing afterwards keeps those views valid while nothing
    advances, so a later GUI Play can take over.
    """
    sim.reset()
    if timeline is not None:
        timeline.pause()
    if hasattr(sim, "pause"):
        # pause() only clears SimulationContext's own playing flag; the GUI
        # timeline stays authoritative for step_frame().
        sim.pause()


def step_frame(
    simulation_app: object,
    timeline: object,
    sim: object,
    scene: object,
    *,
    efforts: Sequence[float] | None = None,
    apply_effort=None,
) -> bool:
    """Advance, or hold, exactly one interactive frame.

    Returns True when physics advanced.  While paused the app is merely updated,
    which keeps the GUI responsive without changing the physics step count.
    """
    if not timeline.is_playing():
        simulation_app.update()
        return False
    if efforts is not None and apply_effort is not None:
        apply_effort(efforts)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    return True


def run_interactive_app(
    simulation_app: object,
    timeline: object,
    sim: object,
    scene: object,
    *,
    actuation: BodyActuation,
    link: StateLink,
    metrics: "Metrics",
    read_state,
    read_body_q_dq,
    apply_effort,
    duration_s: float | None = None,
    warmup_s: float = 0.0,
    settle_frames: int = 0,
    on_play=None,
    on_physics_step=None,
) -> int:
    """Main GUI loop; returns the number of physics steps taken."""
    physics_steps = 0
    pending_control_sample = None
    play_started_at: float | None = None
    measurement_started_at: float | None = None
    settle_remaining = 0

    while simulation_app.is_running():
        now = time.monotonic()
        link.accept()
        link.publish(read_state())

        # Timeline transitions are resolved before anything is counted, so the
        # measurement window's counters start clean on the frame it opens.
        playing = timeline.is_playing()
        if playing and play_started_at is None:
            play_started_at = now
            settle_remaining = max(0, int(settle_frames))
            print(json.dumps({"event": "timeline_play_started"}), flush=True)

        if playing and settle_remaining > 0:
            # Kit can advance physics by the whole paused interval on the
            # resume frame. Hold the reset pose until that backlog is spent;
            # nothing is measured and no command is applied meanwhile.
            simulation_app.update()
            if on_play is not None:
                on_play(now)
            settle_remaining -= 1
            metrics.observe(None)
            continue

        if playing and measurement_started_at is None and (now - play_started_at) >= warmup_s:
            if on_play is not None:
                on_play(now)
            measurement_started_at = now
            metrics.start_measurement(now)
            print(json.dumps({"event": "measurement_started", "warmup_s": warmup_s}), flush=True)

        command = link.take_command()
        if command is not None:
            outcome = actuation.submit(_to_sample(command, now))
            metrics.command_seen(command, now, accepted=outcome.accepted)
            if outcome.accepted:
                pending_control_sample = command

        step = None
        if playing:
            body_q, body_dq = read_body_q_dq()
            if pending_control_sample is not None and metrics.first_control_sample is None:
                metrics.record_first_control_sample(pending_control_sample, body_q, body_dq)
            step = actuation.step(now, body_q, body_dq)
            if step.mode is ActuationMode.CONTROLLED:
                metrics.controlled_steps += 1
            else:
                metrics.passive_steps += 1

        advanced = step_frame(
            simulation_app,
            timeline,
            sim,
            scene,
            efforts=None if step is None else step.efforts,
            apply_effort=apply_effort,
        )
        if advanced:
            physics_steps += 1
            if on_physics_step is not None:
                on_physics_step(physics_steps)
        metrics.observe(step)

        if (
            duration_s is not None
            and measurement_started_at is not None
            and (now - measurement_started_at) >= duration_s
        ):
            print(json.dumps({"event": "measurement_complete"}), flush=True)
            break

    return physics_steps


def _to_sample(command: LowCmdFrame, now: float) -> LowCmdSample:
    return LowCmdSample(
        sequence=command.sequence,
        received_at=now,
        q=command.q,
        dq=command.dq,
        tau=command.tau,
        kp=command.kp,
        kd=command.kd,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class Metrics:
    """Accumulates the quantities each acceptance gate reads."""

    def __init__(self, *, fall_drop_m: float = 0.40, pelvis_up_z_max: float = 0.50) -> None:
        self.fall_drop_m = float(fall_drop_m)
        self.pelvis_up_z_max = float(pelvis_up_z_max)
        self.measurement_started: float | None = None
        self.initial_root_pos: tuple[float, float, float] | None = None
        self.root_z_min = math.inf
        self.root_z_final: float | None = None
        self.max_abs_roll = 0.0
        self.max_abs_pitch = 0.0
        self.max_xy_drift = 0.0
        self.pelvis_up_z_min = math.inf
        self.root_xy_final: tuple[float, float] | None = None
        self.nan_observations = 0
        self.joint_limit_violations = 0
        self.controlled_steps = 0
        self.passive_steps = 0
        self.max_lowcmd_age_ms = 0.0
        self.rate = RateMeter()
        self.rejected_commands = 0
        self.accepted_commands = 0
        self._body_limits: Sequence[JointLimits] = ()
        self._body_ids: Sequence[int] = ()
        self._read_joint_pos = None
        self._read_root = None
        self._last_accepted_at: float | None = None
        self._last_quat: tuple[float, ...] | None = None
        self.root_yaw_deg = 0.0
        self.net_xy_m = 0.0
        self.max_speed_mps = 0.0
        self.trace: list[list[float]] = []
        self.first_control_sample: dict | None = None
        self._trace_every = 20  # 10 Hz at the 200 Hz physics rate
        self._steps = 0
        self.passive_since: float | None = None
        self.stop_to_passive_ms: float | None = None

    def bind(
        self,
        *,
        body_ids: Sequence[int],
        limits: Sequence[JointLimits],
        read_joint_pos,
        read_root,
    ) -> None:
        self._body_ids = tuple(body_ids)
        self._body_limits = tuple(limits)
        self._read_joint_pos = read_joint_pos
        self._read_root = read_root

    def start_measurement(self, now: float) -> None:
        self.measurement_started = now
        self.initial_root_pos = None
        self.root_z_min = math.inf
        self.root_z_final = None
        self.max_abs_roll = 0.0
        self.max_abs_pitch = 0.0
        self.max_xy_drift = 0.0
        self.pelvis_up_z_min = math.inf
        self.nan_observations = 0
        self.joint_limit_violations = 0
        self.max_lowcmd_age_ms = 0.0
        self.rate = RateMeter()
        self.rejected_commands = 0
        self.accepted_commands = 0
        self.root_yaw_deg = 0.0
        self.net_xy_m = 0.0
        self.max_speed_mps = 0.0
        self.trace = []
        self._steps = 0

    def record_first_control_sample(self, command: LowCmdFrame, isaac_q, isaac_dq) -> None:
        if self.first_control_sample is not None:
            return
        self.first_control_sample = {
            "names": list(SONIC_BODY_JOINT_NAMES),
            "lowcmd_q": [round(float(v), 5) for v in command.q],
            "lowcmd_dq": [round(float(v), 5) for v in command.dq],
            "lowcmd_tau": [round(float(v), 5) for v in command.tau],
            "lowcmd_kp": [round(float(v), 5) for v in command.kp],
            "lowcmd_kd": [round(float(v), 5) for v in command.kd],
            "isaac_q": [round(float(v), 5) for v in isaac_q],
            "isaac_dq": [round(float(v), 5) for v in isaac_dq],
            "q_error": [
                round(float(c) - float(a), 5) for c, a in zip(command.q, isaac_q)
            ],
        }

    def command_seen(self, command: LowCmdFrame, now: float, *, accepted: bool) -> None:
        if accepted:
            self.accepted_commands += 1
            self._last_accepted_at = now
            if self.measurement_started is not None:
                self.rate.mark(now)
            self.passive_since = None
            self.stop_to_passive_ms = None
        else:
            self.rejected_commands += 1

    def observe(self, step) -> None:
        if self.measurement_started is None or self._read_root is None:
            return
        root_pos, quat, lin_vel = self._read_root()
        if self.initial_root_pos is None:
            self.initial_root_pos = root_pos
        if any(not math.isfinite(value) for value in root_pos + quat):
            self.nan_observations += 1
            return
        self._last_quat = quat
        self.root_xy_final = (root_pos[0], root_pos[1])
        roll, pitch, yaw = quat_to_rpy(quat)
        self.root_yaw_deg = math.degrees(yaw)
        speed = math.hypot(lin_vel[0], lin_vel[1])
        self.max_speed_mps = max(self.max_speed_mps, speed)
        self.net_xy_m = math.hypot(
            root_pos[0] - self.initial_root_pos[0], root_pos[1] - self.initial_root_pos[1]
        )
        self._steps += 1
        if self._steps % self._trace_every == 0:
            self.trace.append(
                [
                    round(self._steps * PHYSICS_DT, 4),
                    round(root_pos[0], 5),
                    round(root_pos[1], 5),
                    round(speed, 5),
                    round(self.root_yaw_deg, 4),
                ]
            )
        self.root_z_min = min(self.root_z_min, root_pos[2])
        self.root_z_final = root_pos[2]
        roll, pitch, _yaw = quat_to_rpy(quat)
        self.max_abs_roll = max(self.max_abs_roll, abs(roll))
        self.max_abs_pitch = max(self.max_abs_pitch, abs(pitch))
        self.max_xy_drift = max(
            self.max_xy_drift,
            math.hypot(
                root_pos[0] - self.initial_root_pos[0],
                root_pos[1] - self.initial_root_pos[1],
            ),
        )
        self.pelvis_up_z_min = min(self.pelvis_up_z_min, up_z(quat))

        if self._read_joint_pos is not None:
            for value, limit in zip(self._read_joint_pos(), self._body_limits):
                if not math.isfinite(value):
                    self.nan_observations += 1
                elif value < limit.position_min or value > limit.position_max:
                    self.joint_limit_violations += 1

        if step is not None:
            if step.lowcmd_age_s is not None:
                self.max_lowcmd_age_ms = max(self.max_lowcmd_age_ms, step.lowcmd_age_s * 1000.0)
            if step.mode is ActuationMode.PASSIVE:
                if self.passive_since is None:
                    self.passive_since = time.monotonic()
                    if self._last_accepted_at is not None:
                        self.stop_to_passive_ms = (
                            self.passive_since - self._last_accepted_at
                        ) * 1000.0

    # The root reader and joint reader are injected by main() once the
    # articulation exists; until then observe() is a no-op.

    def fall_observed(self) -> bool:
        if self.initial_root_pos is None or self.root_z_final is None:
            return False
        drop = self.initial_root_pos[2] - self.root_z_final
        return drop >= self.fall_drop_m or self.pelvis_up_z_min <= self.pelvis_up_z_max

    def summary(self) -> dict:
        return {
            "root_z_initial": _finite(self.initial_root_pos[2] if self.initial_root_pos else None),
            "root_z_min": _finite(self.root_z_min),
            "root_z_final": _finite(self.root_z_final),
            "root_z_drop": _finite(
                None
                if self.initial_root_pos is None or self.root_z_final is None
                else self.initial_root_pos[2] - self.root_z_final
            ),
            "max_abs_roll_deg": math.degrees(self.max_abs_roll),
            "max_abs_pitch_deg": math.degrees(self.max_abs_pitch),
            "max_xy_drift_m": self.max_xy_drift,
            "pelvis_up_z_min": _finite(self.pelvis_up_z_min),
            "pelvis_up_z_final": _finite(up_z(self._last_quat)) if self._last_quat else None,
            "nan_observations": self.nan_observations,
            "joint_limit_violations": self.joint_limit_violations,
            "controlled_steps": self.controlled_steps,
            "passive_steps": self.passive_steps,
            "max_lowcmd_age_ms": self.max_lowcmd_age_ms,
            "lowcmd_hz": self.rate.mean_hz(),
            "lowcmd_max_gap_ms": self.rate.max_gap_ms(),
            "lowcmd_accepted": self.accepted_commands,
            "lowcmd_rejected": self.rejected_commands,
            "stop_to_passive_ms": self.stop_to_passive_ms,
            "root_yaw_deg": self.root_yaw_deg,
            "net_xy_m": self.net_xy_m,
            "max_speed_mps": self.max_speed_mps,
            "trace_columns": ["t_s", "x_m", "y_m", "speed_mps", "yaw_deg"],
            "trace": self.trace,
            "first_control_sample": self.first_control_sample,
        }


def _finite(value):
    if value is None:
        return None
    return None if value in (math.inf, -math.inf) else value


def quat_to_rpy(quat: Sequence[float]) -> tuple[float, float, float]:
    """Roll, pitch, yaw from a w, x, y, z quaternion."""
    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return (0.0, 0.0, 0.0)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


def up_z(quat: Sequence[float]) -> float:
    """World up-axis component of the body z axis (pelvis up-Z)."""
    _w, x, y, _z = (float(value) for value in quat)
    return 1.0 - 2.0 * (x * x + y * y)


# --------------------------------------------------------------------------- #
# Isaac adapters
# --------------------------------------------------------------------------- #


def build_scene(profile: Profile):
    """Interactive scene with a floor and the profile's G1, made free-based."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab_assets.robots.unitree import G1_29DOF_CFG, G1_INSPIRE_FTP_CFG

    template = G1_29DOF_CFG if profile.name == "g1-29dof" else G1_INSPIRE_FTP_CFG
    robot_cfg: ArticulationCfg = template.copy()
    # Both profiles stand on a free base with gravity on; the stock Inspire
    # config is a fixed-base manipulation setup, so both are pinned here.
    robot_cfg.spawn.articulation_props.fix_root_link = False
    robot_cfg.spawn.rigid_props.disable_gravity = False

    @configclass
    class SonicSceneCfg(InteractiveSceneCfg):
        # Isaac renders in "Stage Lights" mode, so the scene needs its own
        # lights: without them the viewport is black and only silhouettes show.
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.85, 0.87, 0.9)),
        )
        key_light = AssetBaseCfg(
            prim_path="/World/KeyLight",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 0.97, 0.92)),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(2.0, 2.0, 6.0), rot=(0.82, 0.35, 0.30, 0.34)
            ),
        )
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return SonicSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


def read_root_state(robot: object):
    """Root pose and linear velocity in world frame."""
    pos = tuple(float(value) for value in robot.data.root_pos_w[0].tolist())
    quat = tuple(float(value) for value in robot.data.root_quat_w[0].tolist())
    lin = tuple(float(value) for value in robot.data.root_lin_vel_w[0].tolist())
    return pos, quat, lin


def read_body_q_dq(robot: object, body_ids: Sequence[int]):
    def read():
        q = tuple(float(robot.data.joint_pos[0, index]) for index in body_ids)
        dq = tuple(float(robot.data.joint_vel[0, index]) for index in body_ids)
        return q, dq

    return read


def sonic_default_joint_target(robot: object, body_ids: Sequence[int]):
    """The SONIC default standing pose written into the asset's joint order.

    Matched by name, so it is correct regardless of how the asset orders DOFs.
    """
    import torch

    by_name = sonic_default_pose_by_name()
    values = [float(by_name[SONIC_BODY_JOINT_NAMES[slot]]) for slot in range(len(body_ids))]
    tensor = robot.data.default_joint_pos.clone()
    index = torch.tensor(list(body_ids), dtype=torch.long, device=tensor.device)
    tensor[:, index] = torch.tensor(values, dtype=tensor.dtype, device=tensor.device)
    return tensor


def body_joint_indices(robot: object) -> tuple[list[int], list[int]]:
    """Return (body ids, hand ids) using name-based matching, never order."""
    names = list(robot.joint_names)
    mapping = build_joint_mapping(SONIC_BODY_JOINT_NAMES, names)
    body_ids = list(mapping.sonic_to_isaac)
    hand_ids = [index for index, name in enumerate(names) if name not in SONIC_BODY_JOINT_NAMES]
    return body_ids, hand_ids


def collect_limits(robot: object, body_ids: Sequence[int]) -> list[JointLimits]:
    limits: list[JointLimits] = []
    for slot, index in enumerate(body_ids):
        limits.append(
            JointLimits(
                name=SONIC_BODY_JOINT_NAMES[slot],
                position_min=float(robot.data.joint_pos_limits[0, index, 0]),
                position_max=float(robot.data.joint_pos_limits[0, index, 1]),
                effort_max=abs(float(robot.data.joint_effort_limits[0, index])),
                velocity_max=abs(float(robot.data.joint_velocity_limits[0, index])),
            )
        )
    return limits


def zero_body_drives(robot: object, body_ids: Sequence[int]) -> None:
    """Zero Isaac's internal stiffness/damping so only SONIC's ``tau`` acts."""
    body_set = set(int(index) for index in body_ids)
    for actuator in robot.actuators.values():
        names = list(getattr(actuator, "joint_names", []))
        # Actuator gain tensors are shaped (num_envs, joints_in_group).
        stiffness = actuator.stiffness
        damping = actuator.damping
        for position, name in enumerate(names):
            try:
                joint_index = robot.joint_names.index(name)
            except ValueError:
                continue
            if joint_index in body_set:
                stiffness[:, position] = 0.0
                damping[:, position] = 0.0
    robot.write_joint_stiffness_to_sim(0.0, joint_ids=list(body_ids))
    robot.write_joint_damping_to_sim(0.0, joint_ids=list(body_ids))


def make_effort_writer(robot: object, body_ids: Sequence[int]):
    import torch

    index = torch.tensor(list(body_ids), dtype=torch.long, device=robot.device)

    def apply(efforts: Sequence[float]) -> None:
        tensor = torch.tensor(list(efforts), dtype=torch.float32, device=robot.device)
        robot.set_joint_effort_target(tensor, joint_ids=index)

    return apply


def read_hand_state(robot: object, hand_ids: Sequence[int]) -> HandStateFrame:
    q = tuple(float(robot.data.joint_pos[0, index]) for index in hand_ids)
    dq = tuple(float(robot.data.joint_vel[0, index]) for index in hand_ids)
    return HandStateFrame(len(hand_ids), q, dq)


# --------------------------------------------------------------------------- #
# Main viewport recording
# --------------------------------------------------------------------------- #


def viewport_diagnostics() -> dict:
    """Describe the viewport being recorded; empty when it cannot be inspected."""
    info: dict = {}
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        info["render_product_path"] = getattr(viewport, "render_product_path", None)
        info["resolution"] = list(getattr(viewport, "resolution", ()) or ())
        try:
            info["camera_path"] = viewport.get_active_camera()
        except Exception:  # noqa: BLE001
            info["camera_path"] = None
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def make_viewport_capture():
    """Return a callable producing the active viewport's RGB frame, or None."""
    try:
        import omni.replicator.core as rep
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        product = getattr(viewport, "render_product_path", None)
        if not product:
            return None
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])

        def capture():
            data = annotator.get_data()
            return data

        return capture
    except Exception:  # noqa: BLE001 - capture is best-effort, reported in evidence
        return None


class VideoRecorder:
    """Collects main-viewport frames and writes one uninterrupted mp4."""

    def __init__(self, path: Path | None, *, fps: int = 20, every: int = 4) -> None:
        self.path = path
        self.fps = fps
        self.every = every
        self.frames: list = []
        self.capture = make_viewport_capture() if path else None
        self.status = "disabled" if path is None else ("ready" if self.capture else "unavailable")

    def maybe_record(self, physics_step: int) -> None:
        if self.capture is None or physics_step % self.every:
            return
        try:
            frame = self.capture()
        except Exception:  # noqa: BLE001
            self.status = "failed"
            return
        if frame is None:
            return
        prepared = self._as_image(frame)
        if prepared is None:
            self.status = "unexpected_frame_shape"
            return
        self.frames.append(prepared)

    @staticmethod
    def _as_image(frame):
        """Normalise an annotator payload to a single HxWxC image, or None."""
        if frame is None:
            return None
        try:
            import numpy as np

            array = np.asarray(frame)
        except Exception:  # noqa: BLE001
            return None
        # Drop leading singleton/leading frame dimensions until 2D or 3D.
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 2:
            return array
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            return array
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            # Channel-first layout.
            return np.moveaxis(array, 0, -1)
        return None

    def write(self) -> dict:
        """Encode the captured frames to one continuous mp4.

        Frames are appended one at a time: imageio's list overload stacks them
        into a single 4-D array and then rejects it as "not 2D", which fails the
        recording silently.
        """
        if self.path is None:
            return {"status": "disabled", "frames": 0}
        if not self.frames:
            return {"status": self.status, "frames": 0, "path": str(self.path)}
        try:
            import imageio

            self.path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(self.path), fps=self.fps, codec="libx264", macro_block_size=None
            )
            try:
                for frame in self.frames:
                    writer.append_data(frame)
            finally:
                writer.close()
        except Exception as exc:  # noqa: BLE001
            return {"status": f"encode_failed: {exc}", "frames": len(self.frames)}
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return {"status": "encode_empty", "frames": len(self.frames)}
        return {
            "status": "written",
            "frames": len(self.frames),
            "path": str(self.path),
            "bytes": self.path.stat().st_size,
        }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=sorted(PROFILES), required=True)
    parser.add_argument("--input", choices=("keyboard", "f310", "acceptance"), default="keyboard")
    parser.add_argument("--ipc-port", type=int, default=DEFAULT_IPC_PORT)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--paused-hold", type=float, default=3.0)
    parser.add_argument(
        "--auto-play",
        action="store_true",
        help="press Play programmatically after --paused-hold (automated gates only)",
    )
    parser.add_argument(
        "--fall-test",
        action="store_true",
        help="Gate B: with no controller, confirm Play makes the robot fall",
    )
    parser.add_argument(
        "--play-trigger",
        type=Path,
        help="wait for this file to appear before pressing Play (automated gates)",
    )
    parser.add_argument(
        "--play-trigger-timeout",
        type=float,
        default=900.0,
        help="give up waiting for --play-trigger after this many seconds",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=20,
        help="frames to hold the reset pose after Play before measuring",
    )
    parser.add_argument("--dump-dir", type=Path)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True
    # The timeline widget must exist so a human can press Play.
    args.kit_args = " ".join(
        (args.kit_args, "--enable omni.anim.window.timeline --/exts/omni.anim.window.timeline/show=true")
    ).strip()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = PROFILES[args.robot]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    exit_code = EXIT_OK
    evidence: dict = {}
    try:
        exit_code, evidence = _run(args, profile, simulation_app)
    except ContractError as exc:
        _write_evidence(
            args.evidence_path,
            {"status": "contract_violation", "reason": str(exc), "robot_profile": profile.name},
        )
        print(json.dumps({"event": "contract_violation", "reason": str(exc)}), file=sys.stderr, flush=True)
        exit_code = EXIT_CONTRACT
    except Exception as exc:  # noqa: BLE001
        _write_evidence(
            args.evidence_path,
            {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "robot_profile": profile.name},
        )
        print(json.dumps({"event": "failed", "reason": str(exc)}), file=sys.stderr, flush=True)
        exit_code = EXIT_BROKEN
    finally:
        # Kit's shutdown routinely takes many minutes; the process is finished
        # either way, so flush and leave without waiting on the teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


def _write_evidence(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run(args: argparse.Namespace, profile: Profile, simulation_app) -> tuple[int, dict]:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from omni.timeline import get_timeline_interface

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device, use_fabric=True)
    )
    scene_cfg = build_scene(profile)
    scene = InteractiveScene(scene_cfg)
    # The Inspire asset keeps a disabled world joint above the pelvis; PhysX
    # cannot create the articulation until the root moves to the pelvis. This
    # has to happen after the prims exist and before sim.reset().
    from isaac_g1_free_base import configure_free_base_articulation

    free_base_overridden = configure_free_base_articulation()
    sim.reset()

    robot = scene["robot"]
    body_ids, hand_ids = body_joint_indices(robot)
    limits = collect_limits(robot, body_ids)
    validate_limits(limits)
    mapping = build_joint_mapping(SONIC_BODY_JOINT_NAMES, list(robot.joint_names))

    # Record where the articulation actually sits, so a height regression is
    # visible in the evidence rather than inferred from the thresholds.
    diagnostics = {
        "root_link": robot.data.body_names[0],
        "asset_dof_count": len(robot.joint_names),
        "body_dof_count": len(body_ids),
        "hand_dof_count": len(hand_ids),
        "body_ids": body_ids,
        "free_base_override_applied": free_base_overridden,
        "root_z_after_scene_reset": float(robot.data.root_pos_w[0][2]),
        "default_joint_pos": [float(v) for v in robot.data.default_joint_pos[0].tolist()],
    }

    timeline = get_timeline_interface()

    # Start from SONIC's own default standing pose: the policy is trained around
    # it, and the asset's initial state (hip pitch -0.10, knee 0.30) would put
    # the robot out of distribution on the very first control step.
    default_pos = sonic_default_joint_target(robot, body_ids)
    robot.write_joint_state_to_sim(default_pos, robot.data.default_joint_vel.clone().zero_())
    robot.set_joint_position_target(default_pos)

    reset_simulation_paused(sim, timeline)
    zero_body_drives(robot, body_ids)
    diagnostics["root_z_after_reset_paused"] = float(robot.data.root_pos_w[0][2])
    diagnostics["stiffness_after_zero"] = [
        float(robot.data.joint_stiffness[0, index]) for index in body_ids[:6]
    ]

    actuation = BodyActuation([limit.effort_max for limit in limits])
    metrics = Metrics()
    metrics.bind(
        body_ids=body_ids,
        limits=limits,
        read_joint_pos=lambda: tuple(float(robot.data.joint_pos[0, i]) for i in body_ids),
        read_root=lambda: read_root_state(robot),
    )
    apply_effort = make_effort_writer(robot, body_ids)
    # Point the viewport at the robot so the recording actually contains it.
    if hasattr(sim, "set_camera_view"):
        try:
            sim.set_camera_view(eye=(2.4, 2.4, 1.5), target=(0.0, 0.0, 0.6))
            diagnostics["camera_view_set"] = True
        except Exception as exc:  # noqa: BLE001
            diagnostics["camera_view_set"] = f"failed: {exc}"

    recorder = VideoRecorder(args.video_path)

    link = StateLink(port=args.ipc_port)
    link.open()

    state_snapshot = {"tick": 0}

    def build_state_frame() -> StateFrame:
        """The loop publishes this every frame; publishing stays in one place."""
        state_snapshot["tick"] += 1
        pos, quat, lin = read_root_state(robot)
        ang = tuple(float(v) for v in robot.data.root_ang_vel_w[0].tolist())
        q = tuple(float(robot.data.joint_pos[0, i]) for i in body_ids)
        dq = tuple(float(robot.data.joint_vel[0, i]) for i in body_ids)
        return StateFrame(
            tick_us=int(time.monotonic() * 1e6),
            root_pos=pos,
            root_quat_wxyz=quat,
            root_lin_vel=lin,
            root_ang_vel=ang,
            root_acc=(0.0, 0.0, 0.0),
            torso_quat_wxyz=quat,
            torso_gyro=ang,
            body_q=q,
            body_dq=dq,
            body_ddq=(0.0,) * BODY_JOINT_COUNT,
            body_tau_est=(0.0,) * BODY_JOINT_COUNT,
        )

    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(
            json.dumps(
                {
                    "robot_profile": profile.name,
                    "ipc_port": args.ipc_port,
                    "controller_required": not args.fall_test,
                }
            )
            + "\n"
        )

    # Paused hold: the GUI stays responsive and physics must not advance. Both
    # the PhysX clock and the robot's own pose are observed, because
    # SimulationContext.current_time does not track app.update() stepping.
    hold_root_before = float(robot.data.root_pos_w[0][2])
    hold_clock_before = physx_timestamp()
    hold_started = time.monotonic()
    hold_updates = 0
    while (time.monotonic() - hold_started) < args.paused_hold:
        simulation_app.update()
        # The bridge may connect during the paused hold; the link must be
        # accepted here as well as in the main loop.
        link.accept()
        link.publish(build_state_frame())
        hold_updates += 1
    hold_clock_after = physx_timestamp()
    hold_root_after = float(robot.data.root_pos_w[0][2])
    paused_hold = {
        "seconds": args.paused_hold,
        "app_updates": hold_updates,
        "physx_time_before": hold_clock_before,
        "physx_time_after": hold_clock_after,
        "root_z_before": hold_root_before,
        "root_z_after": hold_root_after,
        # The pose is the authoritative observable; the PhysX timestamp is
        # reported but may be unavailable, and NaN compares unequal.
        "physics_advanced": bool(hold_root_after != hold_root_before),
        "gui_responsive": hold_updates > 0,
    }

    if args.play_trigger is not None:
        # The controller must be armed before physics can flow, otherwise the
        # robot would fall while SONIC is still loading its models.
        waited = 0.0
        while not args.play_trigger.is_file():
            if waited >= args.play_trigger_timeout:
                print(json.dumps({"event": "play_trigger_timeout"}), file=sys.stderr, flush=True)
                _write_evidence(args.evidence_path, {
                    "status": "play_trigger_timeout",
                    "robot_profile": profile.name,
                    "waited_s": waited,
                })
                return EXIT_BROKEN, {"status": "play_trigger_timeout"}
            simulation_app.update()
            # The bridge connects while the scene is paused, so the link has to
            # be accepted here or the connection would sit unaccepted.
            link.accept()
            link.publish(build_state_frame())
            time.sleep(0.002)
            waited += 0.002
        diagnostics["play_trigger_wait_s"] = waited
        print(json.dumps({"event": "play_trigger_seen", "waited_s": waited}), flush=True)

    if args.auto_play or args.play_trigger is not None:
        sim.play()
        print(json.dumps({"event": "auto_play"}), flush=True)
    diagnostics["root_z_at_play"] = float(robot.data.root_pos_w[0][2])
    diagnostics["physx_time_at_play"] = physx_timestamp()

    # Kit can burst-step physics on the paused -> playing transition, which
    # would drop the robot before the measurement window opens. Re-freeze the
    # reset pose at Play so every run starts from the same standing state.
    reset_root_pos = robot.data.root_pos_w[0].clone()
    reset_root_quat = robot.data.root_quat_w[0].clone()
    zero_velocity = robot.data.default_joint_vel.clone().zero_()

    def freeze_at_play(_now: float) -> None:
        from isaaclab.utils.math import convert_quat

        robot.write_root_pose_to_sim(
            torch.cat([reset_root_pos, convert_quat(reset_root_quat, to="wxyz")]).unsqueeze(0)
        )
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
        robot.write_joint_state_to_sim(default_pos, zero_velocity)
        robot.set_joint_position_target(default_pos)
        scene.write_data_to_sim()
        diagnostics["root_z_at_play_refrozen"] = float(robot.data.root_pos_w[0][2])

    steps = run_interactive_app(
        simulation_app,
        timeline,
        sim,
        scene,
        actuation=actuation,
        link=link,
        metrics=metrics,
        read_state=build_state_frame,
        read_body_q_dq=read_body_q_dq(robot, body_ids),
        apply_effort=apply_effort,
        duration_s=args.duration,
        warmup_s=0.0 if args.fall_test else args.warmup,
        settle_frames=args.settle_frames,
        on_play=freeze_at_play,
        on_physics_step=recorder.maybe_record,
    )

    hand_state = read_hand_state(robot, hand_ids)
    exit_code = EXIT_OK
    if args.fall_test and not metrics.fall_observed():
        exit_code = EXIT_BROKEN

    payload = {
        "status": "ok" if exit_code == EXIT_OK else "fall_not_observed",
        "robot_profile": profile.name,
        "input": args.input,
        "auto_play": bool(args.auto_play),
        "physics_hz": PHYSICS_HZ,
        "physics_dt": PHYSICS_DT,
        "physics_steps": steps,
        "paused_hold": paused_hold,
        "diagnostics": diagnostics,
        "viewport": viewport_diagnostics(),
        "hands_controlled_by_sonic": HANDS_CONTROLLED_BY_SONIC,
        "hand_joint_count": len(hand_ids),
        "hand_q_final": list(hand_state.q),
        "joint_mapping": mapping_evidence_summary(mapping, limits),
        "metrics": metrics.summary(),
        "fall_observed": metrics.fall_observed(),
        "ipc": link.snapshot(),
        "video": recorder.write(),
    }
    link.close()
    _write_evidence(args.evidence_path, payload)
    if args.metrics_path is not None:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_path.write_text(json.dumps(payload["metrics"], indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "result", "status": payload["status"]}), flush=True)
    return exit_code, payload


if __name__ == "__main__":
    raise SystemExit(main())
