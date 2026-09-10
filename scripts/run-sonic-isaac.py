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


def reset_simulation_paused(sim: object, timeline: object | None) -> None:
    """``sim.reset()`` first, then pause; the ordering matters.

    ``reset()`` initialises the PhysX tensor views.  Pausing afterwards keeps
    them valid so the GUI Play button can advance physics later.
    """
    sim.reset()
    if timeline is not None:
        timeline.pause()


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
    on_play=None,
    on_physics_step=None,
) -> int:
    """Main GUI loop; returns the number of physics steps taken."""
    physics_steps = 0
    play_started_at: float | None = None
    measurement_started_at: float | None = None

    while simulation_app.is_running():
        now = time.monotonic()
        link.accept()
        link.publish(read_state())

        # Timeline transitions are resolved before anything is counted, so the
        # measurement window's counters start clean on the frame it opens.
        playing = timeline.is_playing()
        if playing and play_started_at is None:
            play_started_at = now
            print(json.dumps({"event": "timeline_play_started"}), flush=True)
            if on_play is not None:
                on_play(now)
        if playing and measurement_started_at is None and (now - play_started_at) >= warmup_s:
            measurement_started_at = now
            metrics.start_measurement(now)
            print(json.dumps({"event": "measurement_started", "warmup_s": warmup_s}), flush=True)

        command = link.take_command()
        if command is not None:
            outcome = actuation.submit(_to_sample(command, now))
            metrics.command_seen(command, now, accepted=outcome.accepted)

        step = None
        if playing:
            body_q, body_dq = read_body_q_dq()
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
        root_pos, quat = self._read_root()
        if self.initial_root_pos is None:
            self.initial_root_pos = root_pos
        if any(not math.isfinite(value) for value in root_pos + quat):
            self.nan_observations += 1
            return
        self._last_quat = quat
        self.root_xy_final = (root_pos[0], root_pos[1])
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
    from isaaclab_assets.robots.unitree import G1_29DOF_CFG, G1_INSPIRE_FTP_CFG

    template = G1_29DOF_CFG if profile.name == "g1-29dof" else G1_INSPIRE_FTP_CFG
    robot_cfg: ArticulationCfg = template.copy()
    # Both profiles stand on a free base with gravity on; the stock Inspire
    # config is a fixed-base manipulation setup, so both are pinned here.
    robot_cfg.spawn.articulation_props.fix_root_link = False
    robot_cfg.spawn.rigid_props.disable_gravity = False

    namespace = "{ENV_REGEX_NS}"

    @dataclass
    class SonicSceneCfg(InteractiveSceneCfg):
        ground: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()
        )
        robot: ArticulationCfg = robot_cfg.replace(prim_path=f"{namespace}/Robot")

    return SonicSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


def read_root_state(robot: object) -> tuple[tuple[float, ...], tuple[float, ...]]:
    pos = tuple(float(value) for value in robot.data.root_pos_w[0].tolist())
    quat = tuple(float(value) for value in robot.data.root_quat_w[0].tolist())
    return pos, quat


def read_body_q_dq(robot: object, body_ids: Sequence[int]):
    def read():
        q = tuple(float(robot.data.joint_pos[0, index]) for index in body_ids)
        dq = tuple(float(robot.data.joint_vel[0, index]) for index in body_ids)
        return q, dq

    return read


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
        for position, name in enumerate(names):
            try:
                joint_index = robot.joint_names.index(name)
            except ValueError:
                continue
            if joint_index in body_set:
                actuator.stiffness[position] = 0.0
                actuator.damping[position] = 0.0
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

    def __init__(self, path: Path | None, *, fps: int = 20, every: int = 10) -> None:
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
        if frame is not None:
            self.frames.append(frame.copy())

    def write(self) -> dict:
        if self.path is None:
            return {"status": "disabled", "frames": 0}
        if not self.frames:
            return {"status": self.status, "frames": 0}
        try:
            import imageio.v2 as iio

            self.path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(str(self.path), self.frames, fps=self.fps, codec="libx264")
        except Exception as exc:  # noqa: BLE001
            return {"status": f"encode_failed: {exc}", "frames": len(self.frames)}
        return {"status": "written", "frames": len(self.frames), "path": str(self.path)}


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
        simulation_app.close()
    os._exit(exit_code)


def _write_evidence(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run(args: argparse.Namespace, profile: Profile, simulation_app) -> tuple[int, dict]:
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from omni.timeline import get_timeline_interface

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device, use_fabric=True)
    )
    scene_cfg = build_scene(profile)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    body_ids, hand_ids = body_joint_indices(robot)
    limits = collect_limits(robot, body_ids)
    validate_limits(limits)
    mapping = build_joint_mapping(SONIC_BODY_JOINT_NAMES, list(robot.joint_names))

    timeline = get_timeline_interface()

    # Freeze the standing pose before anything can advance, then zero the
    # drives SONIC replaces and keep the fingers at their own open target.
    default_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(default_pos, robot.data.default_joint_vel.clone().zero_())
    robot.set_joint_position_target(default_pos)
    zero_body_drives(robot, body_ids)

    reset_simulation_paused(sim, timeline)

    actuation = BodyActuation([limit.effort_max for limit in limits])
    metrics = Metrics()
    metrics.bind(
        body_ids=body_ids,
        limits=limits,
        read_joint_pos=lambda: tuple(float(robot.data.joint_pos[0, i]) for i in body_ids),
        read_root=lambda: read_root_state(robot),
    )
    apply_effort = make_effort_writer(robot, body_ids)
    recorder = VideoRecorder(args.video_path)

    link = StateLink(port=args.ipc_port)
    link.open()

    state_snapshot = {"tick": 0}

    def publish_state() -> None:
        state_snapshot["tick"] += 1
        pos, quat = read_root_state(robot)
        lin = tuple(float(v) for v in robot.data.root_lin_vel_w[0].tolist())
        ang = tuple(float(v) for v in robot.data.root_ang_vel_w[0].tolist())
        q = tuple(float(robot.data.joint_pos[0, i]) for i in body_ids)
        dq = tuple(float(robot.data.joint_vel[0, i]) for i in body_ids)
        link.publish(
            StateFrame(
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

    # Paused hold: the GUI stays responsive and physics must not advance. The
    # simulation clock is the observable, not a promise from the loop.
    hold_clock_before = float(sim.current_time)
    hold_started = time.monotonic()
    hold_updates = 0
    while (time.monotonic() - hold_started) < args.paused_hold:
        simulation_app.update()
        publish_state()
        hold_updates += 1
    hold_clock_after = float(sim.current_time)
    paused_hold = {
        "seconds": args.paused_hold,
        "app_updates": hold_updates,
        "sim_time_before": hold_clock_before,
        "sim_time_after": hold_clock_after,
        "physics_advanced": hold_clock_after != hold_clock_before,
        "gui_responsive": hold_updates > 0,
    }

    if args.auto_play:
        sim.play()
        print(json.dumps({"event": "auto_play"}), flush=True)

    steps = run_interactive_app(
        simulation_app,
        timeline,
        sim,
        scene,
        actuation=actuation,
        link=link,
        metrics=metrics,
        read_state=publish_state,
        read_body_q_dq=read_body_q_dq(robot, body_ids),
        apply_effort=apply_effort,
        duration_s=args.duration,
        warmup_s=0.0 if args.fall_test else args.warmup,
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
