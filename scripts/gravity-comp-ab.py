#!/usr/bin/env python3
"""A/B: does a physically correct gravity feed-forward shrink the palm tracking error?

Experiment 03 established the mechanism of the 0.21-0.28 m palm gap: with
``tau_ff == 0`` and ``kp = 14.25`` the arm rests where the proportional term
carries its own weight.  This experiment adds the missing term -- PhysX's own
generalized gravity compensation torque of the body joints, taken from the
live articulation pose -- as an **opt-in** evaluation option
(``--gravity-feedforward``, default off) and measures A (current PD) against
B (PD + gravity feed-forward) on deterministic targets that need no checkpoint
and no policy: a neutral hold, a raised static target recorded from
``psi0-rollout-1``, and a slowed descent taken from the same recording.

Stages:

``profiles`` builds the three run profiles and the trajectory reference from
             the recorded rollout targets (deterministic; no simulator needed).
``run``      executes the A/B matrix in the dev container and keeps each run's
             raw outputs.  Nothing is scored here.
``analyse``  reads each run's tracking Parquet and summary, computes the
             acceptance metrics, and writes ``summary.json`` and the tables.
             Needs pyarrow (``data/venvs/hf-datasets``).
``figures``  renders three figures from ``summary.json`` (host matplotlib).
``manifest`` writes input hashes, the exact commands and the pinned constants.

The forward kinematics and the metric definitions are *not* re-implemented:
they are loaded from the validated ``scripts/compare-blockstacking-dataset.py``
and ``scripts/validate-blockstacking-tracking.py``.

    python3 scripts/gravity-comp-ab.py profiles
    python3 scripts/gravity-comp-ab.py run --repeats 3
    data/venvs/hf-datasets/bin/python scripts/gravity-comp-ab.py analyse
    python3 scripts/gravity-comp-ab.py figures
    python3 scripts/gravity-comp-ab.py manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "data" / "outputs" / "blockstacking-debug"
OUTPUT = CAMPAIGN / "experiments" / "04-gravity-comp-ab"
RUNS = OUTPUT / "runs"
DEFAULT_URDF = REPO_ROOT / "third_party" / "Psi0" / "real" / "assets" / "g1" / "g1_body29_hand14.urdf"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare-blockstacking-dataset.py"
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate-blockstacking-tracking.py"
RECORDED_ROLLOUT = CAMPAIGN / "psi0-rollout-1" / "rollouts" / "rollout-01" / "tracking.jsonl"

sys.path.insert(0, str(REPO_ROOT / "src"))
from humanoid_lab.controllers.sonic import (  # noqa: E402
    BODY_EFFORT_LIMIT_NM,
    BODY_JOINT_ORDER,
    SONIC_REFERENCE_JOINT_ORDER,
    deploy_gains,
    standing_pose,
)

SCHEMA_VERSION = 1
SCRIPT_VERSION = "1.0.0"
CONTAINER = "humanoid-lab-dev"
CONTAINER_REPO = "/workspace/humanoid-lab"
CONTAINER_STAGE = "/tmp/humanoid-lab-kit-cwd"

TRACKING_DT_S = 0.02
#: The support band of the evaluation profile is kept, and kept engaged.
#: Measured in this rig: a *static* joint target cannot be held in a free stand
#: by position control alone, in either variant -- the ankle must carry
#: m*g*(x_com - x_ankle) and a fixed ankle target supplies at most kp*e, so the
#: robot pivots until it falls (both variants fell 0.8-1.0 s after a band
#: release, and with a soft band that let the legs load from the start, both
#: fell too).  The deployment balances by moving its targets in closed loop;
#: the reference loop's own holds hang the waist on this same spring.  So the
#: band stays up for the whole run (a static target never arms the motion
#: release, and the timeout is put out of reach), and the question this
#: experiment answers -- what the PD law does with and without the gravity term
#: -- is measured under it.  Free-stand behaviour is reported as a separate,
#: explicitly limited observation.
SUPPORT_MAX_SECONDS = 600.0
SETTLE_MARGIN_S = 0.5
SUPPORT_ARMED_S = 1.5
#: Sim horizon each target is meant to observe (the descent is shorter by design).
SIM_HORIZON_S = {"t0": 9.0, "t1": 9.0, "t2": 9.0, "t3": 8.6, "t4": 9.0}

#: Targets.  ``t2`` is one recorded static target frame of the previous psi0
#: rollout; ``t3`` is a recorded target segment of the same rollout, replayed at
#: half speed.
T2_FRAME = 1395
T3_SEGMENT = (165, 290)
T3_STRETCH = 2
#: The descent starts at tick 0: a setpoint that moves from the first command
#: never arms the band's motion release, so the band stays engaged through it.
#: A 1.5 s ramp from the standing pose to the segment's first pose opens the
#: reference, so the measured window starts from a settled robot instead of
#: from the snap onto a raised pose.
T3_PRE_ROLL_S = 0.0
T3_RAMP_S = 1.5
T3_TAIL_HOLD_S = 2.0
#: The hold targets are one-frame references: frame 0 is held for the whole run.
HOLD_PRE_ROLL_S = 600.0

#: Wall-clock durations.  ``--duration`` is wall time and the loop runs at
#: roughly 0.6x real time on this CPU, so each one is set well above its sim
#: horizon (9 s for the holds, 12 s for the descent) and every run reaches it.
DURATIONS = {"t0": 16.0, "t1": 16.0, "t2": 16.0, "t3": 18.0, "t4": 12.0}
TARGET_PROFILES = {
    "t0": "t0-zerog-hold.json",
    "t1": "t1-neutral-hold.json",
    "t2": "t2-reach-hold.json",
    "t3": "t3-descent.json",
    "t4": "t4-release-check.json",
}
#: Acceptance targets.  t0 is the zero-gravity control, t4 the band-release
#: check: both are rig controls and neither decides the outcome.
ACCEPTANCE_TARGETS = ("t2", "t3")
VARIANTS = ("a", "b")
REFERENCE_NPZ = "reference-t3-descent.npz"

ARM_JOINTS = tuple(
    name
    for name in BODY_JOINT_ORDER
    if any(token in name for token in ("shoulder", "elbow", "wrist"))
)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
LEG_JOINTS = tuple(
    name
    for name in BODY_JOINT_ORDER
    if any(token in name for token in ("hip", "knee", "ankle"))
)
WRIST_PITCH_YAW = tuple(name for name in BODY_JOINT_ORDER if "wrist_pitch" in name or "wrist_yaw" in name)
SATURATION_TOL_NM = 1e-4

#: Acceptance, from the experiment design: a raised pose either lands under
#: 5 cm of palm-z MAE or loses 70 % of the baseline error, without a fall, a
#: sustained oscillation or a serious rise in clamp saturation.
ACCEPTANCE = {
    "palm_dz_mae_m": 0.05,
    "relative_reduction": 0.70,
    "minimum_root_z_m": 0.5,
    "oscillation_peak_to_peak_m": 0.02,
    "saturation_increase_factor": 1.5,
}
MEASURE_TOL_M = 0.02


class ExperimentError(RuntimeError):
    """The experiment cannot proceed; the message is for the operator."""


# ------------------------------------------------------------------- plumbing


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ExperimentError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def container_path(path) -> str:
    return f"{CONTAINER_REPO}/{Path(path).resolve().relative_to(REPO_ROOT)}"


def reorder(values: np.ndarray, source_names, target_names) -> np.ndarray:
    indices = [source_names.index(name) for name in target_names]
    return np.asarray(values)[..., indices]


# ------------------------------------------------------------------- profiles


def _base_profile() -> dict:
    """The evaluation's own rig, reduced to a deterministic target.

    Taken from the block-stacking evaluation profile: the same asset, the same
    scene, the same mass and joint-dynamics alignment to the pinned MuJoCo model
    (that alignment is what puts the simulated gravity torque and friction on
    the model the controller was tuned against -- and, measured here, what keeps
    the wrist joints out of a numerical limit cycle, which an unaligned rig with
    limp hands goes into), and the same deployment hand gains (kp 1.5, kd 0.1,
    read from the recorded rollout).  What changes is the driver: a recorded
    joint reference instead of the SONIC deployment, so the run needs no
    checkpoint and repeats exactly.  The camera service stays closed: these runs
    are headless and must not publish endpoints another run could collide with.
    """
    shipped = json.loads((REPO_ROOT / "configs" / "profiles" / "isaac-g1-sonic-blockstacking-dex3.json").read_text())
    scene = dict(shipped["scene"])
    return {
        "schema_version": 1,
        "robot": shipped["robot"],
        "camera": shipped["camera"],
        "controller": {
            "provider": "trajectory",
            "reference_path": None,
            "pre_roll_s": HOLD_PRE_ROLL_S,
            "torso_link": "torso_link",
            "command_ttl_s": 0.25,
            "hand_fallback": "passive",
            "support_link": "pelvis",
            "mass_alignment": "sonic_mujoco",
            "joint_dynamics_alignment": "sonic_mujoco",
        },
        "scene": {**scene, "camera_enabled": False},
        "simulation": {"physics_dt": 0.005},
        "initial_pose": "sonic_standing",
        "support": {**shipped["support"], "max_seconds": SUPPORT_MAX_SECONDS},
    }


def _pose_target(q_body: np.ndarray) -> dict:
    return {name: float(q_body[index]) for index, name in enumerate(BODY_JOINT_ORDER)}


def _episode(joint_pos: np.ndarray, joint_vel: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray) -> dict:
    """Canonical-episode arrays for the trajectory controller (SONIC order)."""
    frames = len(joint_pos)
    return {
        "timestamps": np.arange(frames, dtype=float) * TRACKING_DT_S,
        "joint_pos": reorder(joint_pos, BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER).astype(np.float32),
        "joint_vel": reorder(joint_vel, BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER).astype(np.float32),
        "body_quat_wxyz": np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 4)).copy(),
        "body_pos": np.zeros((frames, 3)),
        "left_hand_joints": np.asarray(left_hand, dtype=np.float32),
        "right_hand_joints": np.asarray(right_hand, dtype=np.float32),
    }


def _write_reference(name: str, episode: dict, detail: dict) -> dict:
    path = OUTPUT / "profiles" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **episode)
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path),
            "frames": int(len(episode["timestamps"])), **detail}


def build_hold_reference(name: str, body: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray,
                         detail: dict) -> dict:
    """A one-frame reference: the trajectory provider holds frame 0 forever."""
    episode = _episode(body[None, :], np.zeros((1, 29)), left_hand[None, :], right_hand[None, :])
    return _write_reference(name, episode, detail)


def build_descent_reference(name: str, standing: np.ndarray, standing_hand: dict) -> dict:
    """Half-speed replay of a recorded target segment, with a declared tail hold."""
    rows = read_jsonl(RECORDED_ROLLOUT)
    target = np.array([row["body_target"] for row in rows], dtype=float)
    velocity = np.array([row["body_velocity_target"] for row in rows], dtype=float)
    handed = {
        "left": np.array([row["left_hand_target"] for row in rows], dtype=float),
        "right": np.array([row["right_hand_target"] for row in rows], dtype=float),
    }
    start, stop = T3_SEGMENT
    segment = target[start : stop + 1]
    segment_velocity = velocity[start : stop + 1]
    # Time-stretch by linear interpolation over the recorded frames: every
    # stretched position is a convex combination of two recorded targets, and
    # the speed is divided by the stretch factor (a slower, bounded descent,
    # not a different motion).
    steps = (len(segment) - 1) * T3_STRETCH
    source_index = np.arange(0.0, steps + 1.0) / T3_STRETCH
    lower = np.clip(np.floor(source_index).astype(int), 0, len(segment) - 1)
    upper = np.clip(lower + 1, 0, len(segment) - 1)
    weight = (source_index - lower)[:, None]

    def stretch(values: np.ndarray) -> np.ndarray:
        return values[lower] * (1.0 - weight) + values[upper] * weight

    tail = int(round(T3_TAIL_HOLD_S / TRACKING_DT_S))
    ramp = int(round(T3_RAMP_S / TRACKING_DT_S))
    blend = np.linspace(0.0, 1.0, ramp, endpoint=False)[:, None]
    joint_pos = np.concatenate([
        standing[None, :] * (1.0 - blend) + segment[0][None, :] * blend,
        stretch(segment),
        np.broadcast_to(segment[-1], (tail, 29)),
    ], axis=0)
    joint_vel = np.concatenate([np.zeros((ramp, 29)), stretch(segment_velocity) / T3_STRETCH,
                                np.zeros((tail, 29))], axis=0)
    hands = {
        side: np.concatenate([
            standing_hand[side][None, :] * (1.0 - blend) + values[start][None, :] * blend,
            stretch(values[start : stop + 1]),
            np.broadcast_to(values[stop], (tail, 7)),
        ])
        for side, values in handed.items()
    }
    episode = _episode(joint_pos, joint_vel, hands["left"], hands["right"])
    return _write_reference(name, episode, {
        "source_rows": [start, stop],
        "source_window_s": [float(rows[start]["sim_s"]), float(rows[stop]["sim_s"])],
        "stretch": T3_STRETCH,
        "ramp_s": T3_RAMP_S,
        "ramp_from": "the deployment's standing pose",
        "tail_hold_s": T3_TAIL_HOLD_S,
    })


def stage_profiles(_: argparse.Namespace) -> int:
    rows = read_jsonl(RECORDED_ROLLOUT)
    target = np.array([row["body_target"] for row in rows], dtype=float)
    hands = {side: np.array([row[f"{side}_hand_target"] for row in rows], dtype=float)
             for side in ("left", "right")}
    raw = CAMPAIGN / "psi0-rollout-1" / "raw" / "isaac.tracking.parquet"
    import pyarrow.parquet as pq

    raw_hands = pq.read_table(raw, columns=["left_hand_target", "right_hand_target", "support_active"]).to_pydict()
    first_free = int(np.argmax([not value for value in raw_hands["support_active"]]))
    standing_hands = {side: np.array(raw_hands[f"{side}_hand_target"][first_free], dtype=float)
                      for side in ("left", "right")}
    standing = np.array([standing_pose()[name] for name in BODY_JOINT_ORDER])

    payloads = {}
    references = {}
    for key, body, hand in (
        ("t0", standing, standing_hands),
        ("t1", standing, standing_hands),
        ("t2", target[T2_FRAME], {side: hands[side][T2_FRAME] for side in ("left", "right")}),
    ):
        profile = _base_profile()
        profile["profile_id"] = f"isaac-g1-gravity-ab-{key}-hold"
        reference_name = f"reference-{key}-hold.npz"
        references[key] = build_hold_reference(reference_name, body, hand["left"], hand["right"], {
            "source": "recorded standing pose" if key in ("t0", "t1") else f"recorded target frame {T2_FRAME}",
            "source_sim_s": None if key in ("t0", "t1") else float(rows[T2_FRAME]["sim_s"]),
            "hand_source": f"recorded {('pre-policy standing hold' if key in ('t0', 't1') else 'target frame')} hand command",
        })
        profile["controller"]["reference_path"] = container_path(OUTPUT / "profiles" / reference_name)
        profile["controller"]["pre_roll_s"] = HOLD_PRE_ROLL_S
        payloads[key] = profile
    payloads["t0"]["robot"] = {**payloads["t0"]["robot"], "disable_gravity": True}
    # t4: the same neutral hold, with the evaluation's own 3 s band release.
    # It answers "what does this rig do when the support goes away?" and is the
    # evidence behind evaluating the holds on the band.
    release_check = _base_profile()
    release_check["profile_id"] = "isaac-g1-gravity-ab-release-check"
    release_check["controller"]["reference_path"] = container_path(OUTPUT / "profiles" / "reference-t1-hold.npz")
    release_check["controller"]["pre_roll_s"] = HOLD_PRE_ROLL_S
    release_check["support"] = {**release_check["support"], "max_seconds": 3.0}
    payloads["t4"] = release_check
    descent = _base_profile()
    descent["profile_id"] = "isaac-g1-gravity-ab-descent"
    references["t3"] = build_descent_reference(REFERENCE_NPZ, standing, standing_hands)
    descent["controller"]["reference_path"] = container_path(OUTPUT / "profiles" / REFERENCE_NPZ)
    descent["controller"]["pre_roll_s"] = T3_PRE_ROLL_S
    payloads["t3"] = descent

    references["t4"] = {**references["t1"], "shared_with": "t1"}
    description = {}
    for key, payload in payloads.items():
        path = OUTPUT / "profiles" / TARGET_PROFILES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        description[key] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path),
                            "duration_s": DURATIONS[key], "reference": references[key]}
    description["targets_note"] = (
        "t0/t1 hold the deployment's standing pose; t2 holds one recorded static target frame of "
        "psi0-rollout-1; t3 replays a recorded target segment at half speed.  t0 is the same rig as t1 "
        "with the robot's own gravity switched off (a control, not an acceptance target)."
    )
    # Is each target a pose the robot could stand in at all?  With no balance
    # controller in the loop, this is a property of the target, not of the run.
    compare = load_module(COMPARE_SCRIPT, "compare_blockstacking_dataset")
    validate = load_module(VALIDATE_SCRIPT, "validate_blockstacking_tracking")
    urdf = validate.load_urdf(compare, DEFAULT_URDF)
    balance = {
        "standing_pose": balance_check(urdf, standing[None, :]),
        "t2_target_frame": balance_check(urdf, target[T2_FRAME][None, :]),
    }
    reference_episode = np.load(OUTPUT / "profiles" / REFERENCE_NPZ)
    t3_body = reorder(reference_episode["joint_pos"], SONIC_REFERENCE_JOINT_ORDER, BODY_JOINT_ORDER)
    balance["t3_first_frame"] = balance_check(urdf, t3_body[:1])
    balance["t3_last_frame"] = balance_check(urdf, t3_body[-1:])
    description["target_balance"] = balance
    write_json(OUTPUT / "profiles" / "profiles.json", description)
    print(json.dumps(description, indent=2))
    return 0


# ------------------------------------------------------------------------ run


def run_command(profile: Path, duration: float, variant: str, run_dir: Path) -> list[str]:
    tracking = container_path(run_dir / "isaac.tracking.parquet")
    metrics = container_path(run_dir / "isaac.metrics.json")
    samples = container_path(run_dir / "isaac.samples.jsonl")
    flags = f"--gravity-feedforward " if variant == "b" else ""
    if profile.name == TARGET_PROFILES["t3"]:
        flags += f"--trajectory-reference {container_path(OUTPUT / 'profiles' / REFERENCE_NPZ)} "
    inner = (
        "source /opt/humanoid-lab/entrypoint.sh && use-isaac-sonic && "
        f"mkdir -p {CONTAINER_STAGE} && cd {CONTAINER_STAGE} && "
        "exec 9>/tmp/humanoid-lab-isaac-g1.lock && "
        "{ flock -n 9 || { echo 'another Isaac G1 simulation is running' >&2; exit 3; }; } && "
        f"exec python /workspace/humanoid-lab/scripts/run-isaac-g1.py "
        f"--profile {container_path(profile)} --headless --duration {duration} "
        f"{flags}--tracking-output {tracking} --metrics-output {metrics} --samples-output {samples}"
    )
    return ["docker", "exec", "-e", "DISPLAY=:0", CONTAINER, "bash", "-lc", inner]


def execute(profile: Path, duration: float, variant: str, run_dir: Path, timeout_s: float = 900.0) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = run_command(profile, duration, variant, run_dir)
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(f"[run] {run_dir.name}: {profile.name} variant={variant} duration={duration}s", flush=True)
    started = time.monotonic()
    with (run_dir / "run.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    elapsed = time.monotonic() - started
    ok = completed.returncode == 0 and (run_dir / "isaac.metrics.json").exists()
    print(f"[run] {run_dir.name}: exit={completed.returncode} elapsed={elapsed:.1f}s "
          f"metrics={'yes' if (run_dir / 'isaac.metrics.json').exists() else 'no'}", flush=True)
    return ok


def stage_run(args: argparse.Namespace) -> int:
    targets = [item for item in args.targets.split(",") if item]
    variants = [item for item in args.variants.split(",") if item]
    failures = []
    for target in targets:
        profile = OUTPUT / "profiles" / TARGET_PROFILES[target]
        if not profile.exists():
            raise ExperimentError(f"missing profile {profile}; run the profiles stage first")
        for variant in variants:
            for repeat in range(1, args.repeats + 1):
                name = f"{target}-{variant}" + ("" if repeat == 1 else f"-r{repeat}")
                run_dir = RUNS / name
                if (run_dir / "isaac.metrics.json").exists() and not args.force:
                    print(f"[run] {name}: already done, skipping")
                    continue
                if not execute(profile, DURATIONS[target], variant, run_dir):
                    failures.append(name)
                time.sleep(3.0)
    if failures:
        print(f"[run] FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


# -------------------------------------------------------------------- analysis


#: Measurement windows in sim time.  ``primary`` is the window the acceptance
#: speaks about: the static raised hold for T2, and for T3 the raised part of the
#: recorded descent.  The band is never released, so the windows open after the
#: initial transient instead of after a release.
TARGET_WINDOWS = {
    "t0": {"primary": (SUPPORT_ARMED_S, 9.0)},
    "t1": {"primary": (SUPPORT_ARMED_S, 9.0)},
    "t2": {"primary": (SUPPORT_ARMED_S, 9.0)},
    "t3": {
        "primary": (1.5, 6.5),
        "settle": (7.0, 8.5),
    },
    "t4": {"primary": (SUPPORT_ARMED_S, 3.0), "after_release": (3.5, 9.0)},
}
#: The window the settling and oscillation metrics are read on: a static phase.
#: Reading them on a moving reference would report the reference's own motion.
SETTLE_WINDOWS = {"t3": (7.0, 8.5)}


def run_rows(run_dir: Path) -> dict:
    """One run's tracking rows, plus the per-run summary it was written with."""
    import pyarrow.parquet as pq

    table = pq.read_table(run_dir / "isaac.tracking.parquet").to_pydict()
    summary = json.loads((run_dir / "isaac.metrics.json").read_text())
    keys = ("tick", "sim_s", "wall_time_ns", "support_active", "body_target", "body_applied_torque",
            "body_measured", "body_kp", "body_kd", "body_feedforward_torque",
            "body_gravity_feedforward_torque", "body_velocity_target", "body_measured_velocity",
            "root_position", "root_quaternion_wxyz", "torso_quaternion_wxyz")
    values = {key: table[key] for key in keys if key in table}
    return {"table": values, "summary": summary, "dir": run_dir}


def matrix(rows: dict, key: str, width: int) -> np.ndarray:
    if key not in rows["table"]:
        return np.full((len(rows["table"]["tick"]), width), np.nan)
    return np.array([value if value is not None else [np.nan] * width for value in rows["table"][key]], dtype=float)


def window_mask(rows: dict, start_s: float | None = None, stop_s: float | None = None) -> np.ndarray:
    sim_s = np.array(rows["table"]["sim_s"], dtype=float)
    support = np.array(rows["table"]["support_active"], dtype=bool)
    mask = ~support
    if start_s is not None:
        mask &= sim_s >= start_s
    if stop_s is not None:
        mask &= sim_s <= stop_s
    return mask


def settling_and_oscillation(sim_s: np.ndarray, error: np.ndarray, tol: float) -> dict:
    """Settle time, overshoot and the residual oscillation of one error trace."""
    if error.size == 0:
        return {"n_rows": 0}
    inside = np.abs(error) <= tol
    settled_index = None
    for index in range(len(error)):
        if inside[index] and inside[index:].all():
            settled_index = index
            break
    tail = error[sim_s >= sim_s[-1] - 2.0]
    velocity = np.diff(tail)
    sign_changes = int(np.sum(np.sign(velocity[:-1]) * np.sign(velocity[1:]) < 0)) if tail.size > 2 else 0
    return {
        "n_rows": int(error.size),
        "tol_m": float(tol),
        "settled": settled_index is not None,
        "settling_time_s": None if settled_index is None else float(sim_s[settled_index] - sim_s[0]),
        "final_error_m": float(error[-max(1, int(0.2 / TRACKING_DT_S)):].mean()),
        "overshoot_m": float(error.max() - error[-max(1, int(0.5 / TRACKING_DT_S)):].mean()),
        "tail_peak_to_peak_m": float(tail.max() - tail.min()) if tail.size else None,
        "tail_direction_changes": sign_changes,
    }


def saturation_rates(applied: np.ndarray, limits: np.ndarray, groups: dict[str, tuple[str, ...]]) -> dict:
    at_limit = np.abs(np.abs(applied) - limits[None, :]) <= SATURATION_TOL_NM
    out = {}
    for name, joints in groups.items():
        indices = [BODY_JOINT_ORDER.index(joint) for joint in joints]
        out[name] = {
            "mean_rate": float(at_limit[:, indices].mean()),
            "worst_joint_rate": float(at_limit[:, indices].mean(axis=0).max()),
            "peak_abs_nm": float(np.abs(applied[:, indices]).max()),
        }
    return out


def fk_gate(rows: dict, urdf, compare, validate, run_dir: Path) -> dict:
    """Check the FK against the simulator's own palm poses before reporting metrics.

    The simulator prints a per-second palm sample (world position of each palm
    link) with a wall clock, and every tracking row carries one too, so a sample
    can be paired with the row it was taken from.  The same FK is then run on
    that row's measured joints and root pose.  Two negative controls ride along:
    they must be far worse, or the check proves nothing.
    """
    samples_path = run_dir / "isaac.samples.jsonl"
    if not samples_path.exists():
        return {"samples": 0, "reason": "no palm samples (the profile declares no scene)"}
    samples = [row for row in read_jsonl(samples_path) if row.get("palm_pose_w")]
    if not samples:
        return {"samples": 0, "reason": "the sample stream carries no palm poses"}
    wall = np.array(rows["table"]["wall_time_ns"], dtype=np.int64)
    measured = matrix(rows, "body_measured", 29)
    base = validate.base_matrix(matrix(rows, "root_position", 3),
                               matrix(rows, "root_quaternion_wxyz", 4), compare)
    palms = urdf.palms(measured, base)
    left_indices = [index for index, name in enumerate(BODY_JOINT_ORDER) if name.startswith("left_")]
    right_indices = [index for index, name in enumerate(BODY_JOINT_ORDER) if name.startswith("right_")]
    swapped = np.array(measured)
    swapped[:, left_indices] = measured[:, right_indices]
    swapped[:, right_indices] = measured[:, left_indices]
    swapped_palms = urdf.palms(swapped, base)
    zeroed = np.array(measured)
    zeroed[:, [BODY_JOINT_ORDER.index(name) for name in ARM_JOINTS]] = 0.0
    zeroed_palms = urdf.palms(zeroed, base)
    errors: dict[str, list[float]] = {"left": [], "right": [], "swapped": [], "zero_arm": [], "pairing_ms": []}
    for sample in samples:
        stamp = int(sample["wall_time_ns"])
        index = int(np.argmin(np.abs(wall - stamp)))
        gap_ms = abs(int(wall[index]) - stamp) / 1e6
        if gap_ms > 50.0:
            continue
        errors["pairing_ms"].append(gap_ms)
        for side, link in (("left", "left_hand_palm_link"), ("right", "right_hand_palm_link")):
            recorded = np.array(sample["palm_pose_w"][link][:3], dtype=float)
            errors[side].append(float(np.linalg.norm(palms[side][index] - recorded)))
        recorded_left = np.array(sample["palm_pose_w"]["left_hand_palm_link"][:3], dtype=float)
        errors["swapped"].append(float(np.linalg.norm(swapped_palms["left"][index] - recorded_left)))
        errors["zero_arm"].append(float(np.linalg.norm(zeroed_palms["left"][index] - recorded_left)))
    return {
        "samples": len(errors["left"]),
        "pairing_median_ms": float(np.median(errors["pairing_ms"])) if errors["pairing_ms"] else None,
        "max_error_m": float(max(errors["left"] + errors["right"])) if errors["left"] else None,
        "left_mean_error_m": float(np.mean(errors["left"])) if errors["left"] else None,
        "right_mean_error_m": float(np.mean(errors["right"])) if errors["right"] else None,
        "swapped_arm_control_m": float(np.mean(errors["swapped"])) if errors["swapped"] else None,
        "zero_arm_control_m": float(np.mean(errors["zero_arm"])) if errors["zero_arm"] else None,
        "tolerance_m": 1e-3,
    }


def analyse_run(run_dir: Path, urdf, compare, validate) -> dict:
    rows = run_rows(run_dir)
    sim_s = np.array(rows["table"]["sim_s"], dtype=float)
    support = np.array(rows["table"]["support_active"], dtype=bool)
    target = matrix(rows, "body_target", 29)
    measured = matrix(rows, "body_measured", 29)
    applied = matrix(rows, "body_applied_torque", 29)
    kp = matrix(rows, "body_kp", 29)
    kd = matrix(rows, "body_kd", 29)
    feedforward = matrix(rows, "body_feedforward_torque", 29)
    gravity = matrix(rows, "body_gravity_feedforward_torque", 29)
    measured_velocity = matrix(rows, "body_measured_velocity", 29)
    velocity_target = matrix(rows, "body_velocity_target", 29)
    limits = np.array(BODY_EFFORT_LIMIT_NM, dtype=float)

    target_name = run_dir.name.split("-")[0]
    windows = {
        name: (sim_s >= start) & (sim_s <= stop)
        for name, (start, stop) in TARGET_WINDOWS[target_name].items()
    }
    if not any(mask.any() for mask in windows.values()):
        raise ExperimentError(f"{run_dir.name}: no measurement window has rows; the run is too short")

    window = windows["primary"]
    pelvis_target = urdf.palms(target)
    pelvis_measured = urdf.palms(measured)
    arm_index = np.array([BODY_JOINT_ORDER.index(name) for name in ARM_JOINTS])
    waist_index = np.array([BODY_JOINT_ORDER.index(name) for name in WAIST_JOINTS])
    leg_index = np.array([BODY_JOINT_ORDER.index(name) for name in LEG_JOINTS])

    measured_windows = {}
    for name, mask in windows.items():
        palm = validate.palm_metrics(
            {side: pelvis_target[side][mask] for side in ("left", "right")},
            {side: pelvis_measured[side][mask] for side in ("left", "right")},
        )
        measured_windows[name] = {
            "rows": int(mask.sum()),
            "sim_s": [float(sim_s[mask][0]), float(sim_s[mask][-1])] if mask.any() else None,
            "arm_joints": validate.joint_metrics(target[mask], measured[mask], arm_index),
            "waist_joints": validate.joint_metrics(target[mask], measured[mask], waist_index),
            "leg_joints": validate.joint_metrics(target[mask], measured[mask], leg_index),
            "palm": palm,
            "palm_dz_mae_m": palm["pooled"]["dz_mae_m"],
            "palm_dz_bias_m": palm["pooled"]["dz_bias_m"],
            "palm_rmse_3d_m": palm["pooled"]["rmse_3d_m"],
            "saturation": saturation_rates(applied[mask], limits, {
                "arm": ARM_JOINTS, "waist": WAIST_JOINTS, "legs": LEG_JOINTS,
                "wrist_pitch_yaw": WRIST_PITCH_YAW,
            }),
        }
    arms = measured_windows["primary"]["arm_joints"]
    waist = measured_windows["primary"]["waist_joints"]
    legs = measured_windows["primary"]["leg_joints"]
    palm = measured_windows["primary"]["palm"]

    settle_window = SETTLE_WINDOWS.get(target_name)
    settle_mask = window if settle_window is None else (
        (sim_s >= settle_window[0]) & (sim_s <= settle_window[1]))
    error_z = np.mean([pelvis_target[side][settle_mask, 2] - pelvis_measured[side][settle_mask, 2]
                       for side in ("left", "right")], axis=0)
    settle = settling_and_oscillation(sim_s[settle_mask], error_z, MEASURE_TOL_M)
    settle["window_sim_s"] = [float(sim_s[settle_mask][0]), float(sim_s[settle_mask][-1])] if settle_mask.any() else None

    # Torque identity: what was staged must be the clamped PD law, plus the
    # gravity term when the run asked for it.  It fails loudly if the recorded
    # law is not the law that ran.
    pd = feedforward + kp * (target - measured) + kd * (velocity_target - measured_velocity)
    wanted = pd + (gravity if np.isfinite(gravity).any() else 0.0)
    residual = np.nanmax(np.abs(np.clip(wanted, -limits, limits) - applied))

    root = matrix(rows, "root_position", 3)
    torso = matrix(rows, "torso_quaternion_wxyz", 4)
    up_z = 1.0 - 2.0 * (torso[:, 1] ** 2 + torso[:, 2] ** 2)
    after = sim_s >= SUPPORT_ARMED_S
    return {
        "run": run_dir.name,
        "target": target_name,
        "variant": run_dir.name.split("-")[1][0],
        "rows": int(len(sim_s)),
        "sim_seconds": float(sim_s[-1] - sim_s[0]) if len(sim_s) else 0.0,
        "horizon_s": SIM_HORIZON_S[target_name],
        "horizon_reached": bool(len(sim_s)) and float(sim_s[-1] - sim_s[0]) >= SIM_HORIZON_S[target_name] - 0.1,
        "windows": measured_windows,
        "window_rows": int(window.sum()),
        "window_sim_s": [float(sim_s[window][0]), float(sim_s[window][-1])] if window.any() else None,
        "support_release_s": float(sim_s[~support][0]) if (~support).any() else None,
        "arm_joints": arms,
        "waist_joints": waist,
        "leg_joints": legs,
        "palm": palm,
        "palm_dz_mae_m": palm["pooled"]["dz_mae_m"],
        "palm_dz_bias_m": palm["pooled"]["dz_bias_m"],
        "palm_rmse_3d_m": palm["pooled"]["rmse_3d_m"],
        "settling": settle,
        "saturation": measured_windows["primary"]["saturation"],
        "stability": {
            "root_z_min_m": float(root[after, 2].min()) if after.any() else None,
            "root_z_end_m": float(root[-1, 2]),
            "root_x_drift_m": float(root[after, 0].max() - root[after, 0].min()) if after.any() else None,
            "root_y_drift_m": float(root[after, 1].max() - root[after, 1].min()) if after.any() else None,
            "torso_up_z_min": float(up_z[after].min()) if after.any() else None,
            "fell": bool(after.any() and root[after, 2].min() < ACCEPTANCE["minimum_root_z_m"]),
            "root_z_min_after_1s_m": float(root[sim_s >= 1.0][:, 2].min()),
            "fell_at_s": (
                float(sim_s[sim_s >= 1.0][np.argmax(root[sim_s >= 1.0][:, 2] < ACCEPTANCE["minimum_root_z_m"])])
                if bool((root[sim_s >= 1.0][:, 2] < ACCEPTANCE["minimum_root_z_m"]).any()) else None
            ),
        },
        "torque_identity_max_residual_nm": float(residual),
        "gravity_term": {
            "recorded": bool(np.isfinite(gravity).any()),
            "mean_abs_nm": float(np.nanmean(np.abs(gravity))) if np.isfinite(gravity).any() else None,
            "max_abs_nm": float(np.nanmax(np.abs(gravity))) if np.isfinite(gravity).any() else None,
            "summary": rows["summary"].get("gravity_feedforward"),
        },
        "controller": {
            "provider": (rows["summary"].get("controller") or {}).get("kind"),
            "control_mode": (rows["summary"].get("controller") or {}).get("control_mode"),
            "applied_ticks": (rows["summary"].get("controller") or {}).get("applied_ticks"),
            "stale_ticks": (rows["summary"].get("controller") or {}).get("stale_ticks"),
            "rejected_commands": (rows["summary"].get("controller") or {}).get("rejected_commands"),
        },
        "fk_gate": fk_gate(rows, urdf, compare, validate, run_dir),
    }


def reference_metrics(run_dir: Path, urdf) -> dict:
    """Per-joint comparison of the recorded gravity term against the P-term.

    In A the equilibrium is ``kp*e = gravity torque``: the P-term that holds the
    arm is the same physical quantity PhysX reports.  Equal magnitudes and equal
    signs, joint by joint, are what a correct mapping (index, offset, order,
    sign) looks like -- a wrong offset would pair one joint's P-term with
    another joint's gravity torque.
    """
    rows = run_rows(run_dir)
    target = matrix(rows, "body_target", 29)
    measured = matrix(rows, "body_measured", 29)
    kp = matrix(rows, "body_kp", 29)
    applied = matrix(rows, "body_applied_torque", 29)
    kd = matrix(rows, "body_kd", 29)
    measured_velocity = matrix(rows, "body_measured_velocity", 29)
    gravity = matrix(rows, "body_gravity_feedforward_torque", 29)
    sim_s = np.array(rows["table"]["sim_s"], dtype=float)
    window = (sim_s >= TARGET_WINDOWS[run_dir.name.split("-")[0]]["primary"][0]) & (
        sim_s <= TARGET_WINDOWS[run_dir.name.split("-")[0]]["primary"][1])
    p_term = kp * (target - measured)
    rows_out = []
    for index, name in enumerate(BODY_JOINT_ORDER):
        rows_out.append({
            "joint": name,
            "p_term_mean_nm": float(p_term[window, index].mean()),
            "gravity_mean_nm": float(gravity[window, index].mean()) if np.isfinite(gravity).any() else None,
            "applied_mean_nm": float(applied[window, index].mean()),
            "kd_term_mean_nm": float((kd * (0.0 - measured_velocity))[window, index].mean()),
        })
    return {"window_rows": int(window.sum()), "joints": rows_out}


def gravity_model_check(run_dir: Path, urdf, validate) -> dict:
    """Compare the PhysX gravity term against the URDF's own gravity moment.

    ``gravity_torques`` sums, joint by joint, the moment about the joint axis of
    every distal link's weight (masses and centres of mass from the pinned URDF,
    g = 9.81 m/s^2).  The torque an actuator must apply to hold that joint still
    is its negative, which is what PhysX's compensation vector is.  Agreement in
    magnitude, sign and per-joint pattern is therefore a check of the whole
    mapping at once -- offset, order, sign -- and of the world gravity the scene
    runs with, from data that is independent of the simulator's answer.
    """
    rows = run_rows(run_dir)
    measured = matrix(rows, "body_measured", 29)
    gravity = matrix(rows, "body_gravity_feedforward_torque", 29)
    if not np.isfinite(gravity).any():
        return {"run": run_dir.name, "reason": "the run carries no gravity term"}
    sim_s = np.array(rows["table"]["sim_s"], dtype=float)
    window = (sim_s >= TARGET_WINDOWS[run_dir.name.split("-")[0]]["primary"][0]) & (
        sim_s <= TARGET_WINDOWS[run_dir.name.split("-")[0]]["primary"][1])
    modelled = -validate.gravity_torques(urdf, measured[window])
    recorded = gravity[window]
    joints = []
    for index, name in enumerate(BODY_JOINT_ORDER):
        joints.append({
            "joint": name,
            "recorded_mean_nm": float(recorded[:, index].mean()),
            "modelled_mean_nm": float(modelled[:, index].mean()),
            "abs_difference_nm": float(abs(recorded[:, index].mean() - modelled[:, index].mean())),
        })
    strong = [item for item in joints if abs(item["modelled_mean_nm"]) > 0.5]
    return {
        "run": run_dir.name,
        "window_rows": int(window.sum()),
        "g_m_s2": 9.81,
        "joints_compared": len(strong),
        "mean_abs_recorded_nm": float(np.abs(recorded).mean()),
        "mean_abs_modelled_nm": float(np.abs(modelled).mean()),
        "max_abs_joint_difference_nm": max((item["abs_difference_nm"] for item in strong), default=None),
        "sign_agreement": float(np.mean([np.sign(item["recorded_mean_nm"]) == np.sign(item["modelled_mean_nm"])
                                         for item in strong])) if strong else None,
        "joints": joints,
    }


def determinism_check(run_dirs: list[Path]) -> dict:
    """Are repeated cells the same run?  Compared on the common sim horizon."""
    keys = ("tick", "body_target", "body_measured", "root_position", "body_applied_torque",
            "body_gravity_feedforward_torque")
    loaded = [run_rows(path) for path in run_dirs]
    rows = [len(item["table"]["tick"]) for item in loaded]
    common = min(rows)
    differences = {}
    for key in keys:
        if not all(key in item["table"] for item in loaded):
            continue
        arrays = [np.asarray(item["table"][key])[:common] for item in loaded]
        differences[key] = float(max(np.abs(arrays[0] - arrays[index]).max() for index in range(1, len(arrays))))
    return {
        "runs": [path.name for path in run_dirs],
        "rows": rows,
        "common_rows": common,
        "max_abs_difference": differences,
        "identical": all(value == 0.0 for value in differences.values()),
    }


def write_series(results: dict, urdf) -> list[str]:
    """Per-row palm-z error of every run, for the figures and for later reuse."""
    payload = {}
    for name in sorted(results):
        rows = run_rows(RUNS / name)
        sim_s = np.array(rows["table"]["sim_s"], dtype=float)
        target = matrix(rows, "body_target", 29)
        measured = matrix(rows, "body_measured", 29)
        target_palm = urdf.palms(target)
        measured_palm = urdf.palms(measured)
        payload[f"{name}__sim_s"] = sim_s.astype(np.float32)
        payload[f"{name}__support"] = np.array(rows["table"]["support_active"], dtype=bool)
        for side in ("left", "right"):
            payload[f"{name}__dz_{side}"] = (target_palm[side][:, 2] - measured_palm[side][:, 2]).astype(np.float32)
    path = OUTPUT / "series" / "palm_error_series.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return [str(path.relative_to(REPO_ROOT))]


def balance_check(urdf, q_body: np.ndarray) -> dict:
    """Where the whole-body centre of mass sits relative to the feet, in the pelvis frame.

    A command that PD-only control is asked to realise is only standable if its
    centre of mass projects inside the feet; with no balance controller in the
    loop this decides whether a pose can be held at all.  The feet are given a
    nominal half-length because the URDF carries joint frames, not sole
    geometry, so the result is stated against that assumption.
    """
    links, masses, coms = [], [], []
    for link in urdf.link_names:
        mass = urdf.masses.get(link, 0.0)
        if mass <= 0.0:
            continue
        chain = urdf.chain(link)
        tip = urdf.fk_chain(chain, urdf.chain_joint_values(chain, q_body))
        centre = tip[:, :3, 3] + np.einsum("nij,j->ni", tip[:, :3, :3], urdf.com_xyz.get(link, np.zeros(3)))
        links.append(centre)
        masses.append(mass)
        coms.append(centre)
    weights = np.array(masses)[None, :, None]
    centre_of_mass = (np.stack(links, axis=1) * weights).sum(axis=1) / np.array(masses).sum()
    feet = {}
    for side in ("left", "right"):
        link = f"{side}_ankle_roll_link"
        chain = urdf.chain(link)
        tip = urdf.fk_chain(chain, urdf.chain_joint_values(chain, q_body))
        feet[side] = tip[:, :3, 3]
    half = 0.10
    x_min = min(feet["left"][:, 0].min(), feet["right"][:, 0].min()) - half
    x_max = max(feet["left"][:, 0].max(), feet["right"][:, 0].max()) + half
    return {
        "frame": "pelvis",
        "com_x_m": float(centre_of_mass[0, 0]),
        "com_y_m": float(centre_of_mass[0, 1]),
        "foot_x_m": [float(x_min), float(x_max)],
        "foot_half_length_assumed_m": half,
        "com_inside_nominal_support": bool(x_min <= centre_of_mass[0, 0] <= x_max),
        "total_mass_kg": float(weights.sum()),
    }


def stage_analyse(_: argparse.Namespace) -> int:
    compare = load_module(COMPARE_SCRIPT, "compare_blockstacking_dataset")
    validate = load_module(VALIDATE_SCRIPT, "validate_blockstacking_tracking")
    urdf = validate.load_urdf(compare, DEFAULT_URDF)
    run_dirs = sorted(path for path in RUNS.iterdir() if path.is_dir() and (path / "isaac.metrics.json").exists())
    if not run_dirs:
        raise ExperimentError(f"no completed runs under {RUNS}")
    results = {}
    for run_dir in run_dirs:
        results[run_dir.name] = analyse_run(run_dir, urdf, compare, validate)
        print(f"[analyse] {run_dir.name}: palm dz MAE={results[run_dir.name]['palm_dz_mae_m']:.4f} m "
              f"arm MAE={results[run_dir.name]['arm_joints']['mae_rad']:.4f} rad", flush=True)

    # A/B pairing per target, on the first repeat of each cell.
    comparisons = {}
    for target in sorted({item["target"] for item in results.values()}):
        base = results.get(f"{target}-a")
        variant = results.get(f"{target}-b")
        if base is None or variant is None:
            continue
        comparisons[target] = {
            "arm_mae_reduction": 1.0 - variant["arm_joints"]["mae_rad"] / base["arm_joints"]["mae_rad"],
            "palm_dz_mae_reduction": 1.0 - variant["palm_dz_mae_m"] / base["palm_dz_mae_m"],
            "palm_dz_mae_a_m": base["palm_dz_mae_m"],
            "palm_dz_mae_b_m": variant["palm_dz_mae_m"],
            "palm_dz_bias_a_m": base["palm_dz_bias_m"],
            "palm_dz_bias_b_m": variant["palm_dz_bias_m"],
            "palm_rmse_3d_a_m": base["palm_rmse_3d_m"],
            "palm_rmse_3d_b_m": variant["palm_rmse_3d_m"],
            "saturation_arm_a": base["saturation"]["arm"]["mean_rate"],
            "saturation_arm_b": variant["saturation"]["arm"]["mean_rate"],
            "fell_a": base["stability"]["fell"],
            "fell_b": variant["stability"]["fell"],
            "tail_peak_to_peak_b_m": variant["settling"].get("tail_peak_to_peak_m"),
        }
        acceptance = {
            "palm_dz_mae_under_5cm": variant["palm_dz_mae_m"] < ACCEPTANCE["palm_dz_mae_m"],
            "relative_reduction_at_least_70pct": comparisons[target]["palm_dz_mae_reduction"] >= ACCEPTANCE["relative_reduction"],
            "no_fall_a": not base["stability"]["fell"],
            "no_fall_b": not variant["stability"]["fell"],
            "no_sustained_oscillation": (
                variant["settling"].get("tail_peak_to_peak_m") is not None
                and variant["settling"]["tail_peak_to_peak_m"] < ACCEPTANCE["oscillation_peak_to_peak_m"]
            ),
            "saturation_not_seriously_worse": (
                comparisons[target]["saturation_arm_b"]
                <= ACCEPTANCE["saturation_increase_factor"] * max(comparisons[target]["saturation_arm_a"], 1e-6)
            ),
        }
        comparisons[target]["acceptance"] = acceptance
        comparisons[target]["acceptance_result"] = "PASS" if all(acceptance.values()) else "FAIL"

    # Determinism: repeats of one cell compared on their common sim horizon.
    repeats = {}
    for target in sorted({item["target"] for item in results.values()}):
        for variant in VARIANTS:
            names = [f"{target}-{variant}"]
            names += [f"{target}-{variant}-r{repeat}" for repeat in range(2, 9)
                      if (RUNS / f"{target}-{variant}-r{repeat}").is_dir()]
            names = [name for name in names if (RUNS / name).is_dir()]
            if len(names) > 1:
                repeats[f"{target}-{variant}"] = determinism_check([RUNS / name for name in names])
    tripled = {cell: payload for cell, payload in repeats.items() if len(payload["runs"]) >= 3}
    repeats["summary"] = {
        "cells": sorted(repeats),
        "cells_with_three_repeats": sorted(tripled),
        "all_identical": (all(payload["identical"] for payload in tripled.values())
                          if tripled else None),
        "note": "same cell, repeats compared row by row on the common simulated horizon; "
                "a non-zero difference in any column means the run is not reproducible",
    }

    gravity_check = {}
    for variant_run in ("t0-b", "t2-a", "t2-b", "t3-b"):
        if (RUNS / variant_run).is_dir():
            gravity_check[variant_run] = reference_metrics(RUNS / variant_run, urdf)
    model_check = {}
    for variant_run in ("t2-b", "t3-b"):
        if (RUNS / variant_run).is_dir():
            model_check[variant_run] = gravity_model_check(RUNS / variant_run, urdf, validate)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "04-gravity-comp-ab",
        "question": "Does adding a physically correct gravity feed-forward to the evaluation "
                    "controller's PD law reduce the arm/palm tracking error safely?",
        "acceptance": ACCEPTANCE,
        "targets": json.loads((OUTPUT / "profiles" / "profiles.json").read_text()),
        "runs": results,
        "comparisons": comparisons,
        "repeatability": repeats,
        "gravity_term_check": gravity_check,
        "gravity_model_check": model_check,
        "decision": None,
    }
    summary["decision"] = decide(summary)
    summary["series"] = write_series(results, urdf)
    write_json(OUTPUT / "summary.json", summary)
    write_table(OUTPUT / "tables" / "runs.csv", [flatten_run(name, result) for name, result in results.items()])
    write_table(OUTPUT / "tables" / "gravity_term_vs_p_term.csv", gravity_table(gravity_check))
    if model_check:
        write_table(OUTPUT / "tables" / "gravity_model_check.csv",
                    [{"run": run, **joint} for run, payload in model_check.items() for joint in payload["joints"]])
    print(f"[analyse] wrote {OUTPUT / 'summary.json'}")
    print(f"[analyse] decision: {summary['decision']['result']} - {summary['decision']['reason']}")
    return 0


def flatten_run(name: str, result: dict) -> dict:
    return {
        "run": name,
        "target": result["target"],
        "variant": result["variant"],
        "rows": result["rows"],
        "sim_seconds": result["sim_seconds"],
        "horizon_reached": result["sim_seconds"] >= SIM_HORIZON_S[result["target"]] - 0.1,
        "window_rows": result["window_rows"],
        "window_sim_s": f"{result['window_sim_s'][0]:.2f}-{result['window_sim_s'][1]:.2f}" if result["window_sim_s"] else "",
        "support_release_s": result["support_release_s"],
        "arm_mae_rad": result["arm_joints"]["mae_rad"],
        "arm_rmse_rad": result["arm_joints"]["rmse_rad"],
        "arm_bias_rad": result["arm_joints"]["bias_rad"],
        "waist_mae_rad": result["waist_joints"]["mae_rad"],
        "leg_mae_rad": result["leg_joints"]["mae_rad"],
        "palm_dz_mae_m": result["palm_dz_mae_m"],
        "palm_dz_bias_m": result["palm_dz_bias_m"],
        "palm_rmse_3d_m": result["palm_rmse_3d_m"],
        "settling_time_s": result["settling"].get("settling_time_s"),
        "tail_peak_to_peak_m": result["settling"].get("tail_peak_to_peak_m"),
        "overshoot_m": result["settling"].get("overshoot_m"),
        "saturation_arm_mean": result["saturation"]["arm"]["mean_rate"],
        "saturation_arm_worst_joint": result["saturation"]["arm"]["worst_joint_rate"],
        "saturation_legs_mean": result["saturation"]["legs"]["mean_rate"],
        "root_z_min_m": result["stability"]["root_z_min_m"],
        "fell": result["stability"]["fell"],
        "torque_identity_residual_nm": result["torque_identity_max_residual_nm"],
        "gravity_mean_abs_nm": result["gravity_term"]["mean_abs_nm"],
        "gravity_max_abs_nm": result["gravity_term"]["max_abs_nm"],
        "fk_gate_max_error_m": result["fk_gate"].get("max_error_m"),
        "fk_gate_samples": result["fk_gate"].get("samples"),
    }


def gravity_table(gravity_check: dict) -> list[dict]:
    table = []
    for run, payload in gravity_check.items():
        for joint in payload["joints"]:
            table.append({"run": run, **joint})
    return table


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def decide(summary: dict) -> dict:
    """The experiment's own decision rule: A, B, C or D."""
    comparisons = summary["comparisons"]
    raised = [name for name in ("t2", "t3") if name in comparisons]
    if not comparisons:
        return {"result": "D", "reason": "no paired A/B runs were analysed"}
    passed = [name for name in raised if comparisons[name]["acceptance_result"] == "PASS"]
    improved = [name for name in comparisons if comparisons[name]["palm_dz_mae_reduction"] > 0]
    fell = [name for name in ACCEPTANCE_TARGETS
            if name in comparisons and (comparisons[name]["fell_a"] or comparisons[name]["fell_b"])]
    unstable = [
        name for name, item in comparisons.items() if name in ACCEPTANCE_TARGETS
        if item["tail_peak_to_peak_b_m"] is not None
        and item["tail_peak_to_peak_b_m"] >= 5 * ACCEPTANCE["oscillation_peak_to_peak_m"]
    ]
    if fell or unstable:
        return {
            "result": "C",
            "reason": ("a run fell" if fell else "the gravity term left a sustained oscillation")
            + f" (targets: {', '.join(sorted(fell or unstable))})",
        }
    if len(passed) == len(raised) and raised:
        return {
            "result": "A",
            "reason": "gravity feed-forward verified against PhysX and accepted on every raised-pose target: "
                      + ", ".join(f"{name} {comparisons[name]['palm_dz_mae_a_m']:.3f}->{comparisons[name]['palm_dz_mae_b_m']:.3f} m"
                                  for name in raised),
        }
    if improved:
        return {
            "result": "B",
            "reason": "the error improves but the acceptance criteria are not met on every raised-pose target",
        }
    return {"result": "C", "reason": "the gravity feed-forward did not improve the tracking error"}


# ------------------------------------------------------------------- figures


def stage_figures(_: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((OUTPUT / "summary.json").read_text())
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = summary["runs"]
    series = np.load(OUTPUT / "series" / "palm_error_series.npz")

    # 1. The palm-z error over time: where the droop comes from and when it goes.
    panels = [name for name in ("t2", "t3") if f"{name}-a" in runs and f"{name}-b" in runs]
    figure, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.0), squeeze=False)
    for axis, target in zip(axes[0], panels):
        for variant, colour in (("a", "#4477aa"), ("b", "#ee6677")):
            run = f"{target}-{variant}"
            sim_s = series[f"{run}__sim_s"]
            dz = 0.5 * (series[f"{run}__dz_left"] + series[f"{run}__dz_right"])
            axis.plot(sim_s, dz, colour, linewidth=1.4, label=f"variant {variant.upper()}")
        axis.axhline(ACCEPTANCE["palm_dz_mae_m"], color="black", linestyle="--", linewidth=1)
        axis.axhline(0.0, color="grey", linewidth=0.8)
        axis.set_xlim(0.0, float(series[f"{target}-a__sim_s"][-1]))
        axis.set_title(f"{target.upper()}: mean palm z error (target - measured)")
        axis.set_xlabel("sim time (s)")
        axis.set_ylabel("m")
        axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "palm_z_error_series.png", dpi=130)
    plt.close(figure)

    # 2. Acceptance metrics, A against B, per target.
    names = [name for name in sorted({item["target"] for item in runs.values()})]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    width = 0.35
    positions = np.arange(len(names))
    for axis, (getter, label) in zip(axes, (
        (lambda result: result["palm_dz_mae_m"], "palm z MAE (m)"),
        (lambda result: result["arm_joints"]["mae_rad"], "arm joint MAE (rad)"),
    )):
        for offset, variant in ((-width / 2, "a"), (width / 2, "b")):
            values = [getter(runs[f"{target}-{variant}"]) if f"{target}-{variant}" in runs else np.nan
                      for target in names]
            control = [target not in ACCEPTANCE_TARGETS for target in names]
            colours = [("#bbbbbb" if variant == "a" else "#dd9999") if is_control
                       else ("#4477aa" if variant == "a" else "#ee6677")
                       for is_control in control]
            axis.bar(positions + offset, values, width, label=f"variant {variant.upper()}", color=colours)
        if axis is axes[0]:
            axis.axhline(ACCEPTANCE["palm_dz_mae_m"], color="black", linestyle="--", linewidth=1)
        axis.set_xticks(positions, [f"{name.upper()}{'' if name in ACCEPTANCE_TARGETS else '*'}" for name in names])
        axis.set_ylabel(label)
        axis.legend()
    figure.suptitle("Acceptance targets T2, T3;  * controls (T0 zero gravity, T1 neutral hold, T4 band release)")
    axes[0].set_title("Palm-z error")
    axes[1].set_title("Joint-space error")
    figure.tight_layout()
    figure.savefig(figures / "acceptance_ab.png", dpi=130)
    plt.close(figure)

    # 3. The gravity term against the physics model and against the term it replaces.
    check = summary.get("gravity_model_check", {}).get("t2-b")
    p_term = summary.get("gravity_term_check", {}).get("t2-a")
    if check and p_term:
        joints = [item["joint"].replace("_joint", "") for item in check["joints"]]
        positions = np.arange(len(joints))
        figure, axis = plt.subplots(figsize=(12, 4.6))
        axis.bar(positions - 0.22, [item["recorded_mean_nm"] for item in check["joints"]], 0.22,
                 label="PhysX gravity term (T2-B)", color="#ee6677")
        axis.bar(positions, [item["modelled_mean_nm"] for item in check["joints"]], 0.22,
                 label="URDF gravity moment, sign-flipped (analytic)", color="#228833")
        axis.bar(positions + 0.22, [item["p_term_mean_nm"] for item in p_term["joints"]], 0.22,
                 label="PD torque applied in T2-A (at its own drooped pose)", color="#4477aa")
        axis.set_xticks(positions, joints, rotation=90, fontsize=6)
        axis.set_ylabel("N*m")
        axis.legend()
        axis.set_title("Gravity feed-forward against an independent model and against the PD term it replaces")
        figure.tight_layout()
        figure.savefig(figures / "gravity_validation.png", dpi=130)
        plt.close(figure)
    print(f"[figures] wrote {figures}")
    return 0


# ------------------------------------------------------------------ manifest


def stage_manifest(_: argparse.Namespace) -> int:
    inputs = []
    for path in (RECORDED_ROLLOUT, COMPARE_SCRIPT, VALIDATE_SCRIPT, DEFAULT_URDF,
                 REPO_ROOT / "src" / "humanoid_lab" / "simulators" / "isaac" / "service.py",
                 REPO_ROOT / "src" / "humanoid_lab" / "simulators" / "isaac" / "cli.py",
                 REPO_ROOT / "src" / "humanoid_lab" / "simulators" / "isaac" / "contracts.py",
                 REPO_ROOT / "dev.sh"):
        inputs.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path),
                       "bytes": path.stat().st_size})
    for name in TARGET_PROFILES.values():
        path = OUTPUT / "profiles" / name
        if path.exists():
            inputs.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path),
                           "bytes": path.stat().st_size})
    reference = OUTPUT / "profiles" / REFERENCE_NPZ
    if reference.exists():
        inputs.append({"path": str(reference.relative_to(REPO_ROOT)), "sha256": sha256(reference),
                       "bytes": reference.stat().st_size})
    runs = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        entry = {"name": run_dir.name}
        for name in ("isaac.metrics.json", "isaac.tracking.parquet", "command.txt", "run.log"):
            path = run_dir / name
            if path.exists():
                entry[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        runs.append(entry)
    gains_kp, gains_kd = deploy_gains()
    arm = BODY_JOINT_ORDER.index("left_elbow_joint")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
            "version": SCRIPT_VERSION,
        },
        "commands": [
            "python3 scripts/gravity-comp-ab.py profiles",
            "python3 scripts/gravity-comp-ab.py run --repeats 3",
            "data/venvs/hf-datasets/bin/python scripts/gravity-comp-ab.py analyse",
            "python3 scripts/gravity-comp-ab.py figures",
            "python3 scripts/gravity-comp-ab.py manifest",
        ],
        "inputs": inputs,
        "runs": runs,
        "constants": {
            "support_max_seconds": SUPPORT_MAX_SECONDS,
            "settle_margin_s": SETTLE_MARGIN_S,
            "durations_s": DURATIONS,
            "t2_source_frame": T2_FRAME,
            "t3_segment": list(T3_SEGMENT),
            "t3_stretch": T3_STRETCH,
            "t3_pre_roll_s": T3_PRE_ROLL_S,
            "t3_tail_hold_s": T3_TAIL_HOLD_S,
            "arm_joints": list(ARM_JOINTS),
            "waist_joints": list(WAIST_JOINTS),
            "acceptance": ACCEPTANCE,
            "pinned_controller_values": {
                "source": "src/humanoid_lab/controllers/sonic.py (copied from the pinned SONIC deployment)",
                "arm_kp_nm_per_rad": float(gains_kp[arm]),
                "arm_kd": float(gains_kd[arm]),
                "arm_effort_limit_nm": float(BODY_EFFORT_LIMIT_NM[arm]),
                "wrist_pitch_effort_limit_nm": float(BODY_EFFORT_LIMIT_NM[BODY_JOINT_ORDER.index("left_wrist_pitch_joint")]),
                "note": "read only; no gain, limit or controller parameter was changed by this experiment",
            },
            "standing_pose_max_arm_rad": float(max(abs(standing_pose()[name]) for name in BODY_JOINT_ORDER)),
        },
    }
    write_json(OUTPUT / "manifest.json", manifest)
    print(f"[manifest] wrote {OUTPUT / 'manifest.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", nargs="?", default="analyse",
                        choices=("profiles", "run", "analyse", "figures", "manifest"))
    parser.add_argument("--targets", default="t0,t1,t2,t3")
    parser.add_argument("--variants", default="a,b")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "profiles":
        return stage_profiles(args)
    if args.stage == "run":
        return stage_run(args)
    if args.stage == "analyse":
        return stage_analyse(args)
    if args.stage == "figures":
        return stage_figures(args)
    return stage_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
