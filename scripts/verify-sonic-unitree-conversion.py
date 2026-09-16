#!/usr/bin/env python3
"""Independent verification of the Unitree Dex3 conversion, written from scratch.

This deliberately does NOT import the conversion pipeline: it re-derives the
canonical 50 Hz reference from the raw parquet and compares it against the stored
artifacts, so a bug shared between the producer and the checker cannot hide.  It
is the executable form of the claim "the Unitree conversion is arithmetically
consistent"; run it after touching adapters, resampling, the canonical schema,
the encoder observation or the joint orders.

Usage (container paths; run inside the conversion environment):
  ./dev.sh sonic-verify-unitree [episode ...]


Reported checks
  1. source->canonical joint mapping (name based, independent implementation)
  2. resampling (30 Hz source -> 50 Hz canonical) on the arm channels
  3. synthetic lower body / root against the declared standing frame
  4. velocities: canonical finite difference vs independent recomputation, and
     against a 30 Hz source finite difference
  5. joint limits from configs/datasets/sonic/g1_joint_limits.json plus a
     sanity band on the whole trajectory
  6. hand channels: identity claim of the arm<->Dex3 motor order and the
     action[64:78] composition
  7. encoder observation: rebuild the 1751D block by block from the reference
     and compare with the stored encoder_observation.npz
  8. action_tokens.npz internal consistency (token dim, action composition,
     timestamps, frame index)
  9. measured observation.state vs commanded action (tracking sanity)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

RAW = Path("/data/datasets/first_tur_ham/unitree-g1-dex3")
LIMITS = Path("/workspace/humanoid-lab/configs/datasets/sonic/g1_joint_limits.json")
FPS = 50.0

BODY_ORDER = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

# source field name -> canonical joint name, written out literally (no string munging)
ARM_MAP = {
    "kLeftShoulderPitch": "left_shoulder_pitch_joint",
    "kLeftShoulderRoll": "left_shoulder_roll_joint",
    "kLeftShoulderYaw": "left_shoulder_yaw_joint",
    "kLeftElbow": "left_elbow_joint",
    "kLeftWristRoll": "left_wrist_roll_joint",
    "kLeftWristPitch": "left_wrist_pitch_joint",
    "kLeftWristYaw": "left_wrist_yaw_joint",
    "kRightShoulderPitch": "right_shoulder_pitch_joint",
    "kRightShoulderRoll": "right_shoulder_roll_joint",
    "kRightShoulderYaw": "right_shoulder_yaw_joint",
    "kRightElbow": "right_elbow_joint",
    "kRightWristRoll": "right_wrist_roll_joint",
    "kRightWristPitch": "right_wrist_pitch_joint",
    "kRightWristYaw": "right_wrist_yaw_joint",
}
HAND_MAP = {
    "kLeftHandThumb0": "left_hand_thumb_0_joint",
    "kLeftHandThumb1": "left_hand_thumb_1_joint",
    "kLeftHandThumb2": "left_hand_thumb_2_joint",
    "kLeftHandMiddle0": "left_hand_middle_0_joint",
    "kLeftHandMiddle1": "left_hand_middle_1_joint",
    "kLeftHandIndex0": "left_hand_index_0_joint",
    "kLeftHandIndex1": "left_hand_index_1_joint",
    "kRightHandThumb0": "right_hand_thumb_0_joint",
    "kRightHandThumb1": "right_hand_thumb_1_joint",
    "kRightHandThumb2": "right_hand_thumb_2_joint",
    "kRightHandIndex0": "right_hand_index_0_joint",
    "kRightHandIndex1": "right_hand_index_1_joint",
    "kRightHandMiddle0": "right_hand_middle_0_joint",
    "kRightHandMiddle1": "right_hand_middle_1_joint",
}
#: canonical Dex3 order the pipeline declares (side specific, mirrored)
LEFT_HAND_CANON = ["thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
                   "middle_0_joint", "middle_1_joint", "index_0_joint", "index_1_joint"]
RIGHT_HAND_CANON = ["thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
                    "index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint"]

ARM_NAMES_SONIC = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def load_episode(dataset: Path, episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return action, state and timestamps of one LeRobot v3 episode."""
    rows = []
    for path in sorted((dataset / "meta/episodes").glob("chunk-*/*.parquet")):
        rows.extend(r for r in pq.read_table(path).to_pylist() if int(r["episode_index"]) == episode)
    assert len(rows) == 1, rows
    row = rows[0]
    chunk = int(row["data/chunk_index"])
    file_index = int(row["data/file_index"])
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    table = pq.read_table(
        dataset / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet",
        columns=["episode_index", "timestamp", "action", "observation.state"],
    ).slice(start, stop - start)
    assert all(int(v) == episode for v in table.column("episode_index").to_pylist())
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    ts = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64).reshape(-1)
    return action, state, ts


def uniform_timeline(ts: np.ndarray, fps: float = FPS) -> np.ndarray:
    count = int(np.floor((ts[-1] - ts[0]) * fps + 1e-9)) + 1
    return ts[0] + np.arange(count, dtype=np.float64) / fps


def interp(ts: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(target, ts, values[:, i]) for i in range(values.shape[1])], axis=1)


def verify(episode: int) -> None:
    dataset = RAW / "G1_Dex3_ObjectPlacement_Dataset"
    info = json.loads((dataset / "meta/info.json").read_text())
    action, state, ts = load_episode(dataset, episode)
    names = list(info["features"]["action"]["names"][0])
    print(f"\n=== episode {episode}: {len(action)} source frames @ {info['fps']} fps, "
          f"{ts[-1]-ts[0]:.3f} s ===")

    # find the pilot dir for this episode
    pilot_root = Path("/data/datasets/first_tur_processed/sonic_v1_1/pilots")
    candidates = sorted(pilot_root.glob(f"unitree_*_ep{episode:03d}/*"))
    if not candidates:
        check(f"ep{episode} pilot exists", False, "no pilot directory")
        return
    pilot = candidates[-1]
    ref = np.load(pilot / "reference.npz")
    obs = np.load(pilot / "encoder_observation.npz")
    print(f"pilot: {pilot}")

    # ---- 1. joint mapping -------------------------------------------------
    idx = {n: i for i, n in enumerate(names)}
    src_index = {src: idx[src] for src in ARM_MAP}
    canonical = np.tile(np.zeros(29), (len(action), 1))
    for src, dst in ARM_MAP.items():
        canonical[:, BODY_ORDER.index(dst)] = action[:, src_index[src]]
    target = uniform_timeline(ts)
    arm_target = interp(ts, canonical, target)

    stored = np.asarray(ref["joint_pos"], dtype=np.float64)
    arm_ids = [BODY_ORDER.index(n) for n in ARM_NAMES_SONIC]
    err = np.abs(stored[:, arm_ids] - arm_target[:, arm_ids])
    check("1. arm channel mapping (source name -> canonical index)",
          err.max() < 1e-5,
          f"max |stored - independent interp| = {err.max():.3e} rad "
          f"(worst joint {ARM_NAMES_SONIC[int(np.argmax(err.max(axis=0)))]})")

    # a wrong permutation would put a large value on a different joint; report the
    # best-matching source channel per canonical slot to expose any swap
    mismatched = []
    for k, canonical_name in enumerate(ARM_NAMES_SONIC):
        column = stored[:, BODY_ORDER.index(canonical_name)]
        best = min(
            ARM_MAP,
            key=lambda s: np.abs(interp(ts, action[:, [idx[s]]], target)[:, 0] - column).max(),
        )
        if ARM_MAP[best] != canonical_name:
            mismatched.append((canonical_name, best))
    check("1b. no cross-joint swap (best matching source channel)",
          not mismatched, f"mismatched slots: {mismatched}" if mismatched else "every slot matches its own source channel")

    # ---- 2. resampling/rate ----------------------------------------------
    dt = np.diff(np.asarray(ref["timestamps"], dtype=np.float64))
    check("2. canonical timeline is uniform 50 Hz",
          len(dt) > 0 and np.abs(dt - 1 / FPS).max() < 1e-9,
          f"frames={len(dt)+1}, dt min={dt.min():.9f} max={dt.max():.9f}")
    expected_frames = int(np.floor((ts[-1] - ts[0]) * FPS + 1e-9)) + 1
    check("2b. frame count matches the source duration",
          len(ref["timestamps"]) == expected_frames,
          f"{len(ref['timestamps'])} == floor(({ts[-1]:.3f}-{ts[0]:.3f})*50)+1 = {expected_frames}")

    # ---- 3. synthetic lower body / root ----------------------------------
    frozen_ids = [i for i, n in enumerate(BODY_ORDER) if n not in ARM_NAMES_SONIC]
    spread = stored[:, frozen_ids].max(axis=0) - stored[:, frozen_ids].min(axis=0)
    check("3. lower body / waist frozen for the whole episode",
          spread.max() < 1e-9,
          f"max temporal spread over {len(frozen_ids)} joints = {spread.max():.3e} rad")
    root_pos = np.asarray(ref["body_pos"], dtype=np.float64)
    root_quat = np.asarray(ref["body_quat_wxyz"], dtype=np.float64)
    check("3b. root is static and upright",
          root_pos.std(axis=0).max() < 1e-9 and np.abs(root_quat - np.array([1.0, 0, 0, 0])).max() < 1e-9,
          f"root z = {root_pos[0,2]:.6f} m, quat = {root_quat[0]}")

    # standing pose must equal the deployment's default angles.
    # NOTE: stored/BODY_ORDER above are the SONIC *reference* order; the
    # deployment dict is keyed by joint name, so index it by name, never by a
    # position taken from the other vocabulary (that was a checker bug once).
    sys.path.insert(0, "/workspace/humanoid-lab/src")
    from humanoid_lab.controllers.sonic import DEFAULT_STANDING_POSE_RAD
    expected_standing = np.array([DEFAULT_STANDING_POSE_RAD[BODY_ORDER[i]] for i in frozen_ids])
    # The stored canonical pose is float32, so compare at float32 resolution: a
    # 1e-9 bound fails on ~3e-8 of pure representation error while catching
    # nothing a real mapping bug would produce.
    float32_eps = float(np.finfo(np.float32).eps)
    check("3c. frozen frame equals the deployment's default standing pose",
          np.abs(stored[0, frozen_ids] - expected_standing).max() < 10 * float32_eps,
          f"max diff = {np.abs(stored[0, frozen_ids] - expected_standing).max():.3e} rad "
          f"(float32 eps = {float32_eps:.2e})")

    # ---- 4. velocities ----------------------------------------------------
    vel = np.asarray(ref["joint_vel"], dtype=np.float64)
    independent = np.zeros_like(stored)
    independent[:-1] = np.diff(stored, axis=0) * FPS
    independent[-1] = independent[-2]
    check("4. joint_vel == finite difference of joint_pos at 50 Hz",
          np.abs(vel - independent).max() < 1e-6,
          f"max diff = {np.abs(vel - independent).max():.3e} rad/s")

    # velocity vs the raw 30 Hz source derivative.
    # The canonical 50 Hz velocity is a backward difference of the *interpolated*
    # trajectory; on a piecewise-linear resample it is a one-sample-delayed,
    # attenuated copy of the source derivative, so a per-frame comparison is
    # dominated by the estimator difference, not by an indexing bug.  Compare
    # integrated (low-pass) signals instead, and cross-correlate to pin lag.
    src_vel = np.zeros_like(canonical)
    src_vel[:-1] = np.diff(canonical, axis=0) * float(info["fps"])
    src_vel[-1] = src_vel[-2]
    src_vel_target = interp(ts, src_vel, target)
    a = src_vel_target[:, arm_ids].ravel()
    b = vel[:, arm_ids].ravel()
    corr = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    lag = int(np.argmax(corr) - (len(a) - 1))
    rel_lag = lag / FPS
    scale = float(np.dot(a, b) / np.dot(a, a))
    check("4b. canonical velocity matches the source derivative (gain + lag)",
          0.5 < scale < 1.5 and abs(rel_lag) <= 0.03,
          f"best-fit gain = {scale:.3f} (1.0 expected), lag = {rel_lag*1000:+.1f} ms "
          f"(one 50 Hz sample = 20 ms; interpolation delay is expected to be < 1 sample)")

    # integrated (position-equivalent) comparison: a wrong scale would drift
    cum_err = np.abs(np.cumsum(vel[:, arm_ids], axis=0) / FPS -
                     np.cumsum(src_vel_target[:, arm_ids], axis=0) / FPS)
    horizon = cum_err[-1]
    check("4c. integrated arm displacement agrees with the source",
          float(horizon.max()) < 0.05,
          f"max |integral(v_canonical) - integral(v_source)| = {horizon.max():.4f} rad over "
          f"{len(target)/FPS:.1f} s")

    # ---- 5. joint limits --------------------------------------------------
    limits = json.loads(LIMITS.read_text())["joints"]
    worst = (0.0, None)
    for i, name in enumerate(BODY_ORDER):
        lo, hi = limits[name]["lower"], limits[name]["upper"]
        excess = max(0.0, float(lo - stored[:, i].min()), float(stored[:, i].max() - hi))
        if excess > worst[0]:
            worst = (excess, name)
    check("5. every canonical joint stays inside the pinned limits",
          worst[0] == 0.0,
          f"worst excess = {worst[0]:.3e} rad" + (f" on {worst[1]}" if worst[1] else ""))

    # independent sanity: arm values must be plausible radians, not degrees or normalized
    arm_absmax = float(np.abs(stored[:, arm_ids]).max())
    check("5b. arm values are plausible radians (not degrees / normalized)",
          0.0 < arm_absmax < np.pi and float(np.abs(stored[:, arm_ids]).max()) > 0.01,
          f"max |arm| = {arm_absmax:.3f} rad")

    # ---- 6. hands ---------------------------------------------------------
    hand_src = {src: action[:, idx[src]] for src in HAND_MAP}
    for side, canon in (("left", LEFT_HAND_CANON), ("right", RIGHT_HAND_CANON)):
        stored_hand = np.asarray(ref[f"{side}_hand_joints"], dtype=np.float64)
        expected = np.stack(
            [interp(ts, hand_src[f"k{side.title()}Hand{part.replace('_','').replace('joint','').title()}"][:, None], target)[:, 0]
             for part in canon], axis=1)
        diff = np.abs(stored_hand - expected).max()
        check(f"6. {side} hand mapping ({canon[0]}..{canon[-1]})",
              diff < 1e-5, f"max diff vs independent source remap = {diff:.3e} rad")
        check(f"6b. {side} hand channels carry real motion",
              float(np.ptp(stored_hand, axis=0).max()) > 1e-3,
              f"per-channel ptp = {np.round(np.ptp(stored_hand, axis=0), 4).tolist()}")

    # ---- 7. encoder observation ------------------------------------------
    reference = np.asarray(ref["joint_pos"], dtype=np.float64)
    stored_obs = np.asarray(obs["observation"], dtype=np.float32)
    frames = len(reference)
    offsets = np.arange(0, 10 * 5, 5)
    idx_future = np.minimum(np.arange(frames)[:, None] + offsets[None, :], frames - 1)
    rebuilt = np.zeros((frames, 1751), dtype=np.float32)
    rebuilt[:, 0] = 0.0
    rebuilt[:, 4:294] = reference[idx_future].reshape(frames, -1)
    rebuilt[:, 294:584] = np.asarray(ref["joint_vel"], dtype=np.float64)[idx_future].reshape(frames, -1)
    quat = np.asarray(ref["body_quat_wxyz"], dtype=np.float64)
    # heading-relative with base = reference root at the current frame:
    # identity quaternion here because the root is upright, so the block must be
    # the raw 6D rotation of the identity = [1,0,0,0,1,0] repeated
    base = quat[:, None, :]
    fut = quat[idx_future]
    # quat: left = heading_quat(base); relative = conj(left) * fut
    def heading_quat(q):
        x = q[:, :, 0] ** 2 - q[:, :, 1] ** 2 - q[:, :, 2] ** 2 + q[:, :, 3] ** 2
        y = 2 * (q[:, :, 0] * q[:, :, 3] + q[:, :, 1] * q[:, :, 2])
        psi = np.arctan2(y, x)
        out = np.zeros_like(q)
        out[:, :, 0] = np.cos(psi / 2)
        out[:, :, 3] = np.sin(psi / 2)
        return out
    left = heading_quat(np.broadcast_to(base, fut.shape))
    conj = left.copy(); conj[:, :, 1:] *= -1
    a, b = conj, fut
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    rel = np.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                    aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), axis=-1)
    n = np.linalg.norm(rel, axis=-1, keepdims=True); rel = np.divide(rel, n, out=np.zeros_like(rel), where=n > 1e-12)
    w, x, y, z = rel[..., 0], rel[..., 1], rel[..., 2], rel[..., 3]
    R = np.stack((np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)), -1),
                  np.stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)), -1),
                  np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)), -2)
    rebuilt[:, 584:644] = R[..., :, :2].reshape(frames, 60).astype(np.float32)
    check("7. encoder observation rebuilt block-by-block from the reference",
          np.abs(rebuilt - stored_obs).max() < 1e-6,
          f"max abs diff over {stored_obs.shape} = {np.abs(rebuilt - stored_obs).max():.3e}")
    nonzero_blocks = {
        "mode": int(np.count_nonzero(stored_obs[:, 0:4])),
        "joint_pos": int(np.count_nonzero(stored_obs[:, 4:294])),
        "joint_vel": int(np.count_nonzero(stored_obs[:, 294:584])),
        "orientation": int(np.count_nonzero(stored_obs[:, 584:644])),
        "rest": int(np.count_nonzero(stored_obs[:, 644:1751])),
    }
    check("7b. only the four G1 mode observations are populated",
          nonzero_blocks["rest"] == 0,
          f"non-zero counts per block = {nonzero_blocks}")
    check("7c. mode block is [mode_id, 0, 0, 0] and not one-hot",
          np.array_equal(stored_obs[:, 0:4], np.zeros((frames, 4), dtype=np.float32)),
          f"unique rows = {np.unique(stored_obs[:, 0:4], axis=0)}")

    # ---- 8. tokens --------------------------------------------------------
    tok_path = pilot / "action_tokens.npz"
    if tok_path.is_file():
        tok = np.load(tok_path)
        motion = np.asarray(tok["motion_token"])
        act = np.asarray(tok["action"])
        check("8. action_tokens shapes",
              motion.shape == (frames, 64) and act.shape == (frames, 78),
              f"motion_token {motion.shape}, action {act.shape}")
        composed = np.concatenate([motion, np.asarray(ref["left_hand_joints"], dtype=np.float32),
                                   np.asarray(ref["right_hand_joints"], dtype=np.float32)], axis=1)
        check("8b. action = [motion_token | left hand | right hand]",
              np.abs(composed - act).max() < 1e-6,
              f"max diff = {np.abs(composed - act).max():.3e}")
        check("8c. timestamps and frame index line up",
              np.array_equal(np.asarray(tok["frame_index"]), np.arange(frames)) and
              np.abs(np.asarray(tok["timestamp"]) - np.asarray(ref["timestamps"])).max() < 1e-12,
              f"frame_index 0..{frames-1}, dt = {np.diff(np.asarray(tok['timestamp'])).mean():.9f} s")
        check("8d. tokens are finite and non-degenerate",
              bool(np.isfinite(motion).all()) and float(np.linalg.norm(motion, axis=1).std()) > 1e-3,
              f"token norm mean={np.linalg.norm(motion,axis=1).mean():.4f} "
              f"std={np.linalg.norm(motion,axis=1).std():.4f} min={np.linalg.norm(motion,axis=1).min():.4f}")
    else:
        check("8. action_tokens.npz present", False, f"missing in {pilot}")

    # ---- 9. commanded action vs measured state ---------------------------
    track = np.abs(interp(ts, state[:, [idx[s] for s in ARM_MAP]], target) -
                   interp(ts, action[:, [idx[s] for s in ARM_MAP]], target))
    check("9. recorded state tracks the commanded action (arm channels)",
          float(track.mean()) < 0.15,
          f"mean |state - action| = {track.mean():.4f} rad, p95 = {np.percentile(track, 95):.4f} rad, "
          f"max = {track.max():.4f} rad")


def main() -> int:
    episodes = [int(v) for v in sys.argv[1:]] or [74]
    for episode in episodes:
        verify(episode)
    failures = [name for name, ok, _ in results if not ok]
    print("\n" + "=" * 72)
    print(f"{len(results) - len(failures)}/{len(results)} checks passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
