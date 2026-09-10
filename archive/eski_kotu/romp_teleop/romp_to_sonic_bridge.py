#!/usr/bin/env python3
"""Stage 2 (SONIC env): convert ROMP SMPL -> SONIC Protocol v3 and publish.

Runs where `gear_sonic` is importable (the container, e.g.):
    docker exec -it humanoid-lab-dev bash -lc \
      'source /opt/humanoid-lab/entrypoint.sh && use-sonic-sim && \
       python /workspace/humanoid-lab/romp_teleop/romp_to_sonic_bridge.py \
         --sonic-root /opt/src/sonic --play motion.npz --fps 50 --loop'

Inputs (one of):
  --play motion.npz        precomputed sequence from romp_pose_streamer.py
  --subscribe              live raw SMPL on tcp://localhost:5558

Output: SONIC v3 'pose' message on tcp://*:5556 (SMPL encoder mode 2).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--play", default=None, help="NPZ from romp_pose_streamer.py")
    p.add_argument("--subscribe", action="store_true",
                   help="Live raw SMPL from stage 1 on port 5558")
    p.add_argument("--raw-port", type=int, default=5558)
    p.add_argument("--port", type=int, default=5556, help="SONIC v3 publish port")
    p.add_argument("--fps", type=float, default=50.0, help="Publish rate (SONIC 50 Hz)")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--smooth", type=float, default=0.7,
                   help="EMA history weight (0=off, 0.6-0.85 smoother)")
    p.add_argument("--root-mode", choices=["yaw", "full", "identity"], default="yaw")
    p.add_argument("--no-flip-x", dest="flip_x", action="store_false", default=True,
                   help="Disable the ROMP Y-down -> SMPL Y-up root flip")
    p.add_argument("--sonic-root", default=os.environ.get("SONIC_ROOT", "/opt/src/sonic"))
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="Convert + print, do not open ZMQ")
    p.add_argument("--record-debug", default=None, help="Save debug NPZ here")
    return p.parse_args()


def slerp(q0, q1, t):
    q0 = q0.astype(np.float64)
    q1 = q1.astype(np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return (q / (np.linalg.norm(q) + 1e-12)).astype(np.float32)
    th0 = math.acos(max(-1.0, min(1.0, d)))
    th = th0 * t
    q2 = q1 - d * q0
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    return (math.cos(th) * q0 + math.sin(th) * q2).astype(np.float32)


def resample(joints, quats, src_times, out_fps):
    """Resample converted (joints, quats) onto a uniform out_fps grid."""
    joints = np.asarray(joints, dtype=np.float32)
    quats = np.asarray(quats, dtype=np.float32)
    src_times = np.asarray(src_times, dtype=np.float64)
    n = len(joints)
    if n < 2 or src_times[-1] <= src_times[0]:
        return joints.copy(), quats.copy(), np.arange(n) / out_fps

    dur = float(src_times[-1] - src_times[0])
    n_out = max(2, int(round(dur * out_fps)) + 1)
    t_out = np.clip(src_times[0] + np.arange(n_out) / out_fps,
                    src_times[0], src_times[-1])

    idx = np.clip(np.searchsorted(src_times, t_out, side="right") - 1,
                  0, n - 2).astype(np.int64)
    t0 = src_times[idx]
    t1 = src_times[idx + 1]
    alpha = np.where(t1 > t0, (t_out - t0) / np.maximum(t1 - t0, 1e-9), 0.0).astype(np.float32)

    j0 = joints[idx]
    j1 = joints[idx + 1]
    j_out = ((1.0 - alpha)[:, None, None] * j0
             + alpha[:, None, None] * j1).astype(np.float32)
    q_out = np.stack(
        [slerp(quats[idx[i]], quats[idx[i] + 1], float(alpha[i]))
         for i in range(n_out)]
    ).astype(np.float32)
    return j_out, q_out, t_out


def main():
    args = parse_args()

    from romp_sonic.coordinates import RompToSmpl24
    from romp_sonic.protocol import V3Publisher
    from romp_sonic.smoothing import EmaSmoother

    conv = RompToSmpl24(sonic_root=args.sonic_root, root_mode=args.root_mode,
                        flip_x=args.flip_x)
    smoother = EmaSmoother(args.smooth)
    print(f"[stage2] sonic_root={args.sonic_root} root_mode={args.root_mode} "
          f"smooth={args.smooth} out_fps={args.fps}")

    # ---- Load input sequence ----
    if args.play:
        data = np.load(args.play, allow_pickle=True)
        thetas = np.asarray(data["smpl_thetas"], dtype=np.float32)
        raw_joints = np.asarray(data["joints24_raw"], dtype=np.float32)
        fps_src = float(data["fps"][0]) if "fps" in data else 30.0
        print(f"[stage2] loaded {len(thetas)} frames from {args.play} (src fps {fps_src:.2f})")
    elif args.subscribe:
        thetas, raw_joints, fps_src = None, None, args.fps
    else:
        raise SystemExit("provide --play <npz> or --subscribe")

    publisher = None if args.dry_run else V3Publisher(
        port=args.port, topic="pose", sonic_root=args.sonic_root)

    debug = {"raw_joints": [], "sonic_joints": [], "root_raw": [],
             "body_quat": [], "frames": []}

    def run_sequence(thetas, src_times, base_index=0):
        son_j, son_q = [], []
        for t in range(len(thetas)):
            theta = thetas[t]
            j, q = conv.convert(theta[:3], theta[3:])
            j, q = smoother(j, q)
            son_j.append(j)
            son_q.append(q)
            debug["raw_joints"].append(raw_joints[t] if raw_joints is not None else np.zeros((24, 3), np.float32))
            debug["sonic_joints"].append(j)
            debug["root_raw"].append(theta[:3])
            debug["body_quat"].append(q)
            debug["frames"].append(base_index + t)

        son_j = np.asarray(son_j, dtype=np.float32)
        son_q = np.asarray(son_q, dtype=np.float32)
        j_out, q_out, t_out = resample(son_j, son_q, src_times, args.fps)

        if args.dry_run:
            for i in range(len(j_out)):
                if i % max(1, len(j_out) // 10) == 0:
                    print(f"[dry-run] i={i} smpl_joints{j_out[i].shape} "
                          f"range=[{j_out[i].min():.3f},{j_out[i].max():.3f}] "
                          f"quat_norm={np.linalg.norm(q_out[i]):.4f}")
            return len(j_out)

        sent = 0
        t_start = time.time()
        for i in range(len(j_out)):
            idx = publisher.publish(j_out[i], q_out[i])
            sent += 1
            if sent % int(max(1, args.fps)) == 0:
                lag = time.time() - t_start - sent / args.fps
                print(f"\r[stage2] {sent}/{len(j_out)} sent  "
                      f"frame_index={idx}  lag={lag*1000:+.0f}ms", end="", flush=True)
            # pace to real time
            target = t_start + (sent) / args.fps
            dt = target - time.time()
            if dt > 0:
                time.sleep(dt)
            if args.max_frames and sent >= args.max_frames:
                break
        print()
        return sent

    try:
        if args.play:
            src_times = np.arange(len(thetas)) / max(fps_src, 1e-6)
            total = 0
            while True:
                total += run_sequence(thetas, src_times, base_index=total)
                if not args.loop or args.max_frames:
                    break
                print("[stage2] loop restart")
        else:  # subscribe (live)
            import pickle
            import zmq
            ctx = zmq.Context()
            sub = ctx.socket(zmq.SUB)
            sub.connect(f"tcp://localhost:{args.raw_port}")
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            print(f"[stage2] subscribed raw SMPL on tcp://localhost:{args.raw_port}")
            n = 0
            t_start = time.time()
            while True:
                msg = sub.recv()
                d = pickle.loads(msg)
                j, q = conv.convert(d["smpl_thetas"][:3], d["smpl_thetas"][3:])
                j, q = smoother(j, q)
                publisher.publish(j, q)
                n += 1
                if n % int(max(1, args.fps)) == 0:
                    print(f"\r[stage2] live {n} frames", end="", flush=True)
                if args.max_frames and n >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print("\n[stage2] interrupted")
    finally:
        if publisher is not None:
            publisher.close()

    if args.record_debug and debug["frames"]:
        np.savez_compressed(
            args.record_debug,
            raw_joints=np.asarray(debug["raw_joints"], dtype=np.float32),
            sonic_joints=np.asarray(debug["sonic_joints"], dtype=np.float32),
            root_orient_raw=np.asarray(debug["root_raw"], dtype=np.float32),
            body_quat_sonic=np.asarray(debug["body_quat"], dtype=np.float32),
            frame_index=np.asarray(debug["frames"], dtype=np.int64),
        )
        print(f"[stage2] wrote debug to {args.record_debug}")


if __name__ == "__main__":
    main()
