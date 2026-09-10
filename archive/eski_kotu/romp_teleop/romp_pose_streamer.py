#!/usr/bin/env python3
"""Stage 1 (ROMP env): monocular video / webcam -> ROMP -> SMPL-24 motion.

Runs in the ROMP virtualenv (~/romp-venv). Produces, per frame:
    smpl_thetas (72,)  [global_orient(3) | body_pose(69)]
    joints24_raw (24,3) (ROMP SMPL-24, for debug/overlay)
plus frame_index / timestamp.

Modes:
  * precompute (default when --save-smpl given): write an .npz sequence
  * live: publish each frame's raw SMPL over a local ZMQ (--publish, port 5558)
    for the Stage-2 bridge to convert + forward to SONIC.

Nothing here depends on SONIC; SONIC conversion happens in Stage 2.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from romp_sonic.utils import SMPL24_EDGES, log, select_largest_person  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0",
                   help="Webcam index (0) or path to a video file")
    p.add_argument("--save-smpl", default=None,
                   help="Write the per-frame SMPL sequence to this .npz")
    p.add_argument("--publish", action="store_true",
                   help="Also publish raw SMPL frames on tcp://*:5558")
    p.add_argument("--port", type=int, default=5558)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--seconds", type=float, default=0.0,
                   help="Stop after N seconds (0 = unlimited, useful for webcam)")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--fps", type=float, default=0.0,
                   help="Override playback fps for video (0 = use source fps)")
    p.add_argument("--device", type=int, default=0, help="GPU id, -1 = CPU")
    p.add_argument("--person", type=int, default=-1,
                   help="Fixed person id, -1 = select largest each frame")
    p.add_argument("--show", action="store_true",
                   help="Show video + ROMP-24 skeleton overlay")
    p.add_argument("--debug-coordinates", action="store_true",
                   help="Print landmark vectors to reason about the axes")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def open_source(source: str, width: int, height: int):
    is_cam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_cam else source,
                           cv2.CAP_V4L2 if is_cam else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {source}")
    if is_cam:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, is_cam, float(fps)


def draw_overlay(frame, pj2d, color=(0, 255, 0), thr=0.0):
    if pj2d is None:
        return frame
    pts = pj2d[:24]
    h, w = frame.shape[:2]
    for (a, b) in SMPL24_EDGES:
        pa, pb = pts[a], pts[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 color, 2, cv2.LINE_AA)
    for (x, y) in pts:
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()

    log("[stage1] importing ROMP ...")
    import romp
    settings = romp.main.default_settings
    settings.mode = "webcam" if args.source.isdigit() else "video"
    settings.GPU = args.device
    settings.calc_smpl = True
    settings.render_mesh = False
    settings.show = False
    settings.show_largest = False  # we need all people to pick the largest
    settings.temporal_optimize = False
    model = romp.ROMP(settings)
    log("[stage1] ROMP ready")

    cap, is_cam, src_fps = open_source(args.source, args.width, args.height)
    fps = args.fps if args.fps > 0 else src_fps
    log(f"[stage1] source={args.source} camera={is_cam} fps={fps:.2f}")

    publisher = None
    if args.publish:
        import zmq
        ctx = zmq.Context()
        publisher = ctx.socket(zmq.PUB)
        publisher.bind(f"tcp://*:{args.port}")
        log(f"[stage1] publishing raw SMPL on tcp://*:{args.port}")

    thetas_all, joints_all, idx_all, ts_all = [], [], [], []
    t0 = time.time()
    n_read = 0
    n_kept = 0
    last_log = t0
    fps_meter = 0.0
    meter_t = t0
    meter_n = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_cam:
                    time.sleep(0.01)
                    continue
                break  # end of video
            n_read += 1
            if n_read - 1 < args.start_frame:
                continue

            outputs = model(frame)
            if outputs is None:
                continue

            person = args.person
            if person < 0:
                person = select_largest_person(outputs["cam"])
            if person >= len(outputs["smpl_thetas"]):
                continue

            thetas = np.asarray(outputs["smpl_thetas"][person], dtype=np.float32)
            joints = np.asarray(outputs["joints"][person], dtype=np.float32)
            joints24 = joints[:24]

            thetas_all.append(thetas)
            joints_all.append(joints24)
            idx_all.append(n_kept)
            ts_all.append(time.time() - t0)
            n_kept += 1

            if args.debug_coordinates and n_kept <= 10:
                p = joints24
                log(f"[coords] frame {n_kept} pelvis={p[0].round(3)} "
                    f"Lsh-Rsh={(p[16]-p[17]).round(3)} head-pelvis={(p[15]-p[0]).round(3)} "
                    f"Lhip-pelvis={(p[1]-p[0]).round(3)} Lankle-pelvis={(p[7]-p[0]).round(3)}")

            if publisher is not None:
                publisher.send(pickle.dumps({
                    "smpl_thetas": thetas, "joints24_raw": joints24,
                    "frame_index": n_kept - 1, "timestamp": ts_all[-1],
                }))

            meter_n += 1
            now = time.time()
            if now - meter_t >= 1.0:
                fps_meter = meter_n / (now - meter_t)
                meter_t, meter_n = now, 0

            if args.show:
                disp = frame.copy()
                pj = outputs.get("pj2d_org", outputs.get("pj2d"))
                if pj is not None:
                    draw_overlay(disp, np.asarray(pj[person]))
                cv2.putText(disp, f"ROMP {fps_meter:4.1f}fps frame {n_kept}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("ROMP streamer", disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if now - last_log > 2:
                last_log = now
                log(f"[stage1] frame {n_kept} read {n_read} romp {fps_meter:.1f} fps")

            if args.max_frames and n_kept >= args.max_frames:
                break
            if args.seconds and (time.time() - t0) >= args.seconds:
                break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if publisher is not None:
            publisher.close(0)
            ctx.term()

    log(f"[stage1] collected {n_kept} frames in {time.time()-t0:.1f}s")

    if args.save_smpl and thetas_all:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_smpl)), exist_ok=True)
        np.savez_compressed(
            args.save_smpl,
            smpl_thetas=np.stack(thetas_all).astype(np.float32),
            joints24_raw=np.stack(joints_all).astype(np.float32),
            frame_index=np.asarray(idx_all, dtype=np.int64),
            timestamps=np.asarray(ts_all, dtype=np.float64),
            fps=np.array([fps], dtype=np.float32),
            source=np.array([str(args.source)]),
        )
        log(f"[stage1] wrote {args.save_smpl} "
            f"({len(thetas_all)} frames, thetas {np.stack(thetas_all).shape})")


if __name__ == "__main__":
    main()
