#!/usr/bin/env python3
"""Extract probe frames from the local G1 fruit datasets.

Run with the lerobot-viz venv (needs av + PIL + pyarrow):
    /home/aksoy-msi/code/humanoid-lab-main/data/venvs/lerobot-viz/bin/python extract_frames.py

Each fruit dataset carries exactly one instruction ("Pick up the <fruit> and
place it on the plate"), so the same frame can be paired later with any of the
four instructions, matched or deliberately mismatched.
"""

import argparse
import json
import pathlib

import av
from PIL import Image

DATASETS = {
    "apple": "g1-pick-apple",
    "pear": "g1-pick-pear",
    "grapes": "g1-pick-grapes",
    "starfruit": "g1-pick-starfruit",
}


def first_frame(video_path, frame_index):
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_index:
                return Image.fromarray(frame.to_ndarray(format="rgb24"))
    raise RuntimeError(f"{video_path} has no frame {frame_index}")


def instruction_of(meta_dir):
    with open(meta_dir / "episodes.jsonl") as f:
        return json.loads(f.readline())["tasks"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/aksoy-msi/code/humanoid-lab-main/data/datasets/groot/g1-fruits")
    ap.add_argument("--out", default="/home/aksoy-msi/code/humanoid-lab-main/data/outputs/unifolm-wla-probe/frames")
    ap.add_argument("--episodes", type=int, default=4, help="episodes per fruit")
    ap.add_argument("--stride", type=int, default=37, help="episode spacing to sample varied scenes")
    ap.add_argument("--frame-index", type=int, default=0, help="frame within each episode")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for fruit, dataset in DATASETS.items():
        meta = root / dataset / "meta"
        instruction = instruction_of(meta)
        video_dir = root / dataset / "videos" / "chunk-000" / "observation.images.ego_view"
        for k in range(args.episodes):
            episode = k * args.stride
            video = video_dir / f"episode_{episode:06d}.mp4"
            if not video.exists():
                continue
            image = first_frame(video, args.frame_index)
            sample_id = f"{fruit}_ep{episode:06d}_f{args.frame_index}"
            path = out / f"{sample_id}.png"
            image.save(path)
            manifest.append(
                {
                    "id": sample_id,
                    "image": str(path),
                    "fruit": fruit,
                    "dataset": dataset,
                    "episode": episode,
                    "frame_index": args.frame_index,
                    "instruction": instruction,
                    "resolution": list(image.size),
                }
            )
            print(f"{sample_id}: {image.size} {instruction}")
    with open(out / "manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(manifest)} frames + manifest.jsonl to {out}")


if __name__ == "__main__":
    main()
