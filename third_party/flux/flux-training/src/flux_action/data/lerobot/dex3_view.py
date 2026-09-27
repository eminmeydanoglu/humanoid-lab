"""Map-style FLUX3 view over the validated Dex3 index and original videos."""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ...data.droid.video import decode_window
from ...data.windows import valid_window_spans
from .index import ROWS_FILENAME

CAMERA = "observation.images.cam_left_high"


class Dex3Flux3View(Dataset):
    """Each index addresses one complete 1-observation / 32-action training window.

    The existing index owns the repaired Parquet row references and episode split. The
    memory-mapped rows are its validated cache of the original Parquet values; video
    frames are sought from the original MP4s on demand. No statistics are estimated.
    """

    def __init__(self, source_root, index_dir, *, split: str, decoder: str = "pyav"):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        self.source_root = Path(source_root)
        self.index_dir = Path(index_dir)
        self.manifest = json.loads((self.index_dir / "manifest.json").read_text())
        manifest = self.manifest
        if (
            manifest.get("kind") != "lerobot"
            or manifest.get("fps") != 30
            or manifest.get("chunk_size") != 32
            or manifest.get("state_dim") != 28
            or manifest.get("action_dim") != 28
            or manifest.get("cameras") != {"head": CAMERA}
            or manifest.get("camera_hw") != {"head": [480, 640]}
            or manifest.get("state_names") != manifest.get("action_names")
            or len(manifest.get("state_names", [])) != 28
            or manifest.get("counts", {}).get("eligible") != 2851
            or manifest.get("counts", {}).get("train_episodes") != 2567
            or manifest.get("counts", {}).get("val_episodes") != 284
        ):
            raise ValueError("Dex3 index differs from the validated G1 28D collection")
        self.episodes = [e for e in manifest["episodes"] if e["split"] == split]
        self.split = split
        self.decoder = decoder
        self._rows = None
        self._spans = []
        self._ends = []
        for episode in self.episodes:
            if "GraspSquare" in episode["episode_id"]:
                raise ValueError("duplicate GraspSquare in Dex3 index")
            if not episode["caption"].strip() or not episode["data_file"]:
                raise ValueError(f"missing task or Parquet reference: {episode['episode_id']}")
            video = episode["videos"]["head"]
            if video is None or not video["file"].endswith(".mp4"):
                raise ValueError(f"missing camera reference: {episode['episode_id']}")
            spans = valid_window_spans(episode["n_frames"], episode["valid_ranges"], 32)
            if sum(count for _, count in spans) != episode["n_valid_starts"]:
                raise ValueError(f"invalid starts: {episode['episode_id']}")
            for start, count in spans:
                self._spans.append((episode, start, count))
                self._ends.append((self._ends[-1] if self._ends else 0) + count)
        if len(self.episodes) != manifest["counts"][f"{split}_episodes"]:
            raise ValueError("split episode count differs from index")

    @property
    def rows(self):
        if self._rows is None:
            rows = np.load(self.index_dir / ROWS_FILENAME, mmap_mode="r")
            if rows.dtype != np.float32 or rows.shape != (self.manifest["total_frames"], 56):
                raise ValueError("Dex3 indexed rows must contain 28 state + 28 action channels")
            self._rows = rows
        return self._rows

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        span = bisect.bisect_right(self._ends, index)
        episode, start, _ = self._spans[span]
        start += index - (self._ends[span - 1] if span else 0)
        first = episode["from_index"] + start
        # FLUX3 history uses the previous absolute command followed by 32 targets.
        values = np.asarray(self.rows[first - 1 : first + 32], dtype=np.float32)
        if values.shape != (33, 56) or not np.isfinite(values).all():
            raise ValueError(f"invalid state/action rows: {episode['episode_id']} frame {start}")
        video = episode["videos"]["head"]
        frames = decode_window(
            self.source_root / video["file"],
            video["first_frame"] + start,
            33,
            fps=30,
            frame_hw=(192, 256),
            decoder=self.decoder,
            resize=True,
        )
        return {
            CAMERA: torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(),
            "observation.state": torch.from_numpy(values[1:2, :28].copy()),
            "action": torch.from_numpy(values[:, 28:].copy()),
            "task": episode["caption"],
            "episode_index": episode["episode_index"],
            "frame_index": start,
        }
