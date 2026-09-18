"""Fail-closed validation of a produced Psi0 Unitree Dex3 / SONIC v1.1 dataset.

The validator re-derives every episode from the frozen sources with its own
nearest-timestamp search and its own chunk-validity mask, then compares the
result against what is on disk. It shares no resampling code with the converter,
so a wrong layout, a shifted timeline or a leaked invalid tail is caught rather
than confirmed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    ACTION_MODEL_DIM,
    ANCHOR_MASK_KEY,
    BODY_TOKEN_DIM,
    CANONICAL_STATE_NAMES,
    HAND_DIM,
    MASK_KEY,
    ConversionConfig,
    source_state_permutation,
    standing_lower_body,
)
from .convert import read_raw_episode

#: The dataset must declare exactly this camera and no other.
EXPECTED_TASKS = 13


class ValidationError(RuntimeError):
    """Raised when a check cannot even be evaluated."""


def _nearest(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """Independent brute-force nearest search; ties keep the lower index."""
    source = np.asarray(source_timestamps, dtype=np.float64)
    target = np.asarray(target_timestamps, dtype=np.float64)
    return np.asarray(
        [int(np.argmin(np.abs(source - value))) for value in target],
        dtype=np.int64,
    )


def _strict_anchor_mask(valid: np.ndarray, chunk: int) -> np.ndarray:
    """Independent restatement of ``anchor_valid[i] = all(valid[i:i+chunk])``."""
    mask = np.asarray(valid, dtype=bool)
    return np.asarray(
        [bool(mask[i : i + chunk].size == chunk and mask[i : i + chunk].all()) for i in range(mask.size)],
        dtype=bool,
    )


def _video_info(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = 0
        for _ in container.decode(stream):
            frames += 1
        return {
            "frames": frames,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "fps": float(stream.average_rate) if stream.average_rate else None,
        }


def _read_sonic(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"action", "timestamp", "training_valid_mask"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValidationError(f"{path}: frozen SONIC episode is missing {missing}")
        return (
            np.asarray(payload["action"], dtype=np.float32),
            np.asarray(payload["timestamp"], dtype=np.float64),
            np.asarray(payload["training_valid_mask"], dtype=bool),
        )


def _read_parquet(path: Path, columns: list[str]) -> dict[str, list[Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    return {name: table.column(name).to_pylist() for name in columns}


class DatasetValidator:
    """Collects pass/fail checks for one produced dataset root."""

    def __init__(
        self,
        config: ConversionConfig,
        root: Path,
        manifest: dict[str, Any],
        *,
        require_complete: bool = False,
    ):
        self.config = config
        self.root = Path(root)
        self.manifest = manifest
        self.require_complete = require_complete
        self.checks: list[dict[str, str]] = []
        self.episodes_checked = 0
        self.frames_checked = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        return ok

    @property
    def status(self) -> str:
        return "PASS" if all(entry["status"] == "PASS" for entry in self.checks) else "FAIL"

    # -- Gate 0: the split --------------------------------------------------
    def check_split(self) -> None:
        manifest, config = self.manifest, self.config
        self.check("split.config_sha256", manifest["config"]["sha256"] == config.sha256)
        collections = manifest["collections"]
        self.check(
            "split.collections",
            sorted(collections) == sorted(config.collection_names),
            f"{len(collections)} collections",
        )
        total = 0
        for name, entry in sorted(collections.items()):
            usable = config.usable_episodes(name)
            train, val = entry["train"], entry["val"]
            total += len(train) + len(val)
            self.check(f"split.{name}.covers_usable", sorted(train + val) == list(usable))
            self.check(f"split.{name}.disjoint", not (set(train) & set(val)))
            self.check(
                f"split.{name}.no_excluded",
                not (set(train + val) & set(config.collection(name).excluded)),
            )
            self.check(f"split.{name}.val_non_empty", len(val) >= config.min_val_per_collection)
            self.check(
                f"split.{name}.instruction",
                entry["instruction"] == config.tasks[name],
            )
        self.check("split.total_episodes", total == manifest["totals"]["usable"], str(total))
        val_tasks = sum(1 for entry in collections.values() if entry["val"])
        self.check("split.val_represents_all_tasks", val_tasks == len(config.collections), str(val_tasks))

    # -- Gate 1: the dataset -----------------------------------------------
    def check_split_directory(self, split: str) -> None:
        root = self.root / split
        info_path = root / "meta/info.json"
        if not info_path.is_file():
            raise ValidationError(f"missing {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info["features"]
        self.check(f"{split}.fps_is_30", float(info["fps"]) == float(self.config.dataset_fps), str(info["fps"]))
        self.check(f"{split}.codebase_version", info["codebase_version"] == "v2.1", info["codebase_version"])

        camera_keys = [key for key, spec in features.items() if spec["dtype"] in ("video", "image")]
        self.check(
            f"{split}.single_left_camera",
            camera_keys == [self.config.camera_target_key],
            f"cameras={camera_keys}",
        )
        for excluded in self.config.camera_excluded_keys:
            self.check(f"{split}.excluded_camera_absent[{excluded}]", excluded not in features)

        state = features[self.config.state_field]
        self.check(f"{split}.state_is_43d", [int(state["shape"][0])] == [len(CANONICAL_STATE_NAMES)])
        self.check(
            f"{split}.state_joint_names",
            tuple(state["names"]) == CANONICAL_STATE_NAMES,
            f"first={state['names'][0]} hands_at_29={state['names'][29]}",
        )
        self.check(
            f"{split}.hand_action_is_14d",
            [int(features[self.config.action_field]["shape"][0])] == [HAND_DIM],
        )
        self.check(
            f"{split}.body_token_is_64d",
            [int(features[self.config.body_token_field]["shape"][0])] == [BODY_TOKEN_DIM],
        )
        self.check(
            f"{split}.action_mask_is_80d",
            [int(features[MASK_KEY]["shape"][0])] == [ACTION_MODEL_DIM],
        )
        self.check(f"{split}.anchor_mask_declared", ANCHOR_MASK_KEY in features)

        tasks = [
            json.loads(line)
            for line in (root / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.check(f"{split}.task_count", len(tasks) == len(self.config.collections), str(len(tasks)))
        by_index = {int(row["task_index"]): row["task"] for row in tasks}
        self.check(
            f"{split}.task_instructions",
            all(by_index.get(index) == self.config.tasks[name] for index, name in enumerate(self.config.collection_names)),
        )
        self.check(
            f"{split}.grasp_square_label_corrected",
            by_index.get(self.config.collection_names.index("G1_Dex3_GraspSquare_Dataset")) != "camera packaging"
            and "square" in by_index.get(self.config.collection_names.index("G1_Dex3_GraspSquare_Dataset"), "").lower(),
            by_index.get(self.config.collection_names.index("G1_Dex3_GraspSquare_Dataset"), ""),
        )

        episodes = [
            json.loads(line)
            for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.check(f"{split}.episode_count", len(episodes) == int(info["total_episodes"]), str(len(episodes)))
        self.check(
            f"{split}.episode_index_contiguous",
            [int(row["episode_index"]) for row in episodes] == list(range(len(episodes))),
        )
        declared = {(row["source_collection"], int(row["source_episode_index"])) for row in episodes}
        allowed = {
            (collection, episode)
            for collection, entry in self.manifest["collections"].items()
            for episode in entry[split]
        }
        self.check(
            f"{split}.episodes_within_split_manifest",
            declared <= allowed,
            f"{len(declared)} episodes, {len(declared - allowed)} outside the split",
        )
        if self.require_complete:
            missing = allowed - declared
            self.check(
                f"{split}.covers_the_whole_split",
                not missing,
                f"{len(missing)} episodes of the split manifest are missing",
            )
        self.check(
            f"{split}.no_excluded_episode",
            not any(
                set(episode for collection, episode in declared if collection == name)
                & set(self.config.collection(name).excluded)
                for name in self.config.collection_names
            ),
        )
        for row in episodes:
            self._check_episode(split, root, row)

    def _check_episode(self, split: str, root: Path, row: dict[str, Any]) -> None:
        collection = row["source_collection"]
        source_index = int(row["source_episode_index"])
        label = f"{split}.{collection}.ep{source_index:06d}"
        parquet = root / "data" / f"chunk-{int(row['episode_index']) // 1000:03d}" / f"episode_{int(row['episode_index']):06d}.parquet"
        video = root / "videos" / f"chunk-{int(row['episode_index']) // 1000:03d}" / self.config.camera_target_key / f"episode_{int(row['episode_index']):06d}.mp4"
        if not parquet.is_file():
            self.check(f"{label}.parquet_exists", False, str(parquet))
            return

        columns = _read_parquet(
            parquet,
            [
                self.config.state_field,
                self.config.action_field,
                self.config.body_token_field,
                MASK_KEY,
                ANCHOR_MASK_KEY,
                "timestamp",
                "frame_index",
                "episode_index",
            ],
        )
        state = np.asarray(columns[self.config.state_field], dtype=np.float64)
        hand = np.asarray(columns[self.config.action_field], dtype=np.float32)
        token = np.asarray(columns[self.config.body_token_field], dtype=np.float32)
        mask = np.asarray(columns[MASK_KEY], dtype=np.float64)
        anchor = np.asarray(columns[ANCHOR_MASK_KEY], dtype=bool)
        timestamp = np.asarray(columns["timestamp"], dtype=np.float64)
        frames = int(row["length"])
        valid = mask[:, 0] > 0.5 if mask.ndim == 2 and mask.shape[1] else np.zeros(0, dtype=bool)

        self.check(f"{label}.action_mask_width", mask.shape == (frames, ACTION_MODEL_DIM), str(mask.shape))
        if mask.shape == (frames, ACTION_MODEL_DIM):
            neck = BODY_TOKEN_DIM + HAND_DIM
            self.check(
                f"{label}.action_mask_binary",
                bool(np.isin(mask, (0.0, 1.0)).all()),
            )
            self.check(
                f"{label}.action_mask_uniform_over_real_dims",
                bool((mask[:, :neck] == mask[:, :1]).all()),
            )
            self.check(
                f"{label}.action_mask_neck_padding_zero",
                bool((mask[:, neck:] == 0).all()),
            )

        self.check(f"{label}.frame_count", state.shape[0] == frames == len(timestamp), f"{state.shape[0]} vs {frames}")
        self.check(f"{label}.state_width", state.shape[1:] == (len(CANONICAL_STATE_NAMES),))
        self.check(f"{label}.hand_width", hand.shape == (frames, HAND_DIM), str(hand.shape))
        self.check(f"{label}.token_width", token.shape == (frames, BODY_TOKEN_DIM), str(token.shape))
        finite = bool(
            np.isfinite(state).all() and np.isfinite(hand).all() and np.isfinite(token).all()
        )
        self.check(f"{label}.finite", finite)
        if not finite:
            return

        # Independent re-derivation from the frozen sources.
        raw = read_raw_episode(self.config, collection, source_index)
        sonic_path = self.config.sonic_root / collection / "episodes" / f"episode_{source_index:06d}" / "action.npz"
        source_action, source_timestamp, source_valid = _read_sonic(sonic_path)
        index = _nearest(source_timestamp, raw.timestamp)
        mapping = np.abs(source_timestamp[index] - raw.timestamp)
        self.check(
            f"{label}.body_token_matches_source",
            bool(np.array_equal(token, source_action[index, :BODY_TOKEN_DIM])),
        )
        self.check(
            f"{label}.hand_action_matches_source",
            bool(np.array_equal(hand, source_action[index, BODY_TOKEN_DIM : BODY_TOKEN_DIM + HAND_DIM])),
        )
        self.check(
            f"{label}.action_mask_matches_source_validity",
            bool(np.array_equal(valid, source_valid[index])),
        )
        chunk = self.config.action_chunk_size
        self.check(f"{label}.anchor_mask_is_strict_30", bool(np.array_equal(anchor, _strict_anchor_mask(valid, chunk))))
        anchors = np.flatnonzero(anchor)
        self.check(
            f"{label}.selection_monotone",
            bool(mapping.size < 2 or (np.diff(index) >= 0).all()),
        )
        self.check(
            f"{label}.tail_mapping_within_bound",
            float(mapping.max()) <= self.config.tail_timestamp_error_s if mapping.size else True,
            f"max={float(mapping.max()):.5f}s" if mapping.size else "",
        )
        if anchors.size:
            supervised = float(mapping[: int(anchors[-1]) + chunk].max())
            self.check(
                f"{label}.supervised_mapping_within_bound",
                supervised <= self.config.max_timestamp_error_s,
                f"max={supervised:.5f}s",
            )
        else:
            self.check(f"{label}.has_valid_anchor", False, "no frame carries a full valid 30-target window")
        # The plan's Gate 1 wording: no invalid tail reaches a valid anchor.
        strict_only = bool(
            np.all(
                [
                    bool(valid[i : i + chunk].size == chunk and valid[i : i + chunk].all())
                    for i in np.flatnonzero(anchor)
                ]
            )
        )
        self.check(f"{label}.no_invalid_target_in_valid_anchor", strict_only)

        self.check(
            f"{label}.timeline_is_uniform_30hz",
            bool(np.abs(timestamp - np.arange(frames) / self.config.dataset_fps).max() < 1e-5),
            f"max deviation {float(np.abs(timestamp - np.arange(frames) / self.config.dataset_fps).max()):.2e}s",
        )
        self.check(f"{label}.frame_index_contiguous", list(columns["frame_index"]) == list(range(frames)))
        self.check(f"{label}.episode_index", set(columns["episode_index"]) == {int(row["episode_index"])})

        standing = standing_lower_body()
        self.check(
            f"{label}.standing_proxy_constant",
            bool(np.all(state[:, :15] == state[0, :15])),
        )
        self.check(
            f"{label}.standing_proxy_matches_contract",
            bool(np.allclose(state[0, :15], standing, atol=1e-6)),
            f"{state[0, :15].round(4).tolist()}",
        )
        measured = raw.state[:, list(source_state_permutation(raw.state_names))].astype(np.float64)
        self.check(
            f"{label}.measured_state_matches_raw",
            bool(np.allclose(state[:, 15:], measured, atol=1e-6, rtol=0)),
        )

        if not video.is_file():
            self.check(f"{label}.video_exists", False, str(video))
            return
        info = _video_info(video)
        self.check(f"{label}.video_frames", info["frames"] == frames, f"{info['frames']} vs {frames}")
        self.check(
            f"{label}.video_geometry",
            info["width"] == self.config.camera_width
            and info["height"] == self.config.camera_height,
            f"{info['width']}x{info['height']}",
        )
        self.check(f"{label}.video_fps", abs((info["fps"] or 0.0) - self.config.dataset_fps) < 0.01, str(info["fps"]))

        self.episodes_checked += 1
        self.frames_checked += frames

    # -- consumer load ------------------------------------------------------
    def check_lerobot_load(self, split: str) -> None:
        """Open the split with the pinned LeRobot loader, as Psi0 does."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as error:  # pragma: no cover - environment diagnostic
            self.check(f"{split}.lerobot_import", False, f"{type(error).__name__}: {error}")
            return
        self.check(f"{split}.lerobot_import", True)
        try:
            dataset = LeRobotDataset(
                repo_id=split,
                root=self.root / split,
                video_backend="pyav",
            )
            item = dataset[0]
        except Exception as error:
            self.check(f"{split}.lerobot_load", False, f"{type(error).__name__}: {error}")
            return
        image = item.get(self.config.camera_target_key)
        shape = tuple(image.shape) if hasattr(image, "shape") else None
        self.check(f"{split}.lerobot_load", True, f"{len(dataset)} frames")
        self.check(
            f"{split}.lerobot_video_tensor",
            shape is not None and shape[0] == 3 and shape[1:] == (
                self.config.camera_height,
                self.config.camera_width,
            ),
            str(shape),
        )
        state = item.get(self.config.state_field)
        self.check(
            f"{split}.lerobot_state_43d",
            state is not None and tuple(state.shape) == (len(CANONICAL_STATE_NAMES),),
            str(tuple(state.shape) if state is not None else None),
        )

    def report(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "status": self.status,
            "episodes_checked": self.episodes_checked,
            "frames_checked": self.frames_checked,
            "failed": [entry for entry in self.checks if entry["status"] == "FAIL"],
            "checks": self.checks,
        }


def validate_dataset(
    config: ConversionConfig,
    root: Path,
    manifest: dict[str, Any],
    *,
    splits: tuple[str, ...] | None = None,
    check_lerobot: bool = True,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate a produced dataset root and return a machine-readable report."""
    validator = DatasetValidator(config, Path(root), manifest, require_complete=require_complete)
    validator.check_split()
    for split in splits or (config.train_repo, config.val_repo):
        validator.check_split_directory(split)
        if check_lerobot:
            validator.check_lerobot_load(split)
    return validator.report()
