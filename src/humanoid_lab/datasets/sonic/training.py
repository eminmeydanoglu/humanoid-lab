"""Fail-closed reader for production Unitree SONIC VLA action artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_DIM = 78
DEFAULT_ACTION_HORIZON = 40


@dataclass(frozen=True)
class SonicTrainingEpisode:
    action: np.ndarray
    timestamp: np.ndarray
    frame_index: np.ndarray
    training_valid_mask: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "SonicTrainingEpisode":
        with np.load(path, allow_pickle=False) as payload:
            required = {"action", "timestamp", "frame_index", "training_valid_mask"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(
                    f"legacy/incomplete SONIC training artifact {path}: missing {missing}; "
                    "reconvert it with the production Unitree converter"
                )
            episode = cls(
                action=np.asarray(payload["action"], dtype=np.float32),
                timestamp=np.asarray(payload["timestamp"], dtype=np.float64),
                frame_index=np.asarray(payload["frame_index"], dtype=np.int64),
                training_valid_mask=np.asarray(payload["training_valid_mask"], dtype=bool),
            )
        episode.validate()
        return episode

    def validate(self) -> None:
        frames = len(self.timestamp)
        if self.action.shape != (frames, ACTION_DIM):
            raise ValueError(f"training action must be [{frames},{ACTION_DIM}], got {self.action.shape}")
        if self.frame_index.shape != (frames,) or not np.array_equal(self.frame_index, np.arange(frames)):
            raise ValueError("training frame_index must be contiguous from zero")
        if self.training_valid_mask.shape != (frames,):
            raise ValueError("training_valid_mask must have one value per frame")
        if not np.isfinite(self.action).all() or not np.isfinite(self.timestamp).all():
            raise ValueError("training artifact contains NaN or Inf")
        invalid = np.flatnonzero(~self.training_valid_mask)
        if len(invalid) != 45 or not np.array_equal(invalid, np.arange(frames - 45, frames)):
            raise ValueError("SONIC step5 future-window mask must exclude exactly the final 45 frames")

    def valid_actions(self) -> np.ndarray:
        """Individual 78D supervision targets; invalid tail never reaches loss."""
        return self.action[self.training_valid_mask]

    @property
    def motion_token(self) -> np.ndarray:
        return self.action[:, :64]

    @property
    def left_hand_target(self) -> np.ndarray:
        return self.action[:, 64:71]

    @property
    def right_hand_target(self) -> np.ndarray:
        return self.action[:, 71:78]

    def valid_action_chunks(self, horizon: int = DEFAULT_ACTION_HORIZON) -> tuple[np.ndarray, np.ndarray]:
        """Return anchors and chunks for which every member is a valid target."""
        if horizon <= 0:
            raise ValueError("action horizon must be positive")
        windows = np.lib.stride_tricks.sliding_window_view(self.training_valid_mask, horizon)
        anchor_mask = windows.all(axis=1)
        anchors = np.flatnonzero(anchor_mask)
        chunks = np.stack([self.action[index:index + horizon] for index in anchors], axis=0)
        return anchors, chunks
