"""The single canonical whole-body → SONIC production path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from humanoid_lab.datasets.sonic.production import (
    PRODUCTION_ORIENTATION_POLICY,
    build_encoder_input,
    encode_prepared_episode,
)
from humanoid_lab.datasets.sonic.schema import CanonicalEpisode


def episode(frames: int = 4) -> CanonicalEpisode:
    timestamps = np.arange(frames, dtype=np.float64) / 50.0
    joints = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) * 0.001
    hands = np.zeros((frames, 7), dtype=np.float32)
    return CanonicalEpisode(
        timestamps=timestamps,
        joint_pos=joints,
        joint_vel=np.zeros_like(joints),
        body_quat_wxyz=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (frames, 1)),
        body_pos=np.tile(np.array([0, 0, 0.79], dtype=np.float32), (frames, 1)),
        left_hand_joints=hands,
        right_hand_joints=hands.copy(),
    )


class FakeInfo:
    def to_dict(self) -> dict:
        return {"sha256": "model-sha", "providers": ["fake"]}


class FakeRunner:
    info = FakeInfo()

    def __init__(self, *args, **kwargs) -> None:
        pass

    def encode(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation[:64], dtype=np.float32)

    def encode_frames(self, observations: np.ndarray) -> np.ndarray:
        return np.stack([self.encode(row) for row in observations])


class ProductionPathTest(unittest.TestCase):
    def test_encoder_input_has_one_fixed_production_policy(self) -> None:
        artifacts = build_encoder_input(episode())
        self.assertEqual(PRODUCTION_ORIENTATION_POLICY.value, "reference_root_current_frame")
        self.assertEqual(artifacts.observation.shape, (4, 1751))
        self.assertEqual(artifacts.future_clamp_fraction.shape, (4,))

    def test_non_50hz_episode_is_rejected_at_the_common_boundary(self) -> None:
        source = episode()
        source.timestamps[:] = np.arange(len(source.timestamps)) / 30.0
        with self.assertRaisesRegex(ValueError, "uniform 50 Hz"):
            build_encoder_input(source)

    def _prepared(self, directory: Path, *, hands_available: bool) -> None:
        source = episode()
        artifacts = build_encoder_input(source)
        np.savez_compressed(
            directory / "encoder_observation.npz",
            observation=artifacts.observation,
            future_clamp_fraction=artifacts.future_clamp_fraction,
            timestamps=artifacts.timestamps,
        )
        np.savez_compressed(
            directory / "reference.npz",
            timestamps=source.timestamps,
            joint_pos=source.joint_pos,
            joint_vel=source.joint_vel,
            body_quat_wxyz=source.body_quat_wxyz,
            body_pos=source.body_pos,
            left_hand_joints=source.left_hand_joints,
            right_hand_joints=source.right_hand_joints,
        )
        (directory / "run_manifest.json").write_text(json.dumps({
            "encoder_orientation_policy": PRODUCTION_ORIENTATION_POLICY.value,
            "final_action_78d": {
                "status": "available" if hands_available else "blocked",
                "reason": "test contract",
            },
        }), encoding="utf-8")

    def test_available_and_blocked_hands_use_the_same_encoder_function(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "observation_config.yaml").write_text("test", encoding="utf-8")
            for name, available, expected_dim in (("dex3", True, 78), ("apple", False, 0)):
                with self.subTest(name=name):
                    target = root / name
                    target.mkdir()
                    self._prepared(target, hands_available=available)
                    with (
                        patch("humanoid_lab.datasets.sonic.production.G1EncoderRunner", FakeRunner),
                        patch("humanoid_lab.datasets.sonic.production.load_encoder_layout", return_value=SimpleNamespace(total_dim=1751)),
                        patch("humanoid_lab.datasets.sonic.production.sha256_file", return_value="config-sha"),
                    ):
                        manifest = encode_prepared_episode(target, model_dir=model)
                    self.assertEqual(manifest["pipeline"], "canonical_50hz_whole_body_to_sonic_v1_1")
                    self.assertEqual(manifest["action_dim"], expected_dim)
                    with np.load(target / "action_tokens.npz") as payload:
                        self.assertEqual(payload["motion_token"].shape, (4, 64))
                        self.assertEqual("action" in payload.files, available)


if __name__ == "__main__":
    unittest.main()
