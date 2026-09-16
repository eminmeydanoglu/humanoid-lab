from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid_lab.datasets.sonic.training import SonicTrainingEpisode


class SonicTrainingDatasetTest(unittest.TestCase):
    def write(self, path: Path, action: np.ndarray, *, mask: bool = True) -> None:
        arrays = {
            "action": action,
            "timestamp": np.arange(len(action), dtype=np.float64) / 50.0,
            "frame_index": np.arange(len(action), dtype=np.int64),
        }
        if mask:
            valid = np.ones(len(action), dtype=bool)
            valid[-45:] = False
            arrays["training_valid_mask"] = valid
        np.savez(path, **arrays)

    def test_ep003_frame_and_chunk_anchor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.npz"
            self.write(path, np.zeros((822, 78), dtype=np.float32))
            episode = SonicTrainingEpisode.load(path)
            self.assertEqual(len(episode.valid_actions()), 777)
            anchors, chunks = episode.valid_action_chunks(40)
            self.assertEqual(len(anchors), 738)
            self.assertEqual(chunks.shape, (738, 40, 78))
            self.assertEqual(int(anchors[-1]), 737)
            self.assertEqual(episode.motion_token.shape, (822, 64))
            self.assertEqual(episode.left_hand_target.shape, (822, 7))
            self.assertEqual(episode.right_hand_target.shape, (822, 7))

    def test_invalid_tail_never_changes_frame_or_chunk_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            action = np.arange(822 * 78, dtype=np.float32).reshape(822, 78)
            changed = action.copy()
            changed[-45:] = 1e9
            self.write(first, action)
            self.write(second, changed)
            a = SonicTrainingEpisode.load(first)
            b = SonicTrainingEpisode.load(second)
            np.testing.assert_array_equal(a.valid_actions(), b.valid_actions())
            np.testing.assert_array_equal(a.valid_action_chunks(40)[1], b.valid_action_chunks(40)[1])

    def test_legacy_artifact_without_mask_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            self.write(path, np.zeros((100, 78), dtype=np.float32), mask=False)
            with self.assertRaisesRegex(ValueError, "legacy/incomplete"):
                SonicTrainingEpisode.load(path)


if __name__ == "__main__":
    unittest.main()
