from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from humanoid_lab.groot_inference_capture import load_capture
from humanoid_lab.psi0_bridge import groot_capture_runtime as runtime


class FakePolicy:
    def __init__(self) -> None:
        self.calls = []

    def get_action(self, observation):
        self.calls.append(observation)
        return {
            "action.motion_token": np.arange(12, dtype=np.float32).reshape(1, 3, 4),
            "action.left_hand_joints": np.ones((1, 3, 2), dtype=np.float64),
        }, {"server": "synthetic"}


class GrootCaptureRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        runtime._capture = None
        runtime._capture_dir = None
        runtime._request_sequence = 0
        runtime._chunk_sequence = 0
        runtime._links.clear()

    def observation(self):
        return {
            "video": {"ego_view": np.arange(24, dtype=np.uint8).reshape(1, 1, 2, 4, 3)},
            "state": {"left_arm": np.arange(6, dtype=np.float32).reshape(1, 1, 6)},
            "language": {"annotation.human.task_description": [["stack blocks"]]},
            "timestamps": 12.5,
        }

    def test_exact_request_response_and_execution_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {
            "HUMANOID_GROOT_CAPTURE_DIR": directory,
            "HUMANOID_GROOT_CAPTURE_MAX_REQUESTS": "2",
            "HUMANOID_GROOT_LEFT_HAND_CONTRACT": "model-independent",
            "HUMANOID_POLICY_CLOCK": "simulation",
        }, clear=False):
            policy = FakePolicy()
            observation = self.observation()
            raw, info = runtime.capture_get_action(policy, observation)
            processed = {key.replace("action.", ""): value for key, value in raw.items()}
            runtime.bind_processed_action(raw, processed)
            runtime.chunk_installed(processed, 2, 0.041)
            runtime.row_executed(processed, 2, 17)

            bundles = sorted(Path(directory).glob("request-*"))
            self.assertEqual(len(bundles), 1)
            loaded = load_capture(bundles[0])
            np.testing.assert_array_equal(
                loaded["observation"]["video"]["ego_view"], observation["video"]["ego_view"]
            )
            np.testing.assert_array_equal(
                loaded["action"]["action.motion_token"], raw["action.motion_token"]
            )
            self.assertEqual(loaded["metadata"]["source_stamps"]["state_source"], None)
            self.assertEqual(loaded["metadata"]["options"]["left_hand_contract"], "model-independent")
            self.assertEqual(loaded["metadata"]["options"]["policy_clock"], "simulation")
            self.assertEqual(loaded["metadata"]["info"], info)
            rows = [json.loads(line) for line in (Path(directory) / "execution-map.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["kind"], "chunk_installed")
            self.assertEqual(rows[0]["installed_start_row"], 2)
            self.assertEqual(rows[1]["kind"], "row_executed")
            self.assertEqual(rows[1]["row"], 2)
            self.assertEqual(rows[1]["frame_index"], 17)
            self.assertEqual(rows[0]["request_id"], rows[1]["request_id"])
            self.assertEqual(rows[0]["chunk_version"], rows[1]["chunk_version"])

    def test_max_request_limit_does_not_change_policy_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {
            "HUMANOID_GROOT_CAPTURE_DIR": directory,
            "HUMANOID_GROOT_CAPTURE_MAX_REQUESTS": "1",
        }, clear=False):
            policy = FakePolicy()
            first, _ = runtime.capture_get_action(policy, self.observation())
            second, _ = runtime.capture_get_action(policy, self.observation())
            self.assertEqual(len(policy.calls), 2)
            self.assertEqual(len(list(Path(directory).glob("request-*"))), 1)
            np.testing.assert_array_equal(first["action.motion_token"], second["action.motion_token"])


if __name__ == "__main__":
    unittest.main()
