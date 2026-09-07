#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify-release-evidence.py"
SPEC = importlib.util.spec_from_file_location("verify_release_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReleaseEvidenceTests(unittest.TestCase):
    def test_replay_rejects_duplicate_pass_records(self):
        record = {"result": "PASS", "shape": [1, 40, 78], "finite": True}
        with self.assertRaisesRegex(MODULE.EvidenceError, "exactly one"):
            MODULE._validate_replay([record, record])

    def test_native_rejects_fail_marker_after_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.log"
            path.write_text("PASS isolated native harness dds=0 motor_transport=0\nFAIL late failure\n")
            with self.assertRaisesRegex(MODULE.EvidenceError, "one isolated PASS"):
                MODULE._validate_native(path)

    def test_closed_loop_requires_every_runtime_layer(self):
        records = [
            {"event": "closed_loop", "inference_frames": 2, "native_body_frames": 2},
            {"vla_connection": "upstream_policy_native_sonic_connected", "events": ["live_rgb_state"]},
        ]
        with self.assertRaisesRegex(MODULE.EvidenceError, "layer events"):
            MODULE._validate_closed_loop(records)

    def test_rollout_phase_annotations_must_match_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.json"
            video = root / "rollout.mp4"
            metrics.write_text(json.dumps({"approach": True, "contact_proxy": False, "hand_close": True, "stable_grasp": False, "lift": False, "stable_grasp_frames": 0, "max_lift_m": 0.0}))
            video.write_bytes(b"x" * 100)
            item = {"metrics": "metrics.json", "phases": {"approach": False, "contact": False, "hand_close": True, "stable_grasp": False, "lift": False}, "failure_layer": "contact"}
            with self.assertRaisesRegex(MODULE.EvidenceError, "disagree"):
                MODULE._validate_rollout(item, video, root, "0" * 40, 0, {}, 0)

    def test_rollout_rejects_classified_layer_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.json"
            video = root / "rollout.mp4"
            metrics.write_text(json.dumps({"approach": True, "contact_proxy": False, "hand_close": True, "stable_grasp": False, "lift": False, "stable_grasp_frames": 0, "max_lift_m": 0.0}))
            video.write_bytes(b"x" * 100)
            item = {"metrics": "metrics.json", "phases": {"approach": True, "contact": False, "hand_close": True, "stable_grasp": False, "lift": False}, "failure_layer": "contact"}
            with self.assertRaisesRegex(MODULE.EvidenceError, "failed at layer contact"):
                MODULE._validate_rollout(item, video, root, "0" * 40, 0, {}, 0)

    def test_relative_paths_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside-evidence"
            outside.write_text("proof")
            try:
                with self.assertRaisesRegex(MODULE.EvidenceError, "escapes"):
                    MODULE._resolve(root, "../outside-evidence", "artifact")
            finally:
                outside.unlink()


if __name__ == "__main__":
    unittest.main()
