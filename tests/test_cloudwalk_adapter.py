#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cloudwalk_adapter import ACTION_SIZE, CHECKPOINT_REVISION, ContractError, PROMPT, split_action_chunk, validate_observation

class CloudWalkContractTests(unittest.TestCase):
    def test_runner_enables_its_required_rgb_camera(self):
        runner = Path(__file__).resolve().parents[1] / "scripts" / "run-cloudwalk-isaac.py"
        spec = importlib.util.spec_from_file_location("run_cloudwalk_isaac", runner)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        args = type("Args", (), {"enable_cameras": False})()
        module._enable_required_camera(args)
        self.assertTrue(args.enable_cameras)

    def test_compatibility_exports_use_canonical_chunk_validation(self):
        chunk = [[0.25] * ACTION_SIZE for _ in range(40)]
        self.assertEqual(len(split_action_chunk(chunk)), 40)
        with self.assertRaises(ContractError): split_action_chunk(chunk[:-1])

    def test_observation_and_two_independent_replays_are_strict(self):
        image = [[[0] * 3 for _ in range(640)] for _ in range(480)]
        validate_observation(image, [0.0] * 43, PROMPT)
        passed = {"result":"PASS","embodiment":"UNITREE_G1_SONIC","literal_prompt":PROMPT,"checkpoint_revision":CHECKPOINT_REVISION,"shape":[1,40,78],"dtype":"float32","finite":True,"latency_seconds":0.2,"splits":{"motion_token":[1,40,64],"left_hand_joints":[1,40,7],"right_hand_joints":[1,40,7]}}
        verifier = Path(__file__).resolve().parents[1] / "scripts" / "verify-cloudwalk-groot-replay.py"
        with tempfile.TemporaryDirectory() as directory:
            one, two = Path(directory) / "one.log", Path(directory) / "two.log"
            one.write_text(json.dumps(passed)); two.write_text(json.dumps(passed))
            self.assertEqual(subprocess.run([sys.executable, str(verifier), str(one), str(two)], capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run([sys.executable, str(verifier), str(one), str(one)], capture_output=True).returncode, 2)

if __name__ == "__main__": unittest.main()
