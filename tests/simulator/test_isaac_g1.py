from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.contracts import ContractError, RunProfile  # noqa: E402
from humanoid_lab.simulators.isaac.evidence import StreamingVideoWriter  # noqa: E402


class IsaacG1ProfileTests(unittest.TestCase):
    def test_all_shipped_profiles_are_valid_and_distinct(self) -> None:
        profiles = [RunProfile.load(path) for path in sorted((ROOT / "configs/profiles").glob("isaac-g1-*.json"))]
        self.assertEqual(len(profiles), 3)
        self.assertEqual({profile.robot.hand.kind for profile in profiles}, {"no_hands", "inspire-ftp", "dex3"})
        self.assertTrue(all(profile.robot.body_dofs == 29 for profile in profiles))
        self.assertTrue(all(profile.camera.width == 640 and profile.camera.height == 480 for profile in profiles))
        self.assertEqual({profile.robot.hand.dofs for profile in profiles}, {0, 14, 24})

    def test_rejects_non_29dof_body(self) -> None:
        source = json.loads((ROOT / "configs/profiles/isaac-g1-no_hands.json").read_text())
        source["robot"]["body_dofs"] = 27
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ContractError, "29DoF"):
                RunProfile.load(path)

    def test_rejects_non_normalized_camera_rotation(self) -> None:
        source = json.loads((ROOT / "configs/profiles/isaac-g1-no_hands.json").read_text())
        source["camera"]["rotation_wxyz"] = [1.0, 1.0, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ContractError, "normalized"):
                RunProfile.load(path)


class IsaacG1RecorderTests(unittest.TestCase):
    def test_normalize_copies_rgba_without_retaining_source(self) -> None:
        import numpy as np

        source = np.zeros((1, 4, 5, 4), dtype=np.uint8)
        image = StreamingVideoWriter.normalize(source)
        self.assertEqual(image.shape, (4, 5, 3))
        source[..., :3] = 255
        self.assertEqual(int(image.sum()), 0)

    def test_expected_head_camera_shape_is_preserved(self) -> None:
        import numpy as np

        image = StreamingVideoWriter.normalize(np.zeros((1, 480, 640, 4), dtype=np.uint8))
        self.assertEqual(list(image.shape), [480, 640, 3])

    def test_empty_warmup_frame_is_retried(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            writer = StreamingVideoWriter(Path(directory) / "unused.mp4", fps=10)
            self.assertIsNone(writer.append(np.asarray([])))
            self.assertIsNone(writer.error)


class IsaacG1SourceInvariantTests(unittest.TestCase):
    def test_runner_has_no_controller_or_policy_imports(self) -> None:
        source = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        for forbidden in ("sonic_isaac", "cloudwalk", "groot", "StateLink", "LowCmd"):
            self.assertNotIn(forbidden, source)

    def test_loop_has_one_step_and_render_only_paths(self) -> None:
        source = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        run_body = source[
            source.index("    def run(self)") : source.index("    def _runtime_summary")
        ]
        self.assertEqual(source.count("self._sim.step()"), 1)
        self.assertEqual(run_body.count("self._step_physics(record=True)"), 1)
        self.assertEqual(run_body.count("self._step_physics(record=False)"), 1)
        self.assertEqual(run_body.count("self._sim.render()"), 1)
        self.assertIn("if self._pending_ui_reset:", run_body)
        self.assertNotIn("app.update", run_body)
        self.assertIn("self._transition(TimelineState.STOPPED)", run_body)

    def test_gui_exposes_reset_and_head_camera(self) -> None:
        source = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        self.assertIn('ui.Button("Reset Robot"', source)
        self.assertIn('"G1 Head Camera",', source)
        self.assertIn("ui.ByteImageProvider()", source)
        self.assertIn("set_data_array", source)
        self.assertNotIn("create_viewport_window", source)
        self.assertIn("clicked_fn=self._request_reset_from_ui", source)
        request_body = source[
            source.index("    def _request_reset_from_ui") : source.index("    def _open_ui")
        ]
        self.assertIn("self._pending_ui_reset = True", request_body)
        self.assertNotIn("self._timeline.pause()", request_body)
        self.assertNotIn("reset_episode", request_body)
        self.assertIn("self.reset_episode(resume=True)", source)
        reset_branch = source[
            source.index("                if self._pending_ui_reset:") :
            source.index("                if self._timeline.is_playing():")
        ]
        self.assertLess(
            reset_branch.index("self.reset_episode(resume=True)"),
            reset_branch.index("self._step_physics(record=False)"),
        )
        self.assertLess(
            reset_branch.index("self._step_physics(record=False)"),
            reset_branch.index("self.pause()"),
        )
    def test_passive_fall_acceptance_is_an_explicit_test_mode(self) -> None:
        cli = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        service = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        self.assertIn('parser.add_argument("--test", choices=("passive-fall",))', cli)
        self.assertIn('if self.test_mode == "passive-fall"', service)
        self.assertIn('"result": "COMPLETED"', service)

    def test_simulator_uses_fabric_without_usd_writeback(self) -> None:
        source = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        self.assertIn("use_fabric = True", source)
        self.assertIn('settings.set_bool("/physics/updateToUsd", not use_fabric)', source)
        self.assertIn("SimulationManager.enable_fabric(use_fabric)", source)


class IsaacG1TrajectoryTests(unittest.TestCase):
    def test_recording_comparison_is_tick_aligned(self) -> None:
        script = ROOT / "scripts/verify-isaac-trajectory.py"
        spec = importlib.util.spec_from_file_location("verify_isaac_trajectory", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        row = {
            "physics_tick": 1,
            "root_position": [0.0, 0.0, 0.8],
            "root_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "body_position": [0.0] * 29,
            "body_velocity": [0.0] * 29,
        }
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "on", Path(directory) / "off"
            first.mkdir(); second.mkdir()
            payload = "\n".join(json.dumps({**row, "physics_tick": tick}) for tick in (1, 2)) + "\n"
            (first / "simulator-metrics.jsonl").write_text(payload)
            (second / "simulator-metrics.jsonl").write_text(payload)
            result = module.compare(first, second, 1e-6)
            self.assertEqual(result["result"], "PASS")
            self.assertEqual(result["compared_ticks"], 2)


if __name__ == "__main__":
    unittest.main()
