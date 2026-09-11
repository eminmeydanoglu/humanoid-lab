from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from humanoid_lab.simulators.isaac.contracts import ContractError, RunProfile  # noqa: E402

SERVICE = ROOT / "src/humanoid_lab/simulators/isaac/service.py"


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


class IsaacG1SimulationConfigTests(unittest.TestCase):
    def test_shipped_profiles_resolve_to_cpu_and_25hz(self) -> None:
        for path in sorted((ROOT / "configs/profiles").glob("isaac-g1-*.json")):
            profile = RunProfile.load(path)
            self.assertEqual(profile.device, "cpu")
            self.assertEqual(profile.render_interval, 8)
            self.assertAlmostEqual(profile.camera_update_period, 0.04)

    def test_device_override_only_applies_when_requested(self) -> None:
        profile = RunProfile.load(ROOT / "configs/profiles/isaac-g1-dex3.json")
        self.assertEqual(profile.with_device(None).device, "cpu")
        self.assertEqual(profile.with_device("cuda").device, "cuda")


class IsaacG1SourceInvariantTests(unittest.TestCase):
    def test_runner_has_no_controller_or_policy_imports(self) -> None:
        source = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        for forbidden in ("sonic_isaac", "cloudwalk", "groot", "StateLink", "LowCmd"):
            self.assertNotIn(forbidden, source)

    def test_loop_decouples_physics_from_render(self) -> None:
        source = SERVICE.read_text()
        step_body = source[source.index("    def _step_physics") : source.index("    def _render_paused")]
        paused_body = source[
            source.index("    def _render_paused") : source.index("    def run(self)")
        ]
        run_body = source[source.index("    def run(self)") : source.index("    def _accounting")]
        self.assertEqual(source.count("self._sim.step(render=False)"), 1)
        self.assertNotIn("self._sim.step()", source)
        self.assertEqual(step_body.count("self._sim.render()"), 1)
        self.assertIn(
            "rendered = self._is_rendering and self.tick % self.profile.render_interval == 0",
            step_body,
        )
        self.assertIn("if rendered:", step_body)
        self.assertIn("self._consume_head_camera_frame()", step_body)
        self.assertEqual(run_body.count("self._step_physics()"), 2)
        self.assertNotIn("self._sim.render()", run_body)
        self.assertIn("self._render_paused()", run_body)
        self.assertIn("1.0 / self.PAUSED_RENDER_HZ", paused_body)
        self.assertIn("time.monotonic()", paused_body)
        self.assertIn("time.sleep(delay)", paused_body)
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
            reset_branch.index("self._step_physics()"),
        )
        self.assertLess(
            reset_branch.index("self._step_physics()"),
            reset_branch.index("self.pause()"),
        )
    def test_passive_fall_acceptance_is_an_explicit_test_mode(self) -> None:
        cli = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        service = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        self.assertIn('parser.add_argument("--test", choices=("passive-fall",))', cli)
        self.assertIn('if self.test_mode == "passive-fall"', service)
        self.assertIn('"result": "COMPLETED"', service)
        self.assertIn('"head_camera_shape": self._camera_shape', service)
        self.assertIn('"head_camera_flowing": self._camera_frames >= 2', service)
        self.assertNotIn('"clean_stop": True', service)

    def test_fabric_is_conditional_on_the_physics_device(self) -> None:
        source = SERVICE.read_text()
        self.assertIn('use_fabric = self._device.startswith("cuda")', source)
        self.assertIn('settings.set_bool("/physics/fabricEnabled", use_fabric)', source)
        self.assertIn('settings.set_bool("/physics/updateToUsd", not use_fabric)', source)
        self.assertIn("SimulationManager.enable_fabric(use_fabric)", source)
        self.assertIn(
            'settings.set_int("/persistent/physics/numThreads", self.PHYSX_NUM_THREADS)',
            source,
        )
        self.assertIn("device=self._device", source)
        self.assertIn("render_interval=self.profile.render_interval", source)
        self.assertIn("update_period=self.profile.camera_update_period", source)

if __name__ == "__main__":
    unittest.main()
