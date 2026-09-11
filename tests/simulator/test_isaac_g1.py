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
        base_profiles = [
            "isaac-g1-no_hands.json",
            "isaac-g1-inspire-ftp.json",
            "isaac-g1-dex3.json",
        ]
        profiles = [
            RunProfile.load(ROOT / "configs/profiles" / name) for name in base_profiles
        ]
        self.assertEqual({profile.robot.hand.kind for profile in profiles}, {"no_hands", "inspire-ftp", "dex3"})
        self.assertTrue(all(profile.robot.body_dofs == 29 for profile in profiles))
        self.assertTrue(all(profile.camera.width == 640 and profile.camera.height == 480 for profile in profiles))
        self.assertEqual({profile.robot.hand.dofs for profile in profiles}, {0, 14, 24})
        # The base platform profiles declare no controller: that is what keeps
        # Gate 1 behaviour reproducible while control paths are added.
        self.assertTrue(all(profile.controller is None for profile in profiles))

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
    def test_shipped_profiles_resolve_to_cpu_and_smooth_render_cadence(self) -> None:
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
    def test_normal_runs_have_no_implicit_five_minute_timeout(self) -> None:
        cli = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        self.assertNotIn("else 300.0", cli)
        self.assertIn("if args.duration is None and args.test:", cli)
        service = SERVICE.read_text()
        self.assertIn("self.duration is None", service)

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
        self.assertIn("create_viewport_window", source)
        self.assertIn("camera_path=camera_path", source)
        self.assertIn("if self.show_head_camera:", source)
        self.assertIn('if self.test_mode is not None or self.show_head_camera', source)
        self.assertNotIn("ui.ByteImageProvider()", source)
        self.assertNotIn("set_data_array", source)
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
    def test_acceptance_modes_are_explicit_and_measured(self) -> None:
        cli = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        service = (ROOT / "src/humanoid_lab/simulators/isaac/service.py").read_text()
        self.assertIn(
            'parser.add_argument("--test", choices=("passive-fall", "controlled-hold", "controller-hold"))',
            cli,
        )
        self.assertIn('if self.test_mode == "passive-fall"', service)
        self.assertIn('if self.test_mode == "controlled-hold"', service)
        self.assertIn('if self.test_mode == "controller-hold"', service)
        self.assertIn('"result": "COMPLETED"', service)
        self.assertIn('"head_camera_shape": self._camera_shape', service)
        self.assertIn('"head_camera_flowing": self._camera_frames >= 2', service)
        self.assertNotIn('"clean_stop": True', service)

    def test_controller_override_can_only_disable_a_profile_controller(self) -> None:
        """Provider configuration belongs to the profile; a CLI override must
        not accidentally open an unconfigured DDS domain or partial controller."""
        cli = (ROOT / "src/humanoid_lab/simulators/isaac/cli.py").read_text()
        self.assertIn('choices=("none",)', cli)
        self.assertIn("safe passive fallback", cli)

    def test_commands_reach_the_solver_by_one_path(self) -> None:
        """The controller's torque is the torque; the asset's actuator models
        compute their own effort and would silently discard it."""
        source = SERVICE.read_text()
        self.assertEqual(source.count("set_dof_actuation_forces"), 1)
        self.assertIn("self._flush_effort()", source)
        # The staged effort is the only thing written per tick; the asset's own
        # effort path is not used for control.
        write_body = source[
            source.index("    def _write_effort") : source.index("    def _flush_effort")
        ]
        self.assertNotIn("set_joint_effort_target", write_body)
        self.assertIn("self._effort_target[0, list(indices)] = tensor", write_body)

    def test_passive_behaviour_is_untouched_without_a_controller(self) -> None:
        source = SERVICE.read_text()
        step_body = source[source.index("    def _step_physics") : source.index("    def _render_paused")]
        self.assertIn("self._apply_control()", step_body)
        control_body = source[
            source.index("    def _apply_control") : source.index("    def _apply_body_command")
        ]
        # No controller configured means the loop never even polls one.
        self.assertIn("if self._controller is None:\n            return", control_body)

    def test_command_timeout_returns_the_robot_to_passive(self) -> None:
        source = SERVICE.read_text()
        control_body = source[
            source.index("    def _apply_control") : source.index("    def _apply_body_command")
        ]
        self.assertIn("command.is_valid_at(self.tick)", control_body)
        self.assertIn("self._go_passive()", control_body)
        self.assertIn("self._control_mode = PASSIVE", control_body)

    def test_a_bad_command_cannot_kill_the_loop(self) -> None:
        """Undecodable, non-finite or mis-ordered output must degrade to passive,
        not raise out of the physics loop."""
        source = SERVICE.read_text()
        control_body = source[
            source.index("    def _apply_control") : source.index("    def _apply_body_command")
        ]
        self.assertIn("except CommandError as error:", control_body)
        self.assertIn("self._rejected_commands += 1", control_body)
        self.assertIn("self._go_passive()", control_body)
        status_body = source[
            source.index("    def _controller_status") : source.index("    def _controller_hold_acceptance")
        ]
        self.assertIn('"rejected_commands": self._rejected_commands', status_body)

    def test_reset_is_refused_while_a_controller_is_driving(self) -> None:
        """A reset teleports the articulation while the controller still holds
        state from the previous episode; mixing the two is not allowed."""
        source = SERVICE.read_text()
        body = source[
            source.index("    def _request_reset_from_ui") : source.index("    def _open_ui")
        ]
        self.assertIn("if self._control_mode == CONTROLLED:", body)
        self.assertIn("isaac_g1_ui_reset_refused", body)
        # Only the non-controlled path may queue work for the loop.
        refused, _, rest = body.partition('"isaac_g1_ui_reset_refused"')
        self.assertIn("return", rest.split("self._pending_ui_reset = True")[0])
        self.assertIn("self._pending_ui_reset = True", rest)

    def test_a_new_episode_invalidates_old_commands(self) -> None:
        source = (ROOT / "src/humanoid_lab/controllers/sonic_dds.py").read_text()
        publish_body = source[
            source.index("    def publish_state") : source.index("    def _publish_loop")
        ]
        self.assertIn("if state.episode_id != self._episode_id:", publish_body)
        for cleared in ("self._low_command = None", "self._left_hand_command = None"):
            self.assertIn(cleared, publish_body)

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
        self.assertIn("render_interval=1", source)
        self.assertIn('rendering_mode="balanced"', source)
        self.assertIn("enable_dlssg=self.show_ui", source)
        self.assertIn("enable_dl_denoiser=True", source)
        self.assertIn("update_period=self.profile.camera_update_period", source)

if __name__ == "__main__":
    unittest.main()
