import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run-cloudwalk-isaac.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_cloudwalk_isaac_lifecycle", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CloudWalkIsaacLifecycleTests(unittest.TestCase):
    def test_reset_completes_before_timeline_is_paused(self):
        runner = load_runner()
        operations = Mock()
        sim = Mock()
        timeline = Mock()
        sim.reset.side_effect = lambda: operations("reset")
        timeline.pause.side_effect = lambda: operations("pause")

        runner._reset_simulation_paused(sim, timeline)

        self.assertEqual(operations.call_args_list, [call("reset"), call("pause")])

    def test_interactive_loop_steps_physics_only_while_timeline_is_playing(self):
        runner = load_runner()
        simulation_app = Mock()
        timeline = Mock()
        sim = Mock()
        scene = Mock()
        actuation = Mock()
        simulation_app.is_running.side_effect = [True, True, True, False]
        timeline.is_playing.side_effect = [False, True, False]
        sim.get_physics_dt.return_value = 0.01

        with patch("builtins.print") as emit:
            physics_steps = runner._run_interactive_app(simulation_app, timeline, sim, scene, actuation)

        self.assertEqual(physics_steps, 1)
        self.assertEqual(simulation_app.update.call_count, 2)
        actuation.enforce.assert_called_once_with()
        scene.write_data_to_sim.assert_called_once_with()
        sim.step.assert_called_once_with()
        sim.get_physics_dt.assert_called_once_with()
        scene.update.assert_called_once_with(0.01)
        emit.assert_called_once()
        self.assertEqual(emit.call_args.args[0], '{"event": "timeline_play_started"}')
        self.assertTrue(emit.call_args.kwargs["flush"])

    def test_interactive_physics_reapplies_actuation_mode_before_writing_scene_data(self):
        runner = load_runner()
        operations = Mock()
        timeline = Mock()
        timeline.is_playing.return_value = True
        sim = Mock()
        sim.get_physics_dt.return_value = 0.01
        scene = Mock()
        actuation = Mock()
        actuation.enforce.side_effect = lambda: operations("enforce")
        scene.write_data_to_sim.side_effect = lambda: operations("write")
        sim.step.side_effect = lambda: operations("step")

        self.assertTrue(runner._step_interactive_frame(Mock(), timeline, sim, scene, actuation))

        self.assertEqual(operations.call_args_list, [call("enforce"), call("write"), call("step")])

    def test_rollout_metrics_names_are_bound_outside_closed_loop(self):
        source = RUNNER.read_text()
        top_import = source.split("from sonic_isaac_inspire_adapter import", 1)[1].split("\n", 1)[0]
        self.assertIn("ACTION_RATE_HZ", top_import)
        shared_init = source.index("inference_frames = 0")
        self.assertLess(shared_init, source.index("if args.closed_loop:"))
        self.assertLess(source.index("body_frames = 0"), source.index("if args.closed_loop:"))

    def test_fall_acceptance_rejects_frozen_pose_and_accepts_drop_or_tilt(self):
        runner = load_runner()

        self.assertFalse(runner._fall_accepted(0.0, 1.0))
        self.assertTrue(runner._fall_accepted(runner.INTERACTIVE_FALL_MIN_DROP_M, 1.0))
        self.assertTrue(runner._fall_accepted(0.0, runner.INTERACTIVE_FALL_MAX_PELVIS_UP_Z))

    def test_safe_reference_diagnostic_activates_native_control_without_groot(self):
        source = (ROOT / "scripts" / "cloudwalk-controller-session.py").read_text()
        safe_branch = source.split("if args.safe_reference_only:", 1)[1].split("sequence = 0", 1)[0]

        self.assertIn('control.send_json({"op": "run"})', safe_branch)
        self.assertIn("controller_safe_reference_active", safe_branch)
        self.assertNotIn("inference_requests", safe_branch)

    def test_interactive_mode_leaves_physics_to_gui_and_keeps_gpu_fabric_enabled(self):
        source = RUNNER.read_text()
        self.assertIn("SimulationCfg(dt=0.01, device=args.device, use_fabric=True)", source)
        self.assertIn('physics_settings.set_bool("/physics/fabricEnabled", True)', source)
        self.assertIn('physics_settings.set_bool("/physics/updateToUsd", False)', source)
        self.assertIn("SimulationManager.enable_fabric(True)", source)
        self.assertIn("from isaaclab.sim.utils.stage import attach_stage_to_usd_context", source)
        self.assertIn("attach_stage_to_usd_context()", source)
        self.assertLess(source.index("configure_g1_free_base_articulation()"), source.index("attach_stage_to_usd_context()"))
        self.assertLess(source.index("attach_stage_to_usd_context()"), source.index("_reset_simulation_paused(sim, timeline)"))
        self.assertIn('not sim.is_fabric_enabled()', source)
        self.assertNotIn("use_fabric=not interactive_mode", source)
        self.assertNotIn('physics_settings.set_bool("/physics/fabricEnabled", False)', source)
        self.assertNotIn('physics_settings.set_bool("/physics/updateToUsd", True)', source)
        self.assertNotIn("SimulationManager.enable_fabric(False)", source)
        branch = source.split("elif interactive_mode:", 1)[1].split("else:\n            for _ in range(args.steps):", 1)[0]
        normal_interactive = branch.split("else:\n                bridge = SimulatorControllerBridge", 1)[1]

        self.assertIn("actuation.set_passive()", branch)
        self.assertIn("scene.write_data_to_sim()", branch)
        self.assertIn("_run_interactive_app(", normal_interactive)
        self.assertIn("before_frame=poll_controller", normal_interactive)
        self.assertIn("before_step=controller_step", normal_interactive)
        self.assertIn("if args.controller_attach_test_steps:", normal_interactive)
        self.assertIn("controller_bootstrap_observation", normal_interactive)
        self.assertIn('while bridge.status.state != "controlled"', normal_interactive)
        self.assertLess(normal_interactive.index('while bridge.status.state != "controlled"'), normal_interactive.index("timeline.play()"))
        self.assertIn("controller start changed the paused Timeline state", normal_interactive)
        self.assertIn("controller policy activation changed the paused Timeline state", normal_interactive)
        self.assertIn("else:\n                        interactive_physics_steps = _run_interactive_app(", normal_interactive)
        self.assertNotIn("timeline.stop()", source)
        self.assertIn("if args.interactive_fall_test:\n                timeline.play()\n                simulation_app.update()", branch)
        self.assertIn("INTERACTIVE_FALL_MIN_DROP_M", source)
        self.assertIn("INTERACTIVE_FALL_MAX_PELVIS_UP_Z", source)
        self.assertIn('"physical_acceptance_enforced": False', source)
        self.assertNotIn("controller stability acceptance failed", source)
        self.assertNotIn("interactive standing acceptance failed", source)


if __name__ == "__main__":
    unittest.main()
