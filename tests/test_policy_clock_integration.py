from __future__ import annotations

import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from humanoid_lab.psi0_bridge.groot_backend import GrootProcessGroup
from humanoid_lab.psi0_bridge.groot_clock_runtime import advance_chunk_index
from humanoid_lab.psi0_bridge.sim_clock import SimulationClockPacer
from humanoid_lab.psi0_bridge.policy_clock import PolicyClock, PolicyClockError

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_groot_vla", ROOT / "scripts/run-groot-vla.py")
WRAPPER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WRAPPER)


class PolicyClockTest(unittest.TestCase):
    def atomic_write(self, path: Path, value: float) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(f"{value:.6f}\n", encoding="ascii")
        os.replace(temporary, path)

    def test_stall_reset_and_latency_use_simulation_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clock"
            self.atomic_write(path, 2.0)
            clock = PolicyClock("simulation", path, stale_timeout_s=1.0)
            start = clock.now()
            time.sleep(0.01)
            same = clock.now()
            self.assertEqual(clock.latency_seconds(start, same), 0.0)
            self.atomic_write(path, 2.08)
            advanced = clock.now()
            self.assertAlmostEqual(clock.latency_seconds(start, advanced), 0.08)
            self.atomic_write(path, 0.0)
            reset = clock.now()
            self.assertEqual(reset.generation, start.generation + 1)
            with self.assertRaisesRegex(PolicyClockError, "reset"):
                clock.latency_seconds(start, reset)

    def test_stale_clock_fails_closed_without_changing_wall_watchdogs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clock"
            self.atomic_write(path, 1.0)
            clock = PolicyClock("simulation", path, stale_timeout_s=0.01)
            clock.now()
            time.sleep(0.02)
            with self.assertRaisesRegex(PolicyClockError, "stopped"):
                clock.now()
            wall = PolicyClock("wall")
            self.assertGreaterEqual(wall.now().seconds, 0.0)

    def test_clock_retries_an_atomic_replacement_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clock"
            self.atomic_write(path, 1.0)
            clock = PolicyClock("simulation", path)
            real_read = Path.read_bytes
            replaced = False

            def replace_once(candidate: Path) -> bytes:
                nonlocal replaced
                raw = real_read(candidate)
                if not replaced:
                    replaced = True
                    self.atomic_write(path, 1.02)
                return raw

            with mock.patch.object(Path, "read_bytes", replace_once):
                sample = clock.now()
            self.assertAlmostEqual(sample.seconds, 1.02)


class GrootSchedulingBehaviorTest(unittest.TestCase):
    def test_quantized_50hz_clock_stalls_do_not_duplicate_rows(self) -> None:
        pacer = SimulationClockPacer(50.0)
        index = 0
        published = []
        for sim_s in (0.0, 0.0, 0.0, 0.02, 0.02, 0.04):
            due = pacer.update(sim_s, 0).rows_due
            index, send = advance_chunk_index(index, due, 40)
            if send:
                published.append(index)
                index = min(index + 1, 39)
        self.assertEqual(published, [0, 1, 2])

    def test_large_clock_jump_skips_stale_rows_without_burst(self) -> None:
        pacer = SimulationClockPacer(50.0)
        self.assertEqual(pacer.update(0.0, 0).rows_due, 1)
        due = pacer.update(0.1, 0).rows_due
        index, send = advance_chunk_index(1, due, 40)
        self.assertTrue(send)
        self.assertEqual(due, 5)
        self.assertEqual(index, 5)

    def test_reset_restarts_pacer_without_carrying_elapsed_time(self) -> None:
        pacer = SimulationClockPacer(50.0)
        pacer.update(4.0, 0)
        pacer.update(4.04, 0)
        update = pacer.update(0.0, 1)
        self.assertTrue(update.reset)
        self.assertEqual(update.rows_due, 1)


class GrootWrapperIntegrationTest(unittest.TestCase):
    def test_wrapper_defaults_to_simulation_and_trained_hand_contract(self) -> None:
        with mock.patch("sys.argv", ["run-groot-vla.py", "--host=127.0.0.1"]):
            args, upstream = WRAPPER.parse_args()
        self.assertEqual(args.policy_clock, "simulation")
        self.assertEqual(args.left_hand_contract, "model-independent")
        self.assertEqual(upstream, ["--host=127.0.0.1"])

    def test_wrapper_consumes_legacy_left_hand_contract_before_upstream(self) -> None:
        with mock.patch("sys.argv", ["run-groot-vla.py", "--left-hand-contract=compatibility", "--host=127.0.0.1"]):
            args, upstream = WRAPPER.parse_args()
        self.assertEqual(args.left_hand_contract, "compatibility")
        self.assertEqual(upstream, ["--host=127.0.0.1"])

    def test_transformed_upstream_compiles_and_contains_all_three_clock_seams(self) -> None:
        source = '''import time\n\nimport numpy as np\n    # Copy index finger data to middle finger (hardware coupling)\n    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]\n    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]\n        action, _info = policy.get_action(observation)\n        processed_action = concat_action(robot_model, action)\n                inference_start_time = time.monotonic()\n    loop_rate = config.action_publish_rate\n    loop_period = 1.0 / loop_rate\n    last_inference_time = 0.0\n            t_start = time.monotonic()\n            check_keyboard_input()\n                inference_delay = time.monotonic() - inference_start_time\n                last_inference_time = time.monotonic()\n                time_since_last_inference=(time.monotonic() - last_inference_time),\n                    zmq_socket.send(zmq_message)\n                    last_sent_motion_token = motion_token.copy()\n            with telemetry.timer("total_loop"):\n'''
        transformed = WRAPPER.transform_source(source)
        self.assertIn("import os", transformed)
        self.assertIn("inference_start_time = _POLICY_CLOCK.now()", transformed)
        self.assertIn("inference_delay = policy_clock.latency_seconds(inference_start_time)", transformed)
        self.assertNotIn("latency_seconds(inference_start_time, policy_time)", transformed)
        self.assertIn("rows_due = clock_update.rows_due", transformed)
        self.assertIn("time_since_last_inference=(", transformed)
        self.assertNotIn("time_since_last_inference=(time.monotonic()", transformed)

    def test_backend_requires_clock_unless_legacy_wall_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            GrootProcessGroup(Path("/checkpoint"), prompt="task", log_dir=Path("/logs"))
        backend = GrootProcessGroup(Path("/checkpoint"), prompt="task", log_dir=Path("/logs"), policy_clock="wall")
        self.assertEqual(backend.policy_clock, "wall")
        self.assertEqual(backend.left_hand_contract, "model-independent")
        with tempfile.TemporaryDirectory() as directory:
            clocked = GrootProcessGroup(
                Path("/checkpoint"), prompt="task", log_dir=Path(directory),
                policy_clock="simulation", policy_clock_file=Path("/tmp/isaac.clock"),
                capture_dir=Path("/tmp/groot-capture"), capture_max_requests=7,
                left_hand_contract="model-independent",
            )
            commands = []
            with mock.patch("humanoid_lab.psi0_bridge.groot_backend.validate_groot_checkpoint"), \
                 mock.patch.object(clocked, "_wait_server"), \
                 mock.patch.object(clocked, "_spawn", side_effect=lambda command, log: commands.append(command) or mock.Mock(poll=lambda: None)), \
                 mock.patch("humanoid_lab.psi0_bridge.groot_backend.time.sleep"):
                clocked.start()
        self.assertIn("scripts/run-groot-vla.py", commands[1])
        self.assertIn("--policy-clock=simulation", commands[1])
        self.assertIn("--policy-clock-file=/tmp/isaac.clock", commands[1])
        self.assertIn("--capture-dir=/tmp/groot-capture", commands[1])
        self.assertIn("--capture-max-requests=7", commands[1])
        self.assertIn("--left-hand-contract=model-independent", commands[1])

    def test_simulation_mode_refuses_missing_clock_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            GrootProcessGroup(Path("/checkpoint"), prompt="task", log_dir=Path("/logs"),
                              policy_clock="simulation")


if __name__ == "__main__":
    unittest.main()
