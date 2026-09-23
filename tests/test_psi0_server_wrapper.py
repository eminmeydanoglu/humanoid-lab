from __future__ import annotations

import importlib.util
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from humanoid_lab.psi0_bridge.policy_clock import PolicyClockError, PolicyTime

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("serve_psi0_sonic_wrapper",
                                              ROOT / "scripts/serve-psi0-sonic.py")
WRAPPER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WRAPPER)


class FakeTensor:
    def __init__(self, value=None):
        self.value = (np.arange(12, dtype=np.float32).reshape(1, 3, 4)
                      if value is None else value)

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, item):
        return FakeTensor(self.value[item])


class SyntheticClock:
    mode = "simulation"

    def __init__(self, values=(), error: Exception | None = None):
        self.values = iter(values)
        self.error = error
        self.calls = 0
        self.last = None

    def now(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        try:
            self.last = next(self.values)
        except StopIteration:
            pass
        if self.last is None:
            raise RuntimeError("synthetic clock has no value")
        return PolicyTime(*self.last)

    def latency_seconds(self, start, end=None):
        finish = self.now() if end is None else end
        if finish.generation != start.generation or finish.seconds < start.seconds:
            raise PolicyClockError("inference crossed an Isaac simulation reset")
        return finish.seconds - start.seconds


class SyntheticModel:
    device = torch.device("cpu")

    def __init__(self):
        self.initial_calls = 0
        self.rtc_calls = 0
        self.background_started = threading.Event()
        self.background_release = threading.Event()
        self.block_background = False
        self.background_error = None

    @staticmethod
    def actions():
        return torch.zeros((1, 30, 4), dtype=torch.float32)

    def predict_action(self, **_kwargs):
        self.initial_calls += 1
        return self.actions()

    def predict_action_with_rtc_flow(self, **_kwargs):
        self.rtc_calls += 1
        if self.rtc_calls > 2:
            self.background_started.set()
            if self.block_background:
                self.background_release.wait(timeout=2.0)
            if self.background_error is not None:
                raise self.background_error
        return self.actions()


class ServerWrapperTest(unittest.TestCase):
    def test_serve_constructs_pinned_transport_with_rtc_disabled(self) -> None:
        config = mock.Mock(
            policy="psi", ckpt_step=40000, host="127.0.0.1", port=18014,
            run_dir="/tmp/fake-run", device="cpu", rtc=False,
            action_exec_horizon=3, dashboard=False,
        )
        instance = mock.Mock()
        with mock.patch.object(WRAPPER.upstream, "_check_port_free") as check_port, \
             mock.patch.object(WRAPPER.upstream, "Server", return_value=instance) as server:
            WRAPPER._serve(config)
        check_port.assert_called_once_with("127.0.0.1", 18014)
        server.assert_called_once_with(
            "psi", WRAPPER.upstream.Path("/tmp/fake-run"), 40000, "cpu", False, 3,
            dashboard=False,
        )
        instance.run.assert_called_once_with("127.0.0.1", 18014)

    def server(self, *, rtc: bool):
        model = mock.Mock()
        model.predict_action.return_value = FakeTensor()
        return mock.Mock(
            enable_rtc=rtc,
            model=model,
            device="cpu",
            trained_rtc=False,
            rtc_max_delay=8,
            _pending_init_prev=None,
            Ta=2,
        )

    def test_guided_default_selects_upstream_controller(self) -> None:
        server = self.server(rtc=True)
        sentinel = object()
        with mock.patch.object(WRAPPER.upstream, "RealTimeChunkController", return_value=sentinel) as rtc:
            self.assertIs(WRAPPER._init_controller(server, {"obs": 1}), sentinel)
        rtc.assert_called_once_with(policy=server.model, o_first={"obs": 1}, trained_rtc=False,
                                    max_delay=8, init_prev_action=None)
        server.model.predict_action.assert_not_called()

    def test_simulation_clock_rtc_uses_tick_basis_adapter_without_guidance_changes(self) -> None:
        server = self.server(rtc=True)
        sentinel = object()
        with mock.patch.dict("os.environ", {"HUMANOID_POLICY_CLOCK": "simulation"}), \
             mock.patch.object(WRAPPER, "SimulationClockRtcController", return_value=sentinel) as rtc:
            self.assertIs(WRAPPER._init_controller(server, {"obs": 1}), sentinel)
        rtc.assert_called_once_with(policy=server.model, o_first={"obs": 1}, trained_rtc=False,
                                    max_delay=8, init_prev_action=None)

    @staticmethod
    def observation():
        return {
            "imgs": ["image"],
            "obs": np.zeros((1, 4), dtype=np.float32),
            "text_instructions": ["task"],
            "pooled_projections": None,
        }

    def simulation_controller(self, model, clock):
        patcher = mock.patch.object(WRAPPER.PolicyClock, "from_environment", return_value=clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        controller = WRAPPER.SimulationClockRtcController(
            policy=model,
            o_first=self.observation(),
            min_exec_horizon=3,
            d_init=1,
            max_delay=8,
        )
        self.addCleanup(controller.stop)
        return controller

    def test_simulation_controller_initial_prediction_is_synchronous_without_clock(self) -> None:
        model = SyntheticModel()
        clock = SyntheticClock(error=PolicyClockError("clock unavailable"))
        controller = self.simulation_controller(model, clock)
        self.assertEqual(model.initial_calls, 2)
        self.assertEqual(model.rtc_calls, 2)
        self.assertEqual(clock.calls, 0)
        self.assertEqual(controller.step(self.observation()).shape, (1, 4))

    def test_idle_controller_does_not_arm_the_stale_clock_watchdog(self) -> None:
        model = SyntheticModel()
        clock = SyntheticClock(error=PolicyClockError("clock stopped advancing"))
        self.simulation_controller(model, clock)
        time.sleep(0.02)
        self.assertEqual(clock.calls, 0)

    def test_idle_reset_is_absorbed_by_a_fresh_controller(self) -> None:
        first = self.simulation_controller(SyntheticModel(), SyntheticClock([(9.0, 0)]))
        first.stop()
        model = SyntheticModel()
        second_clock = SyntheticClock([(0.0, 1), (0.04, 1)])
        second = self.simulation_controller(model, second_clock)
        for _ in range(3):
            second.step(self.observation())
        self.assertTrue(model.background_started.wait(timeout=1.0))
        deadline = time.monotonic() + 1.0
        while len(second.Q) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(second.Q[-1], 1)

    def test_inference_delay_uses_sim_time_while_t_counts_only_executed_rows(self) -> None:
        model = SyntheticModel()
        model.block_background = True
        clock = SyntheticClock([(2.0, 0), (2.30, 0)])
        controller = self.simulation_controller(model, clock)
        for _ in range(3):
            controller.step(self.observation())
        self.assertTrue(model.background_started.wait(timeout=1.0))
        for _ in range(2):
            controller.step(self.observation())
        model.background_release.set()
        deadline = time.monotonic() + 1.0
        while len(controller.Q) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(controller.t, 2)
        # 0.30 s of simulator time is 9 rows at the control rate, above this
        # controller's declared max_delay of 8, so the stored delay is the bound.
        self.assertEqual(controller.Q[-1], 8)

    def test_active_clock_reset_is_fatal_and_does_not_install_old_inference(self) -> None:
        model = SyntheticModel()
        clock = SyntheticClock([(2.0, 0), (0.0, 1)])
        fatal = threading.Event()
        controller = self.simulation_controller(model, clock)
        controller._fatal = lambda _exc: fatal.set()
        old_actions = controller.A_cur
        for _ in range(3):
            controller.step(self.observation())
        self.assertTrue(fatal.wait(timeout=1.0))
        self.assertIs(controller.A_cur, old_actions)

    def test_active_stale_clock_terminates_the_simulation_control_loop(self) -> None:
        server = mock.Mock()
        server._control_stop.is_set.return_value = False
        server._conn_generation = 1
        stale = SyntheticClock(error=PolicyClockError("clock stopped advancing"))
        with mock.patch.dict("os.environ", {"HUMANOID_POLICY_CLOCK": "simulation"}), \
             mock.patch.object(WRAPPER.PolicyClock, "from_environment", return_value=stale), \
             mock.patch.object(WRAPPER.os, "_exit", side_effect=SystemExit(1)) as exit_process, \
             self.assertRaises(SystemExit):
            WRAPPER._control_loop(server, 1)
        exit_process.assert_called_once_with(1)

    def test_background_model_exception_is_reported_fatal(self) -> None:
        model = SyntheticModel()
        model.background_error = RuntimeError("synthetic inference failed")
        controller = self.simulation_controller(
            model, SyntheticClock([(1.0, 0), (1.1, 0)])
        )
        errors = []
        fatal = threading.Event()
        controller._fatal = lambda exc: (errors.append(exc), fatal.set())
        for _ in range(3):
            controller.step(self.observation())
        self.assertTrue(fatal.wait(timeout=1.0))
        self.assertRegex(str(errors[0]), "synthetic inference failed")

    def test_rtc_off_selects_independent_prediction_without_previous_actions(self) -> None:
        server = self.server(rtc=False)
        observation = {
            "imgs": ["image"], "obs": np.zeros((1, 4), dtype=np.float32),
            "text_instructions": ["task"], "pooled_projections": None,
        }
        controller = WRAPPER._init_controller(server, observation)
        self.addCleanup(controller.stop)
        kwargs = server.model.predict_action.call_args.kwargs
        self.assertNotIn("prev_actions", kwargs)
        self.assertNotIn("inference_delay", kwargs)
        self.assertEqual(controller.execution_horizon, 2)
        self.assertEqual(controller.step(observation).shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
