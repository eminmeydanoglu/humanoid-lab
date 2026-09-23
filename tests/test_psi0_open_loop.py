from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from humanoid_lab.psi0_bridge.open_loop import OpenLoopChunkController


def chunk(base: int, horizon: int = 5) -> np.ndarray:
    return np.arange(base, base + horizon, dtype=np.float32)[:, None]


class FakePredictor:
    def __init__(self, chunks: list[np.ndarray], gate: threading.Event | None = None) -> None:
        self.chunks = list(chunks)
        self.gate = gate
        self.calls: list[object] = []

    def __call__(self, observation):
        self.calls.append(observation)
        if len(self.calls) > 1 and self.gate is not None:
            self.gate.wait(timeout=2.0)
        return self.chunks.pop(0)


class OpenLoopChunkControllerTest(unittest.TestCase):
    def test_independent_prediction_and_exact_horizon_progression(self) -> None:
        predictor = FakePredictor([chunk(0), chunk(100)])
        controller = OpenLoopChunkController(predictor, "initial", execution_horizon=3)
        self.addCleanup(controller.stop)
        self.assertEqual([controller.step(f"obs-{i}").item() for i in range(3)], [0, 1, 2])
        deadline = time.monotonic() + 2.0
        while len(predictor.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(predictor.calls, ["initial", "obs-2"])
        self.assertEqual(controller.step("after").item(), 100)

    def test_delayed_prediction_holds_last_executed_row_then_starts_new_chunk_at_zero(self) -> None:
        gate = threading.Event()
        predictor = FakePredictor([chunk(0), chunk(100)], gate)
        controller = OpenLoopChunkController(predictor, "initial", execution_horizon=3)
        self.addCleanup(controller.stop)
        self.assertEqual([controller.step({}).item() for _ in range(3)], [0, 1, 2])
        self.assertEqual([controller.step({}).item() for _ in range(4)], [2, 2, 2, 2])
        gate.set()
        deadline = time.monotonic() + 2.0
        value = None
        while time.monotonic() < deadline:
            value = controller.step({}).item()
            if value == 100:
                break
            time.sleep(0.01)
        self.assertEqual(value, 100)
        self.assertEqual(controller.step({}).item(), 101)

    def test_full_chunk_horizon_and_invalid_horizons(self) -> None:
        controller = OpenLoopChunkController(FakePredictor([chunk(10), chunk(20)]), {},
                                             execution_horizon=5)
        self.addCleanup(controller.stop)
        self.assertEqual([controller.step({}).item() for _ in range(5)], [10, 11, 12, 13, 14])
        with self.assertRaisesRegex(ValueError, "not in"):
            OpenLoopChunkController(FakePredictor([chunk(0)]), {}, execution_horizon=0)
        with self.assertRaisesRegex(ValueError, "not in"):
            OpenLoopChunkController(FakePredictor([chunk(0)]), {}, execution_horizon=6)

    def test_background_inference_failure_is_reported_on_the_action_thread(self) -> None:
        class FailingPredictor:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, observation):
                self.calls += 1
                if self.calls == 1:
                    return chunk(0)
                raise ValueError("model failed")

        predictor = FailingPredictor()
        controller = OpenLoopChunkController(predictor, {}, execution_horizon=2)
        self.addCleanup(controller.stop)
        self.assertEqual([controller.step({}).item() for _ in range(2)], [0, 1])
        deadline = time.monotonic() + 2.0
        while predictor.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        with self.assertRaisesRegex(RuntimeError, "inference failed") as raised:
            controller.step({})
        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_stop_discards_delayed_chunk_and_new_controller_resets_to_row_zero(self) -> None:
        gate = threading.Event()
        first_predictor = FakePredictor([chunk(0), chunk(100)], gate)
        first = OpenLoopChunkController(first_predictor, "episode-1", execution_horizon=2)
        self.assertEqual([first.step({}).item() for _ in range(2)], [0, 1])
        first.stop(join_timeout=0.01)
        gate.set()

        second_predictor = FakePredictor([chunk(200)])
        second = OpenLoopChunkController(second_predictor, "episode-2", execution_horizon=2)
        self.addCleanup(second.stop)
        self.assertEqual(second.step({}).item(), 200)
        self.assertEqual(second_predictor.calls, ["episode-2"])


if __name__ == "__main__":
    unittest.main()
