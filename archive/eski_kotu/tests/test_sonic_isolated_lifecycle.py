#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sonic_isolated_lifecycle import (  # noqa: E402
    ACTION_RATE_HZ,
    BODY_OUTPUT_SIZE,
    HAND_SIZE,
    INFERENCE_RATE_HZ,
    IsolatedLifecycle,
    LifecycleError,
    State,
    TOKEN_SIZE,
)


class IsolatedLifecycleTests(unittest.TestCase):
    token = tuple(0.125 if index == 0 else 0.0 for index in range(TOKEN_SIZE))
    hands = (0.0,) * HAND_SIZE

    def test_rates_and_tensor_contracts_are_pinned(self):
        self.assertEqual((ACTION_RATE_HZ, INFERENCE_RATE_HZ), (50.0, 2.5))
        self.assertEqual((TOKEN_SIZE, HAND_SIZE, BODY_OUTPUT_SIZE), (64, 7, 29))

    def test_shipped_harness_uses_upstream_serializer_and_subscriber(self):
        root = Path(__file__).resolve().parents[1]
        producer = (root / "scripts/sonic-isolated-vla-producer.py").read_text()
        native = (root / "tests/native/sonic_isolated_harness.cpp").read_text()
        self.assertIn("run_vla_inference.py", producer)
        self.assertIn("pack_latent_action_message", producer)
        self.assertIn("pack_pose_message", producer)
        self.assertIn("ZMQPackedMessageSubscriber", native)
        self.assertIn("1751, 64", native)
        self.assertIn("994, 29", native)

    def test_zero_hold_is_rejected(self):
        with self.assertRaisesRegex(LifecycleError, "nonzero 64D"):
            IsolatedLifecycle().initialize((0.0,) * TOKEN_SIZE)

    def test_complete_fail_closed_lifecycle(self):
        lifecycle = IsolatedLifecycle(watchdog_seconds=0.1)
        lifecycle.initialize(self.token)
        self.assertEqual(lifecycle.hold(0.0), self.token)
        lifecycle.start(1.0)
        lifecycle.validate_action(self.token, self.hands, self.hands, 1.0)
        lifecycle.pause()
        self.assertEqual(lifecycle.hold(1.01), self.token)
        lifecycle.start(1.02)
        lifecycle.validate_action(self.token, self.hands, self.hands, 1.02)
        self.assertEqual(lifecycle.tick(1.13), State.TIMED_OUT)
        self.assertEqual(lifecycle.hold(1.13), self.token)
        lifecycle.stop()
        lifecycle.reset()
        self.assertEqual(lifecycle.state, State.RESET)

    def test_malformed_actions_and_invalid_transitions_fail(self):
        lifecycle = IsolatedLifecycle()
        with self.assertRaises(LifecycleError):
            lifecycle.start(0.0)
        lifecycle.initialize(self.token)
        lifecycle.start(0.0)
        with self.assertRaisesRegex(LifecycleError, r"64\+7\+7"):
            lifecycle.validate_action(self.token[:-1], self.hands, self.hands, 0.0)
        with self.assertRaises(LifecycleError):
            lifecycle.reset()


if __name__ == "__main__":
    unittest.main()
