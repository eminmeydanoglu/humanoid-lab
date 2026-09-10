#!/usr/bin/env python3
"""Actuation and fail-safe tests driving the shipped ``BodyActuation``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sonic_isaac_actuation import (  # noqa: E402
    ActuationMode,
    BodyActuation,
    LowCmdSample,
    RateMeter,
    body_effort,
)
from sonic_isaac_contract import BODY_JOINT_COUNT, LOWCMD_MAX_AGE_S  # noqa: E402

EFFORT_LIMIT = 88.0


def vector(value: float) -> tuple[float, ...]:
    return (value,) * BODY_JOINT_COUNT


def sample(
    sequence: int,
    received_at: float,
    *,
    q=0.0,
    dq=0.0,
    tau=0.0,
    kp=0.0,
    kd=0.0,
) -> LowCmdSample:
    """Build a lowcmd where each scalar is broadcast across all 29 joints."""
    def expand(value):
        return value if isinstance(value, tuple) else vector(float(value))

    return LowCmdSample(
        sequence=sequence,
        received_at=received_at,
        q=expand(q),
        dq=expand(dq),
        tau=expand(tau),
        kp=expand(kp),
        kd=expand(kd),
    )


class EffortLawTest(unittest.TestCase):
    def test_effort_matches_upstream_mujoco_equation(self) -> None:
        # tau_ff + kp*(q_des - q) + kd*(dq_des - dq)
        # 0.5 + 10.0*(1.0 - 0.9) + 1.0*(0.0 - 0.1) = 1.4
        efforts = body_effort(
            q_cmd=(1.0,), dq_cmd=(0.0,), tau_ff=(0.5,), kp=(10.0,), kd=(1.0,),
            q=(0.9,), dq=(0.1,), effort_limits=(EFFORT_LIMIT,),
        )
        self.assertEqual(len(efforts), 1)
        self.assertAlmostEqual(efforts[0], 1.4, places=9)

    def test_effort_is_zero_when_tracking_perfectly_without_feedforward(self) -> None:
        efforts = body_effort(
            q_cmd=(0.3,), dq_cmd=(0.2,), tau_ff=(0.0,), kp=(50.0,), kd=(2.0,),
            q=(0.3,), dq=(0.2,), effort_limits=(EFFORT_LIMIT,),
        )
        self.assertAlmostEqual(efforts[0], 0.0, places=9)

    def test_positive_effort_is_clamped_to_limit(self) -> None:
        efforts = body_effort(
            q_cmd=(2.0,), dq_cmd=(0.0,), tau_ff=(0.0,), kp=(1000.0,), kd=(0.0,),
            q=(0.0,), dq=(0.0,), effort_limits=(88.0,),
        )
        self.assertEqual(efforts[0], 88.0)

    def test_negative_effort_is_clamped_to_limit(self) -> None:
        efforts = body_effort(
            q_cmd=(0.0,), dq_cmd=(0.0,), tau_ff=(-5.0,), kp=(0.0,), kd=(0.0,),
            q=(0.0,), dq=(0.0,), effort_limits=(2.0,),
        )
        self.assertEqual(efforts[0], -2.0)

    def test_non_finite_effort_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            body_effort(
                q_cmd=(float("nan"),), dq_cmd=(0.0,), tau_ff=(0.0,), kp=(0.0,), kd=(0.0,),
                q=(0.0,), dq=(0.0,), effort_limits=(1.0,),
            )

    def test_mismatched_lengths_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            body_effort(
                q_cmd=(0.0, 0.0), dq_cmd=(0.0,), tau_ff=(0.0,), kp=(0.0,), kd=(0.0,),
                q=(0.0, 0.0), dq=(0.0, 0.0), effort_limits=(1.0, 1.0),
            )


class PassiveStateTest(unittest.TestCase):
    def test_passive_before_any_controller_command(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        step = gate.step(0.0, vector(0.0), vector(0.0))
        self.assertEqual(step.mode, ActuationMode.PASSIVE)
        self.assertEqual(step.reason, "no_controller")
        self.assertEqual(step.efforts, (0.0,) * BODY_JOINT_COUNT)

    def test_controlled_while_lowcmd_is_fresh(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        gate.submit(sample(1, 10.0, q=0.4, kp=100.0))
        step = gate.step(10.05, vector(0.0), vector(0.0))
        self.assertEqual(step.mode, ActuationMode.CONTROLLED)
        self.assertAlmostEqual(step.efforts[0], 40.0, places=9)

    def test_holds_the_command_for_exactly_one_hundred_ms(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        gate.submit(sample(1, 10.0, q=0.4, kp=100.0))
        boundary = gate.step(10.0 + LOWCMD_MAX_AGE_S, vector(0.0), vector(0.0))
        self.assertEqual(boundary.mode, ActuationMode.CONTROLLED)

    def test_goes_passive_and_zeroes_effort_past_one_hundred_ms(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        gate.submit(sample(1, 10.0, q=0.4, kp=100.0))
        step = gate.step(10.0 + LOWCMD_MAX_AGE_S + 1e-6, vector(0.0), vector(0.0))
        self.assertEqual(step.mode, ActuationMode.PASSIVE)
        self.assertEqual(step.reason, "lowcmd_stale")
        self.assertEqual(step.efforts, (0.0,) * BODY_JOINT_COUNT)

    def test_recovers_from_passive_when_a_fresh_command_arrives(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        gate.submit(sample(1, 10.0, q=0.4, kp=100.0))
        self.assertEqual(gate.step(10.5, vector(0.0), vector(0.0)).mode, ActuationMode.PASSIVE)
        gate.submit(sample(2, 10.6, q=0.4, kp=100.0))
        self.assertEqual(gate.step(10.65, vector(0.0), vector(0.0)).mode, ActuationMode.CONTROLLED)

    def test_invalidate_forces_passive_for_the_next_step(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        gate.submit(sample(1, 10.0, q=0.4, kp=100.0))
        gate.invalidate()
        step = gate.step(10.01, vector(0.0), vector(0.0))
        self.assertEqual(step.mode, ActuationMode.PASSIVE)
        self.assertEqual(step.reason, "no_controller")


class MessageRejectionTest(unittest.TestCase):
    def test_out_of_order_sequence_is_not_applied(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        self.assertTrue(gate.submit(sample(7, 10.0, q=0.4, kp=100.0)).accepted)
        replay = gate.submit(sample(7, 10.02, q=9.9, kp=100.0))
        self.assertFalse(replay.accepted)
        self.assertEqual(replay.reason, "out_of_order")
        backward = gate.submit(sample(3, 10.03, q=9.9, kp=100.0))
        self.assertFalse(backward.accepted)
        # The stale command is still the one applied.
        self.assertAlmostEqual(gate.step(10.04, vector(0.0), vector(0.0)).efforts[0], 40.0, places=9)

    def test_newer_sequence_replaces_the_command(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        gate.submit(sample(1, 10.0, q=0.4, kp=100.0))
        gate.submit(sample(2, 10.02, q=0.5, kp=100.0))
        self.assertAlmostEqual(gate.step(10.03, vector(0.0), vector(0.0)).efforts[0], 50.0, places=9)

    def test_nan_command_is_rejected_and_never_applied(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        gate.submit(sample(1, 10.0, q=0.4, kp=100.0))
        poisoned = sample(2, 10.01, q=0.4, kp=100.0)
        poisoned = LowCmdSample(
            sequence=poisoned.sequence,
            received_at=poisoned.received_at,
            q=(float("nan"),) + poisoned.q[1:],
            dq=poisoned.dq,
            tau=poisoned.tau,
            kp=poisoned.kp,
            kd=poisoned.kd,
        )
        outcome = gate.submit(poisoned)
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "non_finite:q")
        # The previous valid command still governs the step.
        self.assertAlmostEqual(gate.step(10.02, vector(0.0), vector(0.0)).efforts[0], 40.0, places=9)

    def test_inf_gain_is_rejected(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        outcome = gate.submit(sample(1, 10.0, q=0.4, kp=float("inf")))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "non_finite:kp")

    def test_rejection_counts_are_reported(self) -> None:
        gate = BodyActuation([EFFORT_LIMIT] * BODY_JOINT_COUNT)
        gate.submit(sample(5, 10.0))
        gate.submit(sample(5, 10.0))
        gate.submit(sample(6, 10.1, kp=float("nan")))
        snapshot = gate.snapshot()
        self.assertEqual(snapshot["accepted"], 1)
        self.assertEqual(snapshot["rejected"], 2)
        self.assertIn("out_of_order", gate.rejections)
        self.assertIn("non_finite:kp", gate.rejections)


class RateMeterTest(unittest.TestCase):
    def test_reports_fifty_hz_for_a_regular_fifty_hz_stream(self) -> None:
        meter = RateMeter()
        for index in range(51):
            meter.mark(index * 0.02)
        self.assertAlmostEqual(meter.mean_hz(), 50.0, places=6)
        self.assertAlmostEqual(meter.max_gap_ms(), 20.0, places=6)

    def test_reports_the_worst_gap(self) -> None:
        meter = RateMeter()
        for timestamp in (0.0, 0.02, 0.04, 0.19, 0.21):
            meter.mark(timestamp)
        self.assertAlmostEqual(meter.max_gap_ms(), 150.0, places=6)

    def test_empty_meter_is_zero(self) -> None:
        self.assertEqual(RateMeter().mean_hz(), 0.0)
        self.assertEqual(RateMeter().max_gap_ms(), 0.0)


if __name__ == "__main__":
    unittest.main()
