from __future__ import annotations

import unittest

from humanoid_lab.psi0_bridge.sim_clock import (
    SimulationClockError,
    SimulationClockPacer,
    clamp_rtc_delay,
)


class SimulationClockPacerTest(unittest.TestCase):
    def test_rows_follow_simulation_time_not_wall_poll_count(self) -> None:
        pacer = SimulationClockPacer(50.0)
        self.assertEqual(pacer.update(10.0, 4).rows_due, 1)
        for _ in range(20):
            update = pacer.update(10.0, 4)
            self.assertEqual(update.rows_due, 0)
            self.assertTrue(update.stalled)
        self.assertEqual(pacer.update(10.019, 4).rows_due, 0)
        self.assertEqual(pacer.update(10.020, 4).rows_due, 1)
        self.assertEqual(pacer.update(10.040, 4).rows_due, 1)

    def test_large_simulation_step_skips_stale_intermediate_rows(self) -> None:
        pacer = SimulationClockPacer(50.0)
        pacer.update(0.0, 1)
        self.assertEqual(pacer.update(0.1, 1).rows_due, 5)
        self.assertEqual(pacer.update(0.1, 1).rows_due, 0)

    def test_episode_change_and_clock_regression_reset_without_carryover(self) -> None:
        pacer = SimulationClockPacer(50.0)
        pacer.update(2.0, 8)
        pacer.update(2.2, 8)
        changed = pacer.update(0.0, 9)
        self.assertTrue(changed.reset)
        self.assertEqual(changed.rows_due, 1)
        regressed = pacer.update(-1.0, 9)
        self.assertTrue(regressed.reset)
        self.assertEqual(regressed.rows_due, 1)

    def test_latency_skip_uses_the_same_simulation_time_basis(self) -> None:
        pacer = SimulationClockPacer(50.0)
        self.assertEqual(pacer.latency_rows(3.0, 3.0, 40), 0)
        self.assertEqual(pacer.latency_rows(3.0, 3.067, 40), 3)
        self.assertEqual(pacer.latency_rows(3.0, 4.0, 40), 39)
        with self.assertRaisesRegex(SimulationClockError, "reset"):
            pacer.latency_rows(3.0, 2.0, 40)

    def test_invalid_clock_values_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            SimulationClockPacer(0.0)
        pacer = SimulationClockPacer(30.0)
        with self.assertRaisesRegex(SimulationClockError, "finite"):
            pacer.update(float("nan"), 1)
        with self.assertRaises(ValueError):
            pacer.latency_rows(0.0, 0.1, 0)


class RtcDelayClampTest(unittest.TestCase):
    """The simulated RTC delay stays inside the horizon the checkpoint trained on.

    ``Psi0``'s trained RTC path is conditioned on ``model.max_delay`` (8 in the
    served runs), while the simulation clock converts a multi-second forward pass
    into a delay of tens of rows and then caps it at the *chunk* length (30).  The
    bound must therefore be the declared RTC horizon, not the chunk.
    """

    def test_a_slow_forward_pass_is_bounded_by_the_declared_max_delay(self) -> None:
        pacer = SimulationClockPacer(30.0)
        # 1.5 simulated seconds at 30 Hz is 45 rows -> capped at the chunk (29),
        # which is far past the 8 rows the trained path accepts.
        self.assertEqual(pacer.latency_rows(0.0, 1.5, 30), 29)
        self.assertEqual(clamp_rtc_delay(pacer.latency_rows(0.0, 1.5, 30), 8), 8)

    def test_a_fast_forward_pass_is_passed_through_unchanged(self) -> None:
        pacer = SimulationClockPacer(30.0)
        self.assertEqual(clamp_rtc_delay(pacer.latency_rows(0.0, 0.067, 30), 8), 2)
        self.assertEqual(clamp_rtc_delay(pacer.latency_rows(0.0, 0.0, 30), 8), 0)

    def test_the_bound_itself_is_kept(self) -> None:
        self.assertEqual(clamp_rtc_delay(8, 8), 8)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            clamp_rtc_delay(3, 0)
        with self.assertRaises(ValueError):
            clamp_rtc_delay(-1, 8)


if __name__ == "__main__":
    unittest.main()
