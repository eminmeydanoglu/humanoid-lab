#!/usr/bin/env python3
"""Keyboard schedules for the upstream deploy's keyboard input mode."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sonic_isaac_keyboard import (  # noqa: E402
    ARM_KEY,
    FORWARD_KEY,
    MOMENTUM_RESET_KEY,
    PLANNER_TOGGLE_KEY,
    STOP_KEY,
    KeyScheduler,
    drive_schedule,
    presses_during,
    standing_schedule,
)


class BindingTest(unittest.TestCase):
    def test_bindings_match_the_pinned_upstream_keys(self) -> None:
        # These are the pinned keyboard_handler.hpp bindings, not placeholders.
        self.assertEqual(ARM_KEY, "]")
        self.assertEqual(PLANNER_TOGGLE_KEY, "\n")
        self.assertEqual(FORWARD_KEY, "w")
        self.assertEqual(MOMENTUM_RESET_KEY, "r")
        self.assertEqual(STOP_KEY, "o")


class StandingScheduleTest(unittest.TestCase):
    def test_standing_only_arms_the_controller(self) -> None:
        schedule = standing_schedule()
        self.assertEqual([event.keys for event in schedule], [ARM_KEY])
        self.assertEqual(schedule[0].at_s, 0.0)


class DriveScheduleTest(unittest.TestCase):
    def test_sequence_follows_the_plan_order(self) -> None:
        labels = [event.label for event in drive_schedule()]
        self.assertEqual(labels[0], "arm")
        self.assertIn("planner_mode", labels)
        self.assertIn("forward", labels)
        self.assertIn("turn", labels)
        self.assertIn("momentum_reset", labels)
        self.assertEqual(labels[-1], "stop")

    def test_planner_mode_is_selected_after_the_idle_phase(self) -> None:
        schedule = drive_schedule(idle_s=10.0)
        planner = next(e for e in schedule if e.label == "planner_mode")
        forward = next(e for e in schedule if e.label == "forward")
        self.assertEqual(planner.at_s, 10.0)
        self.assertGreater(forward.at_s, planner.at_s)

    def test_forward_precedes_the_turn_and_the_turn_precedes_the_reset(self) -> None:
        schedule = drive_schedule()
        first = lambda label: next(e.at_s for e in schedule if e.label == label)
        self.assertLess(first("forward"), first("turn"))
        self.assertLess(first("turn"), first("momentum_reset"))
        self.assertLess(first("momentum_reset"), first("stop"))

    def test_release_phase_is_long_enough_to_observe_idle_decay(self) -> None:
        # The plan requires the speed to decay to Idle within 5 s of release.
        schedule = drive_schedule(release_s=5.0, forward_s=3.0)
        forward_events = [e for e in schedule if e.label == "forward"]
        turn_events = [e for e in schedule if e.label == "turn"]
        gap = min(e.at_s for e in turn_events) - max(e.at_s for e in forward_events)
        self.assertGreaterEqual(gap, 5.0 - 1e-9)

    def test_turn_direction_is_configurable(self) -> None:
        left = {e.keys for e in drive_schedule(turn_key="a") if e.label == "turn"}
        right = {e.keys for e in drive_schedule(turn_key="d") if e.label == "turn"}
        self.assertEqual(left, {"a"})
        self.assertEqual(right, {"d"})

    def test_forward_is_held_by_repetition(self) -> None:
        events = presses_during(FORWARD_KEY, 1.0, 3.0, 20.0, "forward")
        self.assertEqual(len(events), 60)
        self.assertEqual(events[0].at_s, 1.0)
        self.assertAlmostEqual(events[-1].at_s, 1.0 + 59 * 0.05, places=9)


class SchedulerTest(unittest.TestCase):
    def test_events_are_released_in_time_order(self) -> None:
        scheduler = KeyScheduler(drive_schedule(idle_s=10.0))
        self.assertEqual(scheduler.due(0.0)[0].label, "arm")
        self.assertEqual(scheduler.due(5.0), [])
        self.assertEqual(scheduler.due(10.0)[0].label, "planner_mode")

    def test_no_event_is_released_twice(self) -> None:
        scheduler = KeyScheduler(standing_schedule())
        first = scheduler.due(0.0)
        self.assertEqual(len(first), 1)
        self.assertEqual(scheduler.due(1.0), [])
        self.assertTrue(scheduler.finished)

    def test_duration_covers_the_last_event(self) -> None:
        schedule = drive_schedule()
        self.assertAlmostEqual(KeyScheduler(schedule).duration_s, schedule[-1].at_s, places=9)

    def test_scheduler_accepts_an_unordered_schedule(self) -> None:
        from sonic_isaac_keyboard import KeyEvent

        scheduler = KeyScheduler((KeyEvent(5.0, "b", "later"), KeyEvent(1.0, "a", "earlier")))
        self.assertEqual(scheduler.due(2.0)[0].label, "earlier")
        self.assertEqual(scheduler.due(6.0)[0].label, "later")


if __name__ == "__main__":
    unittest.main()
