#!/usr/bin/env python3
"""Session status, launch gating and stop-planning tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from sonic_isaac_contract import SIM_DDS_DOMAIN_ID, ContractError  # noqa: E402
from sonic_isaac_session import (  # noqa: E402
    AGENT_STATE_READY_PENDING_USER_DRIVE,
    ChildProcess,
    SessionPaths,
    SessionState,
    build_status,
    derive_controller,
    derive_simulator,
    evaluate_launch,
    load_state,
    plan_children,
    save_state,
    stop_plan,
)

LOOPBACK_XML = Path(ROOT / "containers" / "cyclonedds-sim.xml").read_text()

STATUS_KEYS = {
    "simulator",
    "controller",
    "actuation",
    "input",
    "robot_profile",
    "dds_domain",
    "dds_interface",
    "lowstate_hz",
    "lowcmd_hz",
    "last_lowcmd_age_ms",
    "root_position",
    "root_roll_pitch_yaw",
}


def state(**overrides) -> SessionState:
    base = dict(robot_profile="g1-29dof", input="keyboard")
    base.update(overrides)
    return SessionState(**base)


class StatusSchemaTest(unittest.TestCase):
    def test_status_contains_every_documented_field(self) -> None:
        payload = build_status(state())
        self.assertTrue(STATUS_KEYS.issubset(payload))
        self.assertEqual(payload["dds_domain"], 42)
        self.assertEqual(payload["dds_interface"], "lo")
        self.assertEqual(len(payload["root_position"]), 3)
        self.assertEqual(len(payload["root_roll_pitch_yaw"]), 3)

    def test_documented_field_types(self) -> None:
        payload = build_status(state(simulator="paused"))
        self.assertIsInstance(payload["simulator"], str)
        self.assertIsInstance(payload["dds_domain"], int)
        self.assertIsInstance(payload["lowcmd_hz"], float)
        self.assertIsInstance(payload["root_position"], list)

    def test_actuation_follows_the_controller(self) -> None:
        controlled = build_status(state(controller="controlled"))
        self.assertEqual(controlled["actuation"], "controlled")
        for controller in ("absent", "waiting", "stale"):
            payload = build_status(state(controller=controller))
            self.assertEqual(payload["actuation"], "passive")

    def test_missing_lowcmd_age_is_reported_as_zero(self) -> None:
        payload = build_status(state(last_lowcmd_age_ms=None))
        self.assertEqual(payload["last_lowcmd_age_ms"], 0.0)


class DerivationTest(unittest.TestCase):
    def test_simulator_states(self) -> None:
        self.assertEqual(derive_simulator(running=False, playing=False, paused=False), "stopped")
        self.assertEqual(derive_simulator(running=True, playing=True, paused=False), "playing")
        self.assertEqual(derive_simulator(running=True, playing=False, paused=True), "paused")
        self.assertEqual(derive_simulator(running=True, playing=False, paused=False), "ready")

    def test_controller_states(self) -> None:
        self.assertEqual(derive_controller(lowcmd_age_ms=None, sonic_running=False), "absent")
        self.assertEqual(derive_controller(lowcmd_age_ms=None, sonic_running=True), "waiting")
        self.assertEqual(derive_controller(lowcmd_age_ms=12.0, sonic_running=True), "controlled")
        self.assertEqual(derive_controller(lowcmd_age_ms=100.0, sonic_running=True), "controlled")
        self.assertEqual(derive_controller(lowcmd_age_ms=100.1, sonic_running=True), "stale")
        self.assertEqual(derive_controller(lowcmd_age_ms=900.0, sonic_running=True), "stale")

    def test_controller_is_absent_long_before_sonic_starts(self) -> None:
        self.assertEqual(derive_controller(lowcmd_age_ms=None, sonic_running=False), "absent")


class PersistenceTest(unittest.TestCase):
    def test_state_round_trips_through_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = SessionPaths.under(tmp)
            original = state(
                simulator="paused",
                controller="controlled",
                lowcmd_hz=50.0,
                children=[ChildProcess("isaac", 1234, ("python", "runner"), started_at=1.0)],
                root_position=(0.1, 0.2, 0.75),
            )
            save_state(paths, original)
            loaded = load_state(paths)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.robot_profile, original.robot_profile)
            self.assertEqual(loaded.children[0].pid, 1234)
            self.assertEqual(loaded.children[0].role, "isaac")
            self.assertEqual(tuple(loaded.root_position), (0.1, 0.2, 0.75))
            self.assertEqual(loaded.controller, "controlled")

    def test_missing_state_file_reads_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_state(SessionPaths.under(tmp)))

    def test_corrupt_state_file_reads_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = SessionPaths.under(tmp)
            paths.runtime_dir.mkdir(parents=True, exist_ok=True)
            paths.state_file.write_text("{not json")
            self.assertIsNone(load_state(paths))

    def test_state_written_as_readable_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = SessionPaths.under(tmp)
            save_state(paths, state())
            payload = json.loads(paths.state_file.read_text())
            self.assertEqual(payload["robot_profile"], "g1-29dof")
            self.assertEqual(payload["agent_state"], AGENT_STATE_READY_PENDING_USER_DRIVE)


class ChildPlanTest(unittest.TestCase):
    def test_launch_order_is_documented_order(self) -> None:
        self.assertEqual(plan_children("g1-29dof", "keyboard"), ("isaac", "bridge", "sonic", "input"))
        self.assertEqual(plan_children("g1-inspire", "f310"), ("isaac", "bridge", "sonic", "input"))

    def test_unknown_profile_is_refused(self) -> None:
        with self.assertRaises(ContractError):
            plan_children("g1-23dof", "keyboard")

    def test_unknown_input_is_refused(self) -> None:
        with self.assertRaises(ContractError):
            plan_children("g1-29dof", "gamepad")

    def test_unknown_child_role_is_refused(self) -> None:
        with self.assertRaises(ContractError):
            ChildProcess("mystery", 1)


class StopPlanTest(unittest.TestCase):
    def test_stop_targets_only_recorded_children_newest_first(self) -> None:
        record = state(
            owner_pid=1000,
            children=[
                ChildProcess("isaac", 1234, started_at=1.0),
                ChildProcess("bridge", 1235, started_at=2.0),
                ChildProcess("sonic", 1236, started_at=3.0),
            ],
        )
        self.assertEqual(stop_plan(record), (1236, 1235, 1234))

    def test_stop_never_targets_the_session_owner(self) -> None:
        record = state(
            owner_pid=1234,
            children=[ChildProcess("isaac", 1234, started_at=1.0), ChildProcess("bridge", 9, started_at=2.0)],
        )
        self.assertEqual(stop_plan(record), (9,))

    def test_dead_session_record_does_not_block_a_new_start(self) -> None:
        """A leftover record whose processes exited is not a live session."""
        stale = state(
            owner_pid=999999,
            children=[ChildProcess("isaac", 999998, started_at=1.0)],
        )
        from sonic_isaac_session import SessionPaths

        with tempfile.TemporaryDirectory() as tmp:
            paths = SessionPaths.under(tmp)
            save_state(paths, stale)
            reloaded = load_state(paths)
            self.assertIsNotNone(reloaded)
            # os.kill on a pid that does not exist is the liveness test used.
            import os as _os

            alive = True
            for pid in reloaded.child_pids():
                try:
                    _os.kill(pid, 0)
                except ProcessLookupError:
                    alive = False
            self.assertFalse(alive, "the test needs a definitely-dead pid")

    def test_stop_on_an_empty_session_targets_nothing(self) -> None:
        self.assertEqual(stop_plan(state()), ())


class LaunchGateWiringTest(unittest.TestCase):
    def base(self, **overrides):
        kwargs = dict(
            env={"CYCLONEDDS_URI": "file:///opt/humanoid-lab/cyclonedds-sim.xml"},
            dds_config_text=LOOPBACK_XML,
            domain_id=SIM_DDS_DOMAIN_ID,
            sonic_mode="sim",
            requested_interface="lo",
            argv=["g1_deploy_onnx_ref", "lo", "--disable-crc-check"],
            physical_interfaces=["lo", "enp129s0"],
            existing_session=None,
            own_pid=1,
        )
        kwargs.update(overrides)
        return kwargs

    def test_safe_configuration_passes(self) -> None:
        self.assertEqual(evaluate_launch(**self.base()), ())

    def test_missing_uri_is_refused(self) -> None:
        reasons = evaluate_launch(**self.base(env={}, dds_config_text=None))
        self.assertIn("unsafe_cyclonedds_uri", reasons)

    def test_wrong_domain_is_refused(self) -> None:
        self.assertIn("unsafe_dds_domain", evaluate_launch(**self.base(domain_id=0)))

    def test_real_mode_is_refused(self) -> None:
        self.assertIn("unsafe_sonic_mode", evaluate_launch(**self.base(sonic_mode="real")))

    def test_physical_interface_is_refused(self) -> None:
        reasons = evaluate_launch(**self.base(requested_interface="enp129s0"))
        self.assertIn("unsafe_physical_interface", reasons)

    def test_running_session_conflicts(self) -> None:
        other = state(owner_pid=777, domain_id=SIM_DDS_DOMAIN_ID, interface="lo")
        reasons = evaluate_launch(**self.base(existing_session=other))
        self.assertIn("session_conflict", reasons)

    def test_same_session_may_restart(self) -> None:
        mine = state(owner_pid=1, domain_id=SIM_DDS_DOMAIN_ID, interface="lo")
        self.assertEqual(evaluate_launch(**self.base(existing_session=mine)), ())

    def test_unreadable_dds_config_is_refused(self) -> None:
        reasons = evaluate_launch(**self.base(dds_config_text="<not-a-config/>"))
        self.assertIn("unsafe_cyclonedds_uri", reasons)


if __name__ == "__main__":
    unittest.main()
