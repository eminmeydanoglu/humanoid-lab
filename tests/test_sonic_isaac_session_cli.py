#!/usr/bin/env python3
"""Drive the shipped ``sonic-isaac-session.py start`` launcher with unsafe input.

Each case must exit non-zero with the documented refusal reason and spawn no
child process and write no session state.  The CLI is invoked as a subprocess,
so these assertions cover the real entry point rather than a reimplementation of
its predicates.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "sonic-isaac-session.py"


def run_cli(args: list[str], *, env_overrides: dict | None = None, runtime_dir: str):
    env = dict(os.environ)
    env.pop("CYCLONEDDS_URI", None)
    env.pop("ROS_DOMAIN_ID", None)
    env.update(env_overrides or {})
    return subprocess.run(  # noqa: S603
        [sys.executable, str(CLI), "--runtime-dir", runtime_dir, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=120,
    )


class LauncherRefusalTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.runtime = self._tmp.name
        self.loopback = ROOT / "containers" / "cyclonedds-sim.xml"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def assert_refused(self, result, reason: str) -> None:
        self.assertNotEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload.get("started") is False, msg=result.stdout)
        self.assertIn(reason, payload.get("refused", []), msg=result.stdout)
        # Nothing may be spawned or recorded when a gate fails.
        self.assertFalse((Path(self.runtime) / "session.json").exists())

    def test_refuses_inline_cyclonedds_uri(self) -> None:
        result = run_cli(
            ["start", "--robot", "g1-29dof", "--input", "acceptance"],
            env_overrides={"CYCLONEDDS_URI": "<CycloneDDS/>", "ROS_DOMAIN_ID": "42"},
            runtime_dir=self.runtime,
        )
        self.assert_refused(result, "unsafe_cyclonedds_uri")

    def test_refuses_missing_cyclonedds_uri(self) -> None:
        result = run_cli(
            ["start", "--robot", "g1-29dof", "--input", "acceptance"],
            env_overrides={"ROS_DOMAIN_ID": "42"},
            runtime_dir=self.runtime,
        )
        self.assert_refused(result, "unsafe_cyclonedds_uri")

    def test_refuses_wrong_dds_domain(self) -> None:
        result = run_cli(
            ["start", "--robot", "g1-29dof", "--input", "acceptance"],
            env_overrides={
                "CYCLONEDDS_URI": f"file://{self.loopback}",
                "ROS_DOMAIN_ID": "0",
            },
            runtime_dir=self.runtime,
        )
        self.assert_refused(result, "unsafe_dds_domain")

    def test_refuses_non_sim_sonic_mode(self) -> None:
        result = run_cli(
            ["start", "--robot", "g1-29dof", "--input", "acceptance", "--sonic-mode", "real"],
            env_overrides={
                "CYCLONEDDS_URI": f"file://{self.loopback}",
                "ROS_DOMAIN_ID": "42",
            },
            runtime_dir=self.runtime,
        )
        self.assert_refused(result, "unsafe_sonic_mode")

    def test_refuses_physical_interface(self) -> None:
        result = run_cli(
            ["start", "--robot", "g1-29dof", "--input", "acceptance",
             "--sonic-interface", "enp129s0"],
            env_overrides={
                "CYCLONEDDS_URI": f"file://{self.loopback}",
                "ROS_DOMAIN_ID": "42",
            },
            runtime_dir=self.runtime,
        )
        self.assert_refused(result, "unsafe_physical_interface")

    def test_refuses_keyboard_without_a_tty(self) -> None:
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(CLI), "--runtime-dir", self.runtime,
             "start", "--robot", "g1-29dof", "--input", "keyboard"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env={
                **os.environ,
                "CYCLONEDDS_URI": f"file://{self.loopback}",
                "ROS_DOMAIN_ID": "42",
            },
            cwd=str(ROOT),
            timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("keyboard_requires_tty", result.stdout)
        self.assertFalse((Path(self.runtime) / "session.json").exists())

    def test_status_on_an_empty_runtime_reports_stopped(self) -> None:
        result = run_cli(["status"], runtime_dir=self.runtime)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        for key in (
            "simulator", "controller", "actuation", "input", "robot_profile",
            "dds_domain", "dds_interface", "lowstate_hz", "lowcmd_hz",
            "last_lowcmd_age_ms", "root_position", "root_roll_pitch_yaw",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["simulator"], "stopped")
        self.assertEqual(payload["controller"], "absent")

    def test_stop_on_an_empty_runtime_is_a_no_op(self) -> None:
        result = run_cli(["stop"], runtime_dir=self.runtime)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("no recorded session", result.stdout)


class DeployArgvTest(unittest.TestCase):
    def test_deploy_command_is_simulation_only(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib.util

        spec = importlib.util.spec_from_file_location("session_cli", CLI)
        module = importlib.util.module_from_spec(spec)
        sys.modules["session_cli"] = module
        spec.loader.exec_module(module)

        argv = module.deploy_argv(robot="g1-29dof", input_mode="keyboard")
        self.assertIn("sim", argv)
        self.assertIn("lo", argv)
        self.assertNotIn("real", argv)

    def test_f310_input_selects_the_bridge_input_type(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib.util

        spec = importlib.util.spec_from_file_location("session_cli_f310", CLI)
        module = importlib.util.module_from_spec(spec)
        sys.modules["session_cli_f310"] = module
        spec.loader.exec_module(module)

        argv = module.deploy_argv(robot="g1-inspire", input_mode="f310")
        self.assertIn("f310_bridge", argv)


if __name__ == "__main__":
    unittest.main()
