#!/usr/bin/env python3
"""Drive the shipped ``dev.sh`` dispatch with a stub ``docker`` on PATH.

The dispatch builds a container command from the user's arguments. A bug there
is invisible to unit tests that call the session script directly, so this runs
the real ``dev.sh`` and inspects the command it would hand to Docker.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_SH = ROOT / "dev.sh"

FAKE_DOCKER = """#!/usr/bin/env bash
# Stub: record the invocation instead of touching Docker.
printf 'DOCKER_CALL: %s\\n' "$*"
exit 0
"""

MINIMAL_ENV = """COMPOSE_PROJECT_NAME=humanoid-lab
HUMANOID_DATA_ROOT=/tmp/humanoid-lab-data
HUMANOID_SOURCE_ROOT=/workspace/humanoid-lab
ROS_DOMAIN_ID=42
HOST_INPUT_GID=995
"""


def run_dev_sh(args: list[str]) -> str:
    """Run dev.sh with a stub docker and return its captured output."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bindir = tmp_path / "bin"
        bindir.mkdir()
        fake = bindir / "docker"
        fake.write_text(FAKE_DOCKER)
        fake.chmod(0o755)

        env_file = tmp_path / ".env"
        env_file.write_text(MINIMAL_ENV)

        # dev.sh reads ./.env next to itself; pass an override through a copy of
        # the script so the repo's working tree is never modified.
        script = tmp_path / "dev.sh"
        script.write_text(DEV_SH.read_text())
        script.chmod(0o755)

        env = dict(os.environ)
        env["PATH"] = f"{bindir}:{env['PATH']}"
        result = subprocess.run(  # noqa: S603
            ["bash", str(script), *args],
            capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120,
        )
    return result.stdout + result.stderr


class DevShDispatchTest(unittest.TestCase):
    def session_call(self, args: list[str]) -> str:
        out = run_dev_sh(args)
        calls = [line for line in out.splitlines() if "sonic-isaac-session.py" in line]
        self.assertTrue(calls, f"no session invocation found in:\n{out}")
        self.assertEqual(len(calls), 1, f"multiple invocations:\n{out}")
        return calls[0]

    def assert_subcommand_once(self, command: str, subcommand: str) -> None:
        occurrences = re.findall(rf"\b{re.escape(subcommand)}\b", command)
        self.assertEqual(
            len(occurrences), 1,
            f"subcommand {subcommand!r} appears {len(occurrences)}x in: {command}",
        )

    def test_accept_passes_its_arguments_exactly_once(self) -> None:
        command = self.session_call(["sonic-isaac", "accept", "--robot", "g1-29dof"])
        self.assert_subcommand_once(command, "accept")
        self.assertIn("--robot g1-29dof", command)

    def test_start_passes_its_arguments_exactly_once(self) -> None:
        command = self.session_call(
            ["sonic-isaac", "start", "--robot", "g1-29dof", "--input", "keyboard"]
        )
        self.assert_subcommand_once(command, "start")
        self.assertIn("--robot g1-29dof", command)
        self.assertIn("--input keyboard", command)

    def test_accept_for_the_inspire_profile_keeps_the_profile_argument(self) -> None:
        command = self.session_call(["sonic-isaac", "accept", "--robot", "g1-inspire"])
        self.assertIn("--robot g1-inspire", command)
        self.assert_subcommand_once(command, "accept")

    def test_status_uses_the_session_subcommand_once(self) -> None:
        self.assert_subcommand_once(self.session_call(["sonic-isaac", "status"]), "status")

    def test_stop_uses_the_session_subcommand_once(self) -> None:
        self.assert_subcommand_once(self.session_call(["sonic-isaac", "stop"]), "stop")

    def test_unknown_action_prints_usage_and_fails(self) -> None:
        out = run_dev_sh(["sonic-isaac", "bogus"])
        self.assertIn("usage:", out)


if __name__ == "__main__":
    unittest.main()
