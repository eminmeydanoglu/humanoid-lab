"""scripts/psi0-isaac-eval.py and its dev.sh entrypoint: the fail-fast gates.

These are the checks that keep a wrong run (missing checkpoint, non-integer
step, occupied action port) from starting Isaac, the SONIC controller and a 3B
policy server only to fail minutes later.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "psi0-isaac-eval.py"
DEV_SH = ROOT / "dev.sh"
PROFILE = ROOT / "configs" / "profiles" / "isaac-g1-sonic-blockstacking-dex3.json"

try:
    import zmq
except ImportError:  # pragma: no cover - the psi0 environment ships pyzmq
    zmq = None


def load_launcher():
    spec = importlib.util.spec_from_file_location("psi0_isaac_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_config(*, action_dim: int = 80, resize=(240, 320), pad_state_dim: int = 45,
               num_past_frames: int | None = 0) -> dict:
    repack: dict = {"dataset_name": "sonic"}
    if num_past_frames is not None:
        repack["num_past_frames"] = num_past_frames
    return {
        "model": {"action_dim": action_dim, "action_chunk_size": 30, "action_exec_horizon": 30},
        "data": {
            "transform": {
                "model": {
                    "resize": {"size": list(resize)},
                    "center_crop": {"size": list(resize)},
                },
                "field": {"pad_state_dim": pad_state_dim, "normalize_state": True},
                "repack": repack,
            }
        },
    }


class RunDirectoryValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.launcher = load_launcher()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "postpre.sonic.20260918-1200"
        (self.run_dir / "checkpoints" / "ckpt_40000").mkdir(parents=True)
        (self.run_dir / "argv.txt").write_text("finetune-real-psi0\n", encoding="utf-8")
        self.write_config(run_config())

    def write_config(self, config: dict) -> None:
        (self.run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    def test_a_well_formed_run_is_accepted(self) -> None:
        summary = self.launcher.validate_run_dir(self.run_dir, 40000)
        self.assertEqual(summary["action_dim"], 80)
        self.assertEqual(summary["ckpt_step"], 40000)
        self.assertEqual(summary["resize"], [240, 320])
        self.assertEqual(summary["state_dim"], 45)
        self.assertEqual(summary["num_past_frames"], 0)

    def test_state_history_is_derived_from_the_run_config(self) -> None:
        # The bridge sends one state frame, so a run packed with a past-frame
        # window must be refused instead of being silently mis-served.
        self.write_config(run_config(num_past_frames=1))
        with self.assertRaises(self.launcher.LaunchError) as ctx:
            self.launcher.validate_run_dir(self.run_dir, 40000)
        self.assertIn("num_past_frames", str(ctx.exception))

        # No key -> no evidence: reported as unknown rather than guessed.
        self.write_config(run_config(num_past_frames=None))
        self.assertIsNone(self.launcher.validate_run_dir(self.run_dir, 40000)["num_past_frames"])

    def test_rejections(self) -> None:
        cases = {
            "missing directory": (Path(self.tmp.name) / "nope", 40000, {}),
            "missing step": (self.run_dir, 12345, {}),
            "zero step": (self.run_dir, 0, {}),
            "wrong action width": (self.run_dir, 40000, {"action_dim": 77}),
            "wrong resize": (self.run_dir, 40000, {"resize": (128, 128)}),
            "wrong state pad": (self.run_dir, 40000, {"pad_state_dim": 64}),
        }
        for name, (run_dir, step, overrides) in cases.items():
            with self.subTest(case=name):
                if overrides:
                    self.write_config(run_config(**overrides))
                try:
                    with self.assertRaises(self.launcher.LaunchError):
                        self.launcher.validate_run_dir(run_dir, step)
                finally:
                    self.write_config(run_config())

    def test_missing_argv_and_config_are_rejected(self) -> None:
        (self.run_dir / "argv.txt").unlink()
        with self.assertRaises(self.launcher.LaunchError):
            self.launcher.validate_run_dir(self.run_dir, 40000)
        (self.run_dir / "run_config.json").unlink()
        with self.assertRaises(self.launcher.LaunchError):
            self.launcher.validate_run_dir(self.run_dir, 40000)

    def test_center_crop_must_be_the_pinned_size(self) -> None:
        config = run_config()
        config["data"]["transform"]["model"]["center_crop"] = {"size": [128, 128]}
        self.write_config(config)
        with self.assertRaises(self.launcher.LaunchError) as ctx:
            self.launcher.validate_run_dir(self.run_dir, 40000)
        self.assertIn("center_crop", str(ctx.exception))

    def test_the_base_step_zero_needs_the_explicit_minimum(self) -> None:
        (self.run_dir / "checkpoints" / "ckpt_0").mkdir()
        with self.assertRaises(self.launcher.LaunchError):
            self.launcher.validate_run_dir(self.run_dir, 0)
        summary = self.launcher.validate_run_dir(self.run_dir, 0, minimum_step=0)
        self.assertEqual(summary["ckpt_step"], 0)

    def test_checkpoint_step_must_be_an_integer(self) -> None:
        with self.assertRaises(SystemExit):
            self.launcher.parse_args(["--checkpoint-dir", str(self.run_dir), "--checkpoint-step", "latest"])

    def test_missing_run_directory_exits_two_with_a_reason(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--checkpoint-dir", str(self.run_dir / "missing"),
             "--checkpoint-step", "40000", "--check-only"],
            capture_output=True, text=True, cwd=ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("preflight failed", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)


@unittest.skipIf(zmq is None, "pyzmq is not installed in this environment")
class ActionSocketOwnershipTest(unittest.TestCase):
    """The session owns the action socket from construction until close()."""

    def test_a_busy_action_port_fails_session_construction(self) -> None:
        from humanoid_lab.psi0_bridge.session import Session, SessionConfig, SessionError

        context = zmq.Context()
        other = context.socket(zmq.PUB)
        other.setsockopt(zmq.LINGER, 0)
        other.bind("tcp://127.0.0.1:0")
        endpoint = other.getsockopt_string(zmq.LAST_ENDPOINT)
        try:
            with self.assertRaises(SessionError) as ctx:
                Session(SessionConfig(action_endpoint=endpoint))
            self.assertIn("action socket", str(ctx.exception))
        finally:
            other.close(linger=0)
            context.term()

        # Once the other publisher is gone the session can take ownership, and
        # only close() releases the socket again (idempotent).
        session = Session(SessionConfig(action_endpoint=endpoint))
        session.close()
        session.close()
        probe_context = zmq.Context()
        probe = probe_context.socket(zmq.PUB)
        probe.setsockopt(zmq.LINGER, 0)
        try:
            probe.bind(endpoint.replace("127.0.0.1", "*"))
        finally:
            probe.close(linger=0)
            probe_context.term()


@unittest.skipIf(zmq is None, "pyzmq is not installed in this environment")
class PreflightComparisonTest(unittest.TestCase):
    """preflight() ties the run directory to what the policy server actually serves."""

    def setUp(self) -> None:
        self.launcher = load_launcher()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run"
        (self.run_dir / "checkpoints" / "ckpt_40000").mkdir(parents=True)
        (self.run_dir / "argv.txt").write_text("finetune-real-psi0\n", encoding="utf-8")
        (self.run_dir / "run_config.json").write_text(json.dumps(run_config()), encoding="utf-8")
        self.args = self.launcher.parse_args([
            "--checkpoint-dir", str(self.run_dir), "--checkpoint-step", "40000",
            "--check-only", "--action-endpoint", "tcp://*:0",
        ])

    def _info(self, action_dim: int, run_dir: str | None = None):
        from humanoid_lab.psi0_bridge.contracts import validate_info

        return validate_info({
            "run_dir": run_dir or str(self.run_dir.resolve()),
            "ckpt_step": 40000,
            "dataset_name": "sonic",
            "transforms": [
                {"name": "resize", "size": [240, 320]},
                {"name": "center_crop", "size": [240, 320]},
            ],
            "expected_keys": {
                "image": {"observation.images.egocentric": "HxWx3 uint8 image array"},
                "state": {"states": "1x45 unnormalized state vector"},
            },
            "observation": {"state_dim": 45, "normalize_state": True},
            "action": {"action_dim": action_dim, "action_chunk_size": 30, "action_exec_horizon": 30},
            "rtc_enabled": True,
        })

    def _preflight(self, info):
        with mock.patch.object(self.launcher, "fetch_info", return_value=info):
            return self.launcher.preflight(self.args)

    def test_a_server_serving_a_different_action_width_is_rejected(self) -> None:
        with self.assertRaises(self.launcher.LaunchError) as ctx:
            self._preflight(self._info(78))
        self.assertIn("does not match the served", str(ctx.exception))

    def test_a_server_serving_a_different_run_dir_is_rejected(self) -> None:
        # Same step and same action width, different run directory: the identity
        # check must still refuse it (no basename-style comparison).
        other = Path(self.tmp.name) / "other-run"
        with self.assertRaises(self.launcher.LaunchError) as ctx:
            self._preflight(self._info(80, run_dir=str(other)))
        self.assertIn("not the selected run", str(ctx.exception))

    def test_a_matching_run_and_width_passes(self) -> None:
        summary = self._preflight(self._info(80))
        self.assertEqual(summary["run_dir"], str(self.run_dir.resolve()))
        self.assertEqual(summary["action_dim"], 80)


class CheckpointEntriesTest(unittest.TestCase):
    """The allowlist the launcher hands to the UI: fine-tuned + real base."""

    def setUp(self) -> None:
        self.launcher = load_launcher()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fine_dir = self.root / "finetune" / "run.2609171455"
        (self.fine_dir / "checkpoints" / "ckpt_40000").mkdir(parents=True)
        (self.fine_dir / "argv.txt").write_text("finetune-real-psi0\n", encoding="utf-8")
        config = run_config()
        config["model"]["model_name_or_path"] = str(self.root / "checkpoints" / "postpre.40k")
        (self.fine_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    def args(self, *extra: str):
        return self.launcher.parse_args([
            "--checkpoint-dir", str(self.fine_dir), "--checkpoint-step", "40000", *extra,
        ])

    def test_labels_are_operator_readable(self) -> None:
        self.assertEqual(self.launcher.finetuned_label(40000), "Fine-tuned (40k)")
        self.assertEqual(self.launcher.finetuned_label(5500), "Fine-tuned (step 5500)")

    def test_an_unbuildable_base_is_unavailable_not_substituted(self) -> None:
        entry = self.launcher.base_entry(self.args())
        self.assertFalse(entry.available)
        self.assertEqual(entry.label, "Base")
        self.assertEqual(entry.step, 0)
        self.assertIsNone(entry.run_dir)
        self.assertIn("does not exist", entry.reason)

    def test_an_explicit_base_run_dir_is_validated(self) -> None:
        base_dir = self.root / "base" / "postpre.40k"
        (base_dir / "checkpoints" / "ckpt_0").mkdir(parents=True)
        (base_dir / "argv.txt").write_text("finetune-real-psi0\n", encoding="utf-8")
        (base_dir / "run_config.json").write_text(json.dumps(run_config()), encoding="utf-8")
        entry = self.launcher.base_entry(self.args("--base-run-dir", str(base_dir)))
        self.assertTrue(entry.available)
        self.assertEqual(entry.run_dir, base_dir.resolve())
        self.assertEqual(entry.step, 0)
        self.assertEqual(entry.detail["action_dim"], 80)

    def test_the_derived_base_is_materialized_next_to_the_finetune(self) -> None:
        source = self.root / "checkpoints" / "postpre.40k"
        source.mkdir(parents=True)
        (source / "model.safetensors").write_bytes(b"")
        (source / "action_header.safetensors").write_bytes(b"")
        expected = self.root / "base" / "postpre.40k"

        def fake_materialize(fine_run_dir, out_dir, *, source, step, log):
            (out_dir / "checkpoints" / f"ckpt_{step}").mkdir(parents=True)
            (out_dir / "argv.txt").write_text("finetune-real-psi0\n", encoding="utf-8")
            (out_dir / "run_config.json").write_text(json.dumps(run_config()), encoding="utf-8")
            (out_dir / "checkpoints" / f"ckpt_{step}" / "model.safetensors").write_bytes(b"")
            return {"merged": {"path": str(out_dir / "checkpoints" / f"ckpt_{step}" / "model.safetensors")}}

        with mock.patch.object(self.launcher.base_artifact, "materialize_base_run_dir",
                               side_effect=fake_materialize) as materialize:
            entry = self.launcher.base_entry(self.args())
        self.assertTrue(entry.available)
        self.assertEqual(entry.run_dir, expected.resolve())
        self.assertEqual(materialize.call_count, 1)
        self.assertEqual(materialize.call_args.args[1], expected)

    def test_a_broken_base_run_dir_keeps_the_fine_tuned_path(self) -> None:
        base_dir = self.root / "base" / "postpre.40k"
        (base_dir / "checkpoints" / "ckpt_0").mkdir(parents=True)
        (base_dir / "argv.txt").write_text("x\n", encoding="utf-8")
        (base_dir / "run_config.json").write_text(json.dumps({"model": {}}), encoding="utf-8")
        entry = self.launcher.base_entry(self.args("--base-run-dir", str(base_dir)))
        self.assertFalse(entry.available)
        self.assertIn("action_dim", entry.reason)
        fine = self.launcher.finetuned_entry(self.args())
        self.assertTrue(fine.available)
        self.assertEqual(fine.step, 40000)


class PolicyDeviceRemovalTest(unittest.TestCase):
    """CPU serving is gone: a device request fails before anything starts."""

    def test_the_policy_device_flag_is_rejected_before_any_server_spawns(self) -> None:
        launcher = load_launcher()
        with mock.patch.object(launcher.PolicyServerProcess, "start") as start, \
             mock.patch.object(launcher, "Session") as session:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    launcher.main(["--checkpoint-dir", "/tmp/run", "--checkpoint-step", "40000",
                                   "--policy-device", "cpu"])
        self.assertEqual(ctx.exception.code, 2)
        start.assert_not_called()
        session.assert_not_called()

    def test_the_environment_cannot_select_a_device_either(self) -> None:
        # The old PSI0_POLICY_DEVICE default is gone; nothing reads it, so a
        # leftover export cannot smuggle a CPU device past the CLI.
        launcher = load_launcher()
        base = ["--checkpoint-dir", "/tmp/run", "--checkpoint-step", "40000"]
        with mock.patch.dict(os.environ, {"PSI0_POLICY_DEVICE": "cpu"}):
            args = launcher.parse_args(base)
        self.assertFalse(hasattr(args, "policy_device"))

    def test_the_usage_does_not_advertise_a_device_option(self) -> None:
        launcher = load_launcher()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit):
                launcher.parse_args(["--help"])
        self.assertNotIn("--policy-device", out.getvalue())

    def test_devsh_forwards_bridge_flags(self) -> None:
        # recv-timeout/port style flags travel to the bridge through the
        # launcher's own argument passthrough.
        self.assertIn("$eval_args_str", DEV_SH.read_text(encoding="utf-8"))


class DevShEntrypointTest(unittest.TestCase):
    def test_devsh_is_valid_and_exposes_the_single_command(self) -> None:
        proc = subprocess.run(["bash", "-n", str(DEV_SH)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        text = DEV_SH.read_text(encoding="utf-8")
        self.assertIn("psi0-isaac-eval)", text)
        self.assertIn("psi0_isaac_eval", text)
        self.assertIn("scripts/psi0-isaac-eval.py", text)
        self.assertIn("configs/profiles/isaac-g1-sonic-blockstacking-dex3.json", text)
        # The eval must consume the policy's pose stream: the keyboard-only input
        # interface has no ZMQ subscriber, so the launcher starts the controller
        # in the upstream pose-streaming mode.
        self.assertIn("run_sonic_controller zmq", text)
        self.assertIn("press Enter to enable the", text)

    def test_the_bridge_owns_the_policy_server(self) -> None:
        text = DEV_SH.read_text(encoding="utf-8")
        # The launcher must not start serve_psi0_sonic as its own job: the bridge
        # spawns and restarts it, and the launcher's watchdog only owns Isaac and
        # the bridge (a UI switch must not abort the session).
        self.assertNotIn("psi0_eval_bg_start psi0-server", text)
        self.assertNotIn("exec serve_psi0_sonic", text)
        self.assertIn("--policy-log='$PSI0_EVAL_LOG_DIR/psi0-server.log'", text)

    def test_the_controller_starts_only_after_isaac_steps_steadily(self) -> None:
        """The deployment exits when LowState is stale for more than 500 ms, and
        the physics loop stalls during the RTX warm-up; handing it the terminal
        before the loop has reported steadily tears the session down."""
        text = DEV_SH.read_text(encoding="utf-8")
        self.assertIn("psi0_eval_wait_isaac_steady", text)
        self.assertIn("^\\[isaac-g1\\] physics=", text)
        self.assertIn("Isaac physics loop steady", text)
        wait_call = text.index('psi0_eval_wait_isaac_steady "$isaac_index"')
        self.assertLess(wait_call, text.index("run_sonic_controller zmq"))
        self.assertIn("did not reach a steady physics loop", text)


if __name__ == "__main__":
    unittest.main()
