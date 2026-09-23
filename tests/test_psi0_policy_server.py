"""The owned policy-server process and the base-artifact materializer.

Both are exercised with tiny local artifacts: a python child stands in for
``serve_psi0_sonic`` (so a restart is really a process restart) and the base
merge runs on a handful of tensors instead of the 11 GB warm start.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from humanoid_lab.psi0_bridge import base_artifact
from humanoid_lab.psi0_bridge.base_artifact import BaseArtifactError
from humanoid_lab.psi0_bridge.policy_server import (
    PolicyServerError,
    PolicyServerProcess,
    serve_command,
)

try:
    import torch  # noqa: F401
    from safetensors.torch import load_file, save_file
except ImportError:  # pragma: no cover - the psi0 environment ships torch
    load_file = save_file = None

WAIT = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"


class _FakeServe(PolicyServerProcess):
    """Spawns a python child instead of the real deployment binary."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spawned: list[tuple[str, int]] = []

    def command(self, run_dir: Path, step: int) -> list[str]:
        self.spawned.append((str(run_dir), step))
        return [sys.executable, "-c", WAIT]


class PolicyServerProcessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run"
        (self.run_dir / "checkpoints" / "ckpt_0").mkdir(parents=True)
        self._probe = mock.patch("humanoid_lab.psi0_bridge.policy_server.probe_info",
                                 return_value=None)
        self._probe.start()
        self.addCleanup(self._probe.stop)

    def test_the_canonical_serve_command_is_the_deployment_invocation(self) -> None:
        command = serve_command(Path("/outputs/run"), 40000, 8014)
        self.assertEqual(
            command,
            ["/workspace/humanoid-lab/scripts/serve-psi0-sonic.py",
             "--host", "0.0.0.0", "--port", "8014",
             "--action_exec_horizon", "30", "--policy", "psi", "--rtc",
             "--run-dir", "/outputs/run", "--ckpt-step", "40000"],
        )
        self.assertNotIn("--device", command)
        self.assertEqual(
            serve_command(Path("/outputs/run"), 40000, 8014, rtc=False,
                          action_exec_horizon=12),
            ["/workspace/humanoid-lab/scripts/serve-psi0-sonic.py",
             "--host", "0.0.0.0", "--port", "8014",
             "--action_exec_horizon", "12", "--policy", "psi",
             "--run-dir", "/outputs/run", "--ckpt-step", "40000"],
        )

    def test_cpu_serving_cannot_be_requested_at_all(self) -> None:
        # The deployment's inference path pins CUDA autocast: a CPU device loads
        # the model and answers /info, then fails on the first forward pass, so
        # the bridge has no device knob -- the canonical cuda:0 command is the
        # only servable path and a CPU request is a TypeError before any spawn.
        with self.assertRaises(TypeError):
            serve_command(Path("/outputs/run"), 40000, 8014, device="cpu")
        with self.assertRaises(TypeError):
            PolicyServerProcess(port=8014, device="cpu")
        self.assertEqual(
            PolicyServerProcess(port=8014).command(Path("/outputs/run"), 0),
            serve_command(Path("/outputs/run"), 0, 8014),
        )

    def test_simulation_clock_configuration_reaches_only_the_owned_child_environment(self) -> None:
        clock_file = Path(self.tmp.name) / "isaac.clock"
        self.assertFalse(clock_file.exists())
        server = PolicyServerProcess(
            port=8014,
            policy_clock="simulation",
            policy_clock_file=clock_file,
            policy_clock_timeout_s=3.5,
        )
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch("humanoid_lab.psi0_bridge.policy_server.subprocess.Popen", return_value=process) as popen:
            server.start(self.run_dir, 0)
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["HUMANOID_POLICY_CLOCK"], "simulation")
        self.assertEqual(environment["HUMANOID_POLICY_CLOCK_FILE"], str(clock_file))
        self.assertEqual(environment["HUMANOID_POLICY_CLOCK_TIMEOUT_S"], "3.5")

    def test_simulation_server_refuses_a_missing_clock_file_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a clock file"):
            PolicyServerProcess(port=8014, policy_clock="simulation")

    def test_start_stop_restart_touches_only_the_child(self) -> None:
        server = _FakeServe(port=8014, log_path=Path(self.tmp.name) / "server.log")
        parent = os.getpid()
        server.start(self.run_dir, 0)
        first_pid = server.pid
        self.assertTrue(server.alive)
        self.assertIsNotNone(first_pid)
        self.assertNotEqual(first_pid, parent)

        server.stop()
        self.assertFalse(server.alive)
        self.assertFalse(Path(f"/proc/{first_pid}").exists())
        self.assertEqual(os.getpid(), parent)  # the caller is untouched

        server.start(self.run_dir, 0)
        self.assertNotEqual(server.pid, first_pid)
        self.assertTrue(server.alive)
        server.stop()
        self.assertFalse(server.alive)

    def test_stop_escalates_when_the_child_ignores_term(self) -> None:
        server = _FakeServe(port=8014, stop_timeout_s=0.3, kill_wait_s=5.0)
        server.start(self.run_dir, 0)
        pid = server.pid
        start = time.monotonic()
        server.stop()
        self.assertLess(time.monotonic() - start, 10.0)
        self.assertFalse(Path(f"/proc/{pid}").exists())

    def test_start_refuses_missing_checkpoint_dir_or_foreign_server(self) -> None:
        server = _FakeServe(port=8014)
        with self.assertRaises(PolicyServerError) as ctx:
            server.start(Path(self.tmp.name) / "missing", 0)
        self.assertIn("ckpt_0", str(ctx.exception))

        with mock.patch("humanoid_lab.psi0_bridge.policy_server.probe_info",
                        return_value={"run_dir": "/outputs/other"}):
            with self.assertRaises(PolicyServerError) as ctx:
                server.start(self.run_dir, 0)
        self.assertIn("already serves", str(ctx.exception))
        self.assertFalse(server.alive)

    def test_a_missing_executable_is_reported(self) -> None:
        server = PolicyServerProcess(port=8014)
        server.command = lambda run_dir, step: ["definitely-not-serve_psi0_sonic"]
        with self.assertRaises(PolicyServerError) as ctx:
            server.start(self.run_dir, 0)
        self.assertIn("cannot start", str(ctx.exception))
        self.assertFalse(server.alive)


@unittest.skipIf(save_file is None, "torch/safetensors are not installed in this environment")
class BaseMaterializationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.fine_dir = root / "finetune" / "run.2609171455"
        (self.fine_dir / "checkpoints" / "ckpt_40000").mkdir(parents=True)
        self.source_dir = root / "checkpoints" / "psi0" / "postpre.sonic1.0.40k"
        self.source_dir.mkdir(parents=True)
        self.out_dir = root / "base" / self.source_dir.name

        save_file({"model.embed_tokens.weight": torch.zeros(2, 3),
                   "model.layers.0.mlp.weight": torch.ones(3)},
                  str(self.source_dir / "model.safetensors"))
        save_file({"transformer_blocks.0.proj.weight": torch.full((2,), 7.0),
                   "state_pos": torch.zeros(1, 1, 2)},
                  str(self.source_dir / "action_header.safetensors"))
        (self.source_dir / "MODEL_PROVENANCE.json").write_text(json.dumps({
            "repo": "USC-PSI-Lab/psi-model", "revision": "abc123",
            "variant": "psi0/postpre.sonic1.0.40k",
            "files": [{"path": "model.safetensors",
                       "sha256": self.sha256(self.source_dir / "model.safetensors")},
                      {"path": "action_header.safetensors",
                       "sha256": self.sha256(self.source_dir / "action_header.safetensors")}],
        }), encoding="utf-8")

        (self.fine_dir / "run_config.json").write_text(json.dumps({
            "model": {"model_name_or_path": str(self.source_dir),
                      "state_null_token": True, "action_dim": 80},
        }), encoding="utf-8")
        (self.fine_dir / "argv.txt").write_text(
            "finetune-real-psi0\n--model.action-dim=80\n--model.state-null-token\n"
            "--train.learning_rate=2.5e-5\n", encoding="utf-8")
        (self.fine_dir / "clip_pooled_cache.pt").write_bytes(b"cache")

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @property
    def merged(self) -> Path:
        return self.out_dir / "checkpoints" / "ckpt_0" / "model.safetensors"

    def materialize(self, **kwargs) -> dict:
        return base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir, **kwargs)

    def test_materialize_merges_keys_and_patches_the_config(self) -> None:
        record = base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        self.assertEqual(record["model_name_or_path"], str(self.source_dir))
        self.assertEqual(record["source_provenance"]["repo"], "USC-PSI-Lab/psi-model")

        argv = (self.out_dir / "argv.txt").read_text(encoding="utf-8")
        self.assertNotIn("state-null-token", argv)
        self.assertIn("--model.action-dim=80", argv)
        self.assertIn("--train.learning_rate=2.5e-5", argv)
        config = json.loads((self.out_dir / "run_config.json").read_text(encoding="utf-8"))
        self.assertFalse(config["model"]["state_null_token"])

        merged_path = self.out_dir / "checkpoints" / "ckpt_0" / "model.safetensors"
        merged = load_file(str(merged_path))
        self.assertEqual(sorted(merged), [
            "action_header.state_pos",
            "action_header.transformer_blocks.0.proj.weight",
            "vlm_model.model.embed_tokens.weight",
            "vlm_model.model.layers.0.mlp.weight",
        ])
        self.assertTrue(torch.equal(merged["vlm_model.model.layers.0.mlp.weight"], torch.ones(3)))
        self.assertTrue(torch.equal(
            merged["action_header.transformer_blocks.0.proj.weight"], torch.full((2,), 7.0)))
        self.assertEqual(record["merged"]["vlm_model_keys"], 2)
        self.assertEqual(record["merged"]["action_header_keys"], 2)
        self.assertEqual(record["source_files"]["model.safetensors"]["sha256"],
                         self.sha256(self.source_dir / "model.safetensors"))
        self.assertEqual(record["merged"]["sha256"], self.sha256(self.merged))
        self.assertEqual((self.out_dir / "clip_pooled_cache.pt").read_bytes(), b"cache")
        self.assertTrue((self.out_dir / "BASE_ARTIFACT.json").is_file())

    def test_materialization_is_idempotent(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        inode = self.merged.stat().st_ino
        record = self.materialize()
        self.assertEqual(self.merged.stat().st_ino, inode)  # verified, not rewritten
        self.assertEqual(record["merged"]["vlm_model_keys"], 2)

    def test_reuse_is_refused_when_a_source_file_changes(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        inode = self.merged.stat().st_ino
        # Same keys, same size, different weights: only the recorded sha256 can
        # tell the stale merge from a fresh one.
        save_file({"model.embed_tokens.weight": torch.full((2, 3), 5.0),
                   "model.layers.0.mlp.weight": torch.ones(3)},
                  str(self.source_dir / "model.safetensors"))
        record = self.materialize()
        self.assertNotEqual(self.merged.stat().st_ino, inode)
        self.assertEqual(record["source_files"]["model.safetensors"]["sha256"],
                         self.sha256(self.source_dir / "model.safetensors"))
        self.assertTrue(torch.equal(
            load_file(str(self.merged))["vlm_model.model.embed_tokens.weight"],
            torch.full((2, 3), 5.0)))

    def test_a_truncated_merge_is_rebuilt(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        good = self.merged.read_bytes()
        self.merged.write_bytes(good[:-4])
        self.materialize()
        self.assertEqual(self.merged.read_bytes(), good)

    def test_a_corrupted_merge_is_rebuilt(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        good = self.merged.read_bytes()
        flipped = bytearray(good)
        flipped[-1] ^= 0xFF  # a payload byte: only the recorded sha256 sees this
        self.merged.write_bytes(bytes(flipped))
        self.materialize()
        self.assertEqual(self.merged.read_bytes(), good)

    def test_merged_keys_outside_the_source_layout_are_rebuilt(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        tensors = load_file(str(self.merged))
        tensors["vlm_model.model.extra.weight"] = torch.zeros(1)
        save_file(tensors, str(self.merged))
        self.materialize()
        self.assertNotIn("vlm_model.model.extra.weight", load_file(str(self.merged)))

    def test_run_config_drift_is_repaired(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        config_path = self.out_dir / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model"]["state_null_token"] = True
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.materialize()
        served = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertIs(served["model"]["state_null_token"], False)

    def test_argv_drift_is_repaired(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        argv_path = self.out_dir / "argv.txt"
        argv_path.write_text(argv_path.read_text(encoding="utf-8") + "--model.state-null-token\n",
                             encoding="utf-8")
        self.materialize()
        self.assertNotIn("state-null-token", argv_path.read_text(encoding="utf-8"))

    def test_a_removed_checkpoint_or_cache_is_restored(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        self.merged.unlink()
        (self.out_dir / "clip_pooled_cache.pt").unlink()
        self.materialize()
        self.assertTrue(self.merged.is_file())
        self.assertEqual((self.out_dir / "clip_pooled_cache.pt").read_bytes(), b"cache")

    def test_a_record_from_another_fine_run_is_not_reused(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        provenance = self.out_dir / "BASE_ARTIFACT.json"
        record = json.loads(provenance.read_text(encoding="utf-8"))
        record["fine_run_dir"] = str(self.fine_dir.parent / "other-run")
        provenance.write_text(json.dumps(record), encoding="utf-8")
        inode = self.merged.stat().st_ino
        self.materialize()
        self.assertNotEqual(self.merged.stat().st_ino, inode)

    def test_an_older_record_without_fingerprints_is_adopted_not_rebuilt(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        inode = self.merged.stat().st_ino
        provenance = self.out_dir / "BASE_ARTIFACT.json"
        record = json.loads(provenance.read_text(encoding="utf-8"))
        record["merged"].pop("sha256")
        record["source_files"]["model.safetensors"].pop("sha256")
        provenance.write_text(json.dumps(record), encoding="utf-8")

        reused = self.materialize()
        self.assertEqual(self.merged.stat().st_ino, inode)  # sound artifact kept
        self.assertEqual(reused["merged"]["sha256"], self.sha256(self.merged))
        self.assertEqual(reused["source_files"]["model.safetensors"]["sha256"],
                         self.sha256(self.source_dir / "model.safetensors"))

    def test_a_failed_rebuild_keeps_the_previous_dir_and_leaves_no_temp(self) -> None:
        base_artifact.materialize_base_run_dir(self.fine_dir, self.out_dir)
        damaged = self.merged.read_bytes()[:-4]
        self.merged.write_bytes(damaged)  # force a rebuild decision
        with mock.patch.object(base_artifact, "_merge_state_dict",
                               side_effect=BaseArtifactError("merge failed")):
            with self.assertRaises(BaseArtifactError):
                self.materialize()
        self.assertEqual(self.merged.read_bytes(), damaged)  # nothing half-written
        leftovers = sorted(p.name for p in self.out_dir.parent.iterdir()
                           if p.name.startswith(f".{self.out_dir.name}."))
        self.assertEqual(leftovers, [])

        self.materialize()  # the next attempt recovers
        self.assertNotEqual(self.merged.read_bytes(), damaged)

    def test_a_stale_temp_dir_from_a_crashed_build_is_removed(self) -> None:
        stale = self.out_dir.parent / f".{self.out_dir.name}.tmp-999999"
        (stale / "checkpoints").mkdir(parents=True)
        (stale / "run_config.json").write_text("{}", encoding="utf-8")
        self.materialize()
        self.assertFalse(stale.exists())
        self.assertTrue(self.merged.is_file())

    def test_default_output_sits_next_to_the_fine_tune_runs(self) -> None:
        source = base_artifact.read_base_source(self.fine_dir)
        self.assertEqual(base_artifact.default_base_run_dir(self.fine_dir, source), self.out_dir)

    def test_missing_or_ambiguous_lineage_is_refused(self) -> None:
        config_path = self.fine_dir / "run_config.json"

        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model"].pop("model_name_or_path")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(BaseArtifactError) as ctx:
            base_artifact.read_base_source(self.fine_dir)
        self.assertIn("model_name_or_path", str(ctx.exception))

        config["model"]["model_name_or_path"] = str(self.source_dir)
        config["model"]["pretrained_action_header_path"] = "/somewhere/else"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(BaseArtifactError) as ctx:
            base_artifact.read_base_source(self.fine_dir)
        self.assertIn("ambiguous", str(ctx.exception))

        config["model"]["model_name_or_path"] = str(self.source_dir / "missing")
        config["model"].pop("pretrained_action_header_path")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(BaseArtifactError) as ctx:
            base_artifact.read_base_source(self.fine_dir)
        self.assertIn("does not exist", str(ctx.exception))

        config["model"]["model_name_or_path"] = str(self.source_dir)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        (self.source_dir / "action_header.safetensors").unlink()
        with self.assertRaises(BaseArtifactError) as ctx:
            base_artifact.read_base_source(self.fine_dir)
        self.assertIn("action_header.safetensors", str(ctx.exception))

    def test_refuses_to_materialize_over_an_input(self) -> None:
        with self.assertRaises(BaseArtifactError):
            base_artifact.materialize_base_run_dir(self.fine_dir, self.fine_dir)
        with self.assertRaises(BaseArtifactError):
            base_artifact.materialize_base_run_dir(self.fine_dir, self.source_dir)


if __name__ == "__main__":
    unittest.main()
