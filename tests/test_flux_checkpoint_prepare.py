"""Focused tests for scripts/flux-prepare-checkpoint.py.

The script is the acceptance gate for the GPU server's checkpoint contract:
a read-only, symlink-free copy of the adapter and its checkpoint-owned
processor files, verified with the server's own checkpoint_identity -- plus the
reachability of the external base policy the adapter references.  These tests
exercise it against synthetic adapter directories (no real weights).
"""

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/flux-prepare-checkpoint.py"
REQUIRED = ("adapter_config.json", "adapter_model.safetensors",
            "policy_preprocessor.json", "policy_postprocessor.json")
PROCESSOR_WEIGHTS = ("policy_preprocessor_step_0_x.safetensors",
                     "policy_postprocessor_step_0_y.safetensors")


def make_base(root: Path, weight: bytes = b"base-weights") -> Path:
    """A mock base policy export: config + one weight file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "flux3"}), encoding="utf-8")
    (root / "model.safetensors").write_bytes(weight)
    return root


def make_checkpoint(root: Path, base_dir: Path | str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base_dir)}), encoding="utf-8")
    (root / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (root / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"state_file": PROCESSOR_WEIGHTS[0]}]}), encoding="utf-8")
    (root / "policy_postprocessor.json").write_text(
        json.dumps({"steps": [{"state_file": PROCESSOR_WEIGHTS[1]}]}), encoding="utf-8")
    for weight in PROCESSOR_WEIGHTS:
        (root / weight).write_bytes(b"processor-weights")
    # Training-only state that the server never hashes and must not copy.
    (root / "optimizer.pt").write_bytes(b"do not copy")
    (root / "ema_state.pt").write_bytes(b"do not copy")


def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True)


def identity(result):
    for line in result.stdout.splitlines():
        if line.startswith("checkpoint_identity="):
            return line.split("=", 1)[1]
    raise AssertionError("no checkpoint_identity in output: %s" % result.stdout)


def test_prepare_makes_an_immutable_minimal_copy(tmp_path):
    base = make_base(tmp_path / "base-policy")
    source = tmp_path / "checkpoint-2500"
    dest = tmp_path / "immutable" / "checkpoint-2500"
    make_checkpoint(source, base)

    result = run_script("--source", str(source), "--dest", str(dest))
    assert result.returncode == 0, result.stderr
    digest = identity(result)
    assert digest.startswith("sha256:") and len(digest) == 71

    assert sorted(path.name for path in dest.iterdir()) == sorted(REQUIRED + PROCESSOR_WEIGHTS)
    assert not (dest / "optimizer.pt").exists()
    assert not (dest / "ema_state.pt").exists()
    assert not dest.is_symlink()
    for path in dest.iterdir():
        assert not path.is_symlink()
        assert stat.S_IMODE(path.stat().st_mode) == 0o444, path
    assert stat.S_IMODE(dest.stat().st_mode) == 0o555

    # Provenance is written beside the served directory, never inside it, so
    # checkpoint_identity covers exactly the artifacts the server hashes.
    sidecar = dest.with_name(dest.name + ".provenance.json")
    assert sidecar.is_file()
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["checkpoint_identity"] == digest
    assert record["source_checkpoint"] == str(source.resolve())
    assert record["base_model"]["base_model_dir"] == str(base.resolve())
    assert record["lerobot_commit_pin"] == "e624f3f7f8411ec3a02635d06e79373341e5ef35"
    assert not (dest / sidecar.name).exists()

    # Idempotent: an existing copy is verified, not rewritten.
    again = run_script("--source", str(source), "--dest", str(dest))
    assert again.returncode == 0
    assert identity(again) == digest
    assert "already prepared" in again.stdout

    check = run_script("--check-only", "--dest", str(dest))
    assert check.returncode == 0
    assert identity(check) == digest
    assert "base_model_dir=%s" % base.resolve() in check.stdout


def test_prepare_reports_a_missing_base_policy_instead_of_baking_a_dead_path(tmp_path):
    # The trained adapter references the absolute path of the machine that
    # produced it; a fresh checkout must be told to redirect it, not fail at
    # model load time.
    source = tmp_path / "checkpoint-2500"
    make_checkpoint(source, "/home/nobody/flux-action/outputs/dex3/g1-base-policy")
    result = run_script("--source", str(source), "--dest", str(tmp_path / "copy"))
    assert result.returncode == 2
    assert "base policy directory missing" in result.stderr
    assert "--base-model-dir" in result.stderr
    assert not (tmp_path / "copy").exists()


def test_base_model_dir_rewrites_only_the_copied_adapter_config(tmp_path):
    old_base = make_base(tmp_path / "old-base")
    new_base = make_base(tmp_path / "new-base", weight=b"new-base-weights")
    source = tmp_path / "checkpoint-2500"
    dest = tmp_path / "copy"
    make_checkpoint(source, old_base)
    source_config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))

    written = run_script("--source", str(source), "--dest", str(dest),
                         "--base-model-dir", str(new_base))
    assert written.returncode == 0, written.stderr
    assert "base_model_dir=%s" % new_base.resolve() in written.stdout

    copied_config = json.loads((dest / "adapter_config.json").read_text(encoding="utf-8"))
    assert copied_config["base_model_name_or_path"] == str(new_base.resolve())
    # The source checkout is never modified, and the copy is still verified by
    # the server's own identity check (the rewrite happens before the hashing).
    assert json.loads((source / "adapter_config.json").read_text(encoding="utf-8")) == source_config

    check = run_script("--check-only", "--dest", str(dest))
    assert check.returncode == 0
    assert "base_model_dir=%s" % new_base.resolve() in check.stdout


def test_base_model_sha256_is_opt_in_and_verified(tmp_path):
    base = make_base(tmp_path / "base-policy")
    source = tmp_path / "checkpoint-2500"
    make_checkpoint(source, base)
    digest = hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest()

    good = run_script("--source", str(source), "--dest", str(tmp_path / "copy"),
                      "--base-model-sha256", digest)
    assert good.returncode == 0, good.stderr
    assert "base_model_weight_sha256=%s" % digest in good.stdout

    bad = run_script("--source", str(source), "--dest", str(tmp_path / "copy2"),
                     "--base-model-sha256", "0" * 64)
    assert bad.returncode == 2
    assert "hash mismatch" in bad.stderr
    assert not (tmp_path / "copy2").exists()


def test_check_only_catches_a_base_policy_that_moved_away(tmp_path):
    base = make_base(tmp_path / "base-policy")
    source = tmp_path / "checkpoint-2500"
    dest = tmp_path / "copy"
    make_checkpoint(source, base)
    assert run_script("--source", str(source), "--dest", str(dest)).returncode == 0

    # Simulate the export being moved/removed after preparation.
    for item in sorted(base.iterdir()):
        item.unlink()
    base.rmdir()

    check = run_script("--check-only", "--dest", str(dest))
    assert check.returncode == 2
    assert "base policy directory missing" in check.stderr

    # The re-entrant prepare path reports the same problem.
    again = run_script("--source", str(source), "--dest", str(dest))
    assert again.returncode == 2
    assert "base policy directory missing" in again.stderr


def test_prepare_force_replaces_and_identity_tracks_content(tmp_path):
    base = make_base(tmp_path / "base-policy")
    source = tmp_path / "checkpoint-2500"
    dest = tmp_path / "copy"
    make_checkpoint(source, base)
    first = run_script("--source", str(source), "--dest", str(dest))
    assert first.returncode == 0, first.stderr

    (source / "adapter_model.safetensors").write_bytes(b"different-weights")
    unchanged = run_script("--source", str(source), "--dest", str(dest))
    assert unchanged.returncode == 0
    assert identity(unchanged) == identity(first)  # no silent rewrite

    forced = run_script("--source", str(source), "--dest", str(dest), "--force")
    assert forced.returncode == 0, forced.stderr
    assert identity(forced) != identity(first)


def test_prepare_rejects_missing_or_writable_artifacts(tmp_path):
    base = make_base(tmp_path / "base-policy")
    source = tmp_path / "checkpoint-2500"
    make_checkpoint(source, base)
    (source / "policy_postprocessor.json").unlink()
    missing = run_script("--source", str(source), "--dest", str(tmp_path / "copy"))
    assert missing.returncode == 2
    assert "missing processor config" in missing.stderr

    # A symlinked adapter is refused before anything is copied.
    make_checkpoint(source, base)
    (source / "adapter_model.safetensors").unlink()
    os.symlink(source / "optimizer.pt", source / "adapter_model.safetensors")
    linked = run_script("--source", str(source), "--dest", str(tmp_path / "copy2"))
    assert linked.returncode == 2
    assert "symlink" in linked.stderr


def test_check_only_rejects_a_writable_copy(tmp_path):
    base = make_base(tmp_path / "base-policy")
    source = tmp_path / "checkpoint-2500"
    dest = tmp_path / "copy"
    make_checkpoint(source, base)
    assert run_script("--source", str(source), "--dest", str(dest)).returncode == 0

    os.chmod(dest, 0o755)
    os.chmod(dest / "adapter_model.safetensors", 0o644)
    check = run_script("--check-only", "--dest", str(dest))
    assert check.returncode == 1
    assert "checkpoint identity rejected" in check.stderr
