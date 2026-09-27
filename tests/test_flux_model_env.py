"""Focused checks for the GPU-side model environment bootstrap/preflight.

`--plan` resolves paths, pins and commands without touching docker, the GPU or
the network, so the resolution order and the pins are testable here.  Building
the environment itself needs network and ~7 GiB of wheels and is documented in
scripts/flux-README.md; these tests never run it.
"""

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO / "scripts/flux-bootstrap-model-env.sh"
SERVER = REPO / "scripts/flux-model-server.sh"
LOCK = REPO / "scripts/flux-model-env.lock.txt"
LEROBOT_COMMIT = "e624f3f7f8411ec3a02635d06e79373341e5ef35"


def clean_env(**overrides):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("FLUX_", "HUMANOID_DATA_ROOT"))}
    environment.update(overrides)
    return environment


def run(script, *args, env=None):
    return subprocess.run(["bash", str(script), *args],
                          capture_output=True, text=True, cwd=REPO, env=env)


def test_lock_file_pins_the_validated_environment_without_ephemeral_paths():
    body = [line for line in LOCK.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]
    assert "torch==2.10.0+cu128" in body
    assert "torchvision==0.25.0+cu128" in body
    assert "natten==0.21.6+torch2100cu128" in body
    assert "peft==0.21.0" in body
    assert "transformers==5.5.4" in body
    assert not [line for line in body if line.startswith("-e ")]
    assert not [line for line in body if "lerobot" in line.lower()]
    assert not [line for line in body if "/tmp/" in line or "/home/" in line]
    header = LOCK.read_text(encoding="utf-8")
    assert LEROBOT_COMMIT in header
    assert "download.pytorch.org/whl/cu128" in header


def test_bootstrap_plan_reports_pins_and_external_artifacts(tmp_path):
    plan = json.loads(run(BOOTSTRAP, "--plan",
                          env=clean_env(HUMANOID_DATA_ROOT=str(tmp_path))).stdout)
    assert plan["lerobot_repo"] == "https://github.com/huggingface/lerobot.git"
    assert plan["lerobot_commit"] == LEROBOT_COMMIT
    assert plan["python_version"] == "3.12"
    assert plan["lock_exists"] is True
    assert plan["venv"] == str(tmp_path / "venvs/flux-model")
    assert plan["venv_exists"] is False  # nothing was created by --plan
    assert not (tmp_path / "venvs").exists()
    joined = " ".join(plan["install_commands"])
    assert LEROBOT_COMMIT in joined
    assert "download.pytorch.org/whl/cu128" in joined
    assert "whl.natten.org" in joined
    assert any("--base-model-dir" in item for item in plan["external_artifacts"])


def test_bootstrap_check_fails_cleanly_when_the_environment_is_absent(tmp_path):
    result = run(BOOTSTRAP, "--check", env=clean_env(HUMANOID_DATA_ROOT=str(tmp_path)))
    assert result.returncode == 1
    assert "no environment at" in result.stderr
    assert not (tmp_path / "venvs").exists()  # --check writes nothing


def test_server_prefers_persistent_env_then_fallback_then_override(tmp_path):
    # A data root with a persistent environment installed.
    with_persistent = tmp_path / "with-persistent"
    persistent = with_persistent / "venvs/flux-model/bin/python"
    persistent.parent.mkdir(parents=True)
    persistent.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    persistent.chmod(persistent.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    planned = json.loads(run(SERVER, "--plan",
                             env=clean_env(HUMANOID_DATA_ROOT=str(with_persistent))).stdout)
    assert planned["model_python"] == str(persistent)
    assert planned["model_python_source"] == "persistent"
    assert planned["persistent_exists"] is True
    assert planned["bootstrap"].endswith("scripts/flux-bootstrap-model-env.sh")
    assert planned["lerobot_commit"] == LEROBOT_COMMIT
    assert planned["lock_exists"] is True
    assert planned["lock_sha256"]

    override = json.loads(run(SERVER, "--plan",
                              env=clean_env(HUMANOID_DATA_ROOT=str(with_persistent),
                                            FLUX_MODEL_PYTHON="/usr/bin/python3")).stdout)
    assert override["model_python"] == "/usr/bin/python3"
    assert override["model_python_source"] == "override"

    # A checkout without the persistent environment: the validated ephemeral
    # interpreter is used when present, and its absence is reported cleanly
    # instead of silently starting nothing.
    without = tmp_path / "without"
    fallback = json.loads(run(SERVER, "--plan",
                              env=clean_env(HUMANOID_DATA_ROOT=str(without),
                                            FLUX_MODEL_FALLBACK_PYTHON="/usr/bin/python3")).stdout)
    assert fallback["model_python"] == "/usr/bin/python3"
    assert fallback["model_python_source"] == "ephemeral-fallback"

    missing = json.loads(run(SERVER, "--plan",
                             env=clean_env(HUMANOID_DATA_ROOT=str(without),
                                           FLUX_MODEL_FALLBACK_PYTHON=str(without / "gone"))).stdout)
    assert missing["model_python"] is None
    assert missing["model_python_source"] == "missing"
    refused = run(SERVER, "--check",
                  env=clean_env(HUMANOID_DATA_ROOT=str(without),
                                FLUX_MODEL_FALLBACK_PYTHON=str(without / "gone")))
    assert refused.returncode == 2
    assert "no model interpreter found" in refused.stderr
    assert "flux-bootstrap-model-env.sh" in refused.stderr


def test_server_plan_does_not_require_docker_or_write_anything(tmp_path):
    result = run(SERVER, "--plan", env=clean_env(HUMANOID_DATA_ROOT=str(tmp_path)))
    assert result.returncode == 0
    plan = json.loads(result.stdout)
    assert plan["port"] == 5561
    assert plan["checkpoint"].startswith(str(tmp_path))
    assert plan["checkpoint_exists"] is False
    assert not (tmp_path / "outputs").exists()
    assert "DEALER/ROUTER" in plan["protocol"]


def test_server_runs_in_foreground_and_rejects_session_management_flags():
    source = SERVER.read_text(encoding="utf-8")
    assert 'exec "$model_python" "$server"' in source
    assert "setsid" not in source
    for flag in ("--stop", "--status", "--foreground", "--log-dir"):
        result = run(SERVER, flag)
        assert result.returncode == 2
        assert "unknown argument" in result.stderr


def test_dev_entry_reaches_the_same_bootstrap_plan():
    # `./dev.sh flux-model-env --plan` is the documented entry point; it must
    # resolve to the same pinned plan without requiring the model environment.
    # dev.sh sources the repository .env, so the data root there is the
    # checkout's own; only the pins and the venv layout are asserted.
    result = subprocess.run(["bash", str(REPO / "dev.sh"), "flux-model-env", "--plan"],
                            capture_output=True, text=True, cwd=REPO,
                            env=clean_env(FLUX_MODEL_VENV="/tmp/flux-model-env-test/venvs/flux-model"))
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["lerobot_commit"] == LEROBOT_COMMIT
    assert plan["venv"] == "/tmp/flux-model-env-test/venvs/flux-model"
    assert plan["venv_exists"] is False
    assert not Path("/tmp/flux-model-env-test").exists()
