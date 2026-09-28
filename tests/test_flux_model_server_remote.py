"""Focused shell checks for remote-bind and CURVE handling in the Flux model
server wrapper.

`scripts/flux-model-server.sh` validates the bind IP and the CURVE key layout
before it resolves an interpreter or checkpoint, so every case here runs
without the model environment, docker, the GPU or a live server: `--plan` and
the rejected paths never execute the vendored zmq_server.py. The key files hold
markers instead of real certificates so the tests can also assert that no key
material reaches stdout or stderr.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "scripts/flux-model-server.sh"
SECRET_MARKER = "FLUX-TEST-SERVER-SECRET-NOT-FOR-LOGS"
CLIENT_MARKER = "FLUX-TEST-CLIENT-PUBLIC-NOT-FOR-LOGS"
DEFAULT_CHECKPOINT = ("models", "flux-dex3", "checkpoint-2500")


def clean_env(**overrides):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("FLUX_", "HUMANOID_DATA_ROOT"))}
    environment.update(overrides)
    return environment


def isolated_env(tmp_path, **overrides):
    # No persistent venv and no ephemeral fallback: only an explicit --python
    # can resolve an interpreter, and every missing-interpreter path stops
    # before exec.
    environment = clean_env(HUMANOID_DATA_ROOT=str(tmp_path),
                            FLUX_MODEL_FALLBACK_PYTHON=str(tmp_path / "no-python"))
    environment.update(overrides)
    return environment


def run(*args, env):
    return subprocess.run(["bash", str(SERVER), *args],
                          capture_output=True, text=True, cwd=REPO, env=env)


def plan(*args, env):
    result = run("--plan", *args, env=env)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def make_curve_material(tmp_path):
    secret = tmp_path / "server.key_secret"
    secret.write_text(SECRET_MARKER + "\n", encoding="utf-8")
    keys = tmp_path / "client-keys"
    keys.mkdir()
    (keys / "robot1.key").write_text(CLIENT_MARKER + "\n", encoding="utf-8")
    return secret, keys


def test_loopback_stays_the_default_and_carries_no_curve_arguments(tmp_path):
    resolved = plan(env=isolated_env(tmp_path))
    assert resolved["bind_ip"] == "127.0.0.1"
    assert resolved["remote"] is False
    assert resolved["server_secret_key"] is None
    assert resolved["client_keys_dir"] is None
    assert resolved["robot_endpoint"] == "tcp://127.0.0.1:5561"
    assert resolved["model_args"] == [
        "--bind-ip", "127.0.0.1",
        "--port", "5561",
        "--checkpoint", str(tmp_path.joinpath(*DEFAULT_CHECKPOINT)),
    ]


def test_remote_plan_selects_the_explicit_ip_and_passes_both_curve_flags(tmp_path):
    secret, keys = make_curve_material(tmp_path)
    resolved = plan("--bind-ip", "192.168.50.10",
                    "--server-secret-key", str(secret),
                    "--client-keys-dir", str(keys),
                    env=isolated_env(tmp_path))
    assert resolved["remote"] is True
    assert resolved["bind_ip"] == "192.168.50.10"
    assert resolved["server_secret_key"] == str(secret)
    assert resolved["client_keys_dir"] == str(keys)
    assert resolved["robot_endpoint"] == "tcp://192.168.50.10:5561"
    assert resolved["server"].endswith("examples/dex3/zmq_server.py")
    assert resolved["model_args"] == [
        "--bind-ip", "192.168.50.10",
        "--port", "5561",
        "--checkpoint", str(tmp_path.joinpath(*DEFAULT_CHECKPOINT)),
        "--server-secret-key", str(secret),
        "--client-keys-dir", str(keys),
    ]


def test_remote_configuration_is_available_through_the_environment(tmp_path):
    secret, keys = make_curve_material(tmp_path)
    resolved = plan(env=isolated_env(tmp_path,
                                     FLUX_MODEL_PORT="5570",
                                     FLUX_MODEL_BIND_IP="10.1.2.3",
                                     FLUX_MODEL_SERVER_SECRET_KEY=str(secret),
                                     FLUX_MODEL_CLIENT_KEYS_DIR=str(keys)))
    assert resolved["remote"] is True
    assert resolved["robot_endpoint"] == "tcp://10.1.2.3:5570"
    assert resolved["model_args"][0:2] == ["--bind-ip", "10.1.2.3"]
    assert resolved["server_secret_key"] == str(secret)
    assert resolved["client_keys_dir"] == str(keys)


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", "0:0:0:0:0:0:0:0"])
def test_wildcard_binds_are_refused_before_any_model_work(tmp_path, wildcard):
    result = run("--plan", "--bind-ip", wildcard, env=isolated_env(tmp_path))
    assert result.returncode == 2
    assert "wildcard" in result.stderr
    assert "no model interpreter" not in result.stderr


def test_wildcard_is_refused_even_with_valid_curve_material(tmp_path):
    secret, keys = make_curve_material(tmp_path)
    result = run("--plan", "--bind-ip", "0.0.0.0",
                 "--server-secret-key", str(secret), "--client-keys-dir", str(keys),
                 env=isolated_env(tmp_path))
    assert result.returncode == 2
    assert "wildcard" in result.stderr


@pytest.mark.parametrize("value", ["localhost", "192.168.1.256",
                                   "192.168.1.5:5561", ""])
def test_hostnames_and_malformed_ips_are_refused(tmp_path, value):
    result = run("--plan", "--bind-ip", value, env=isolated_env(tmp_path))
    assert result.returncode == 2
    assert "explicit IP literal" in result.stderr


def test_non_loopback_bind_requires_both_curve_inputs(tmp_path):
    secret, keys = make_curve_material(tmp_path)
    missing_both = run("--plan", "--bind-ip", "192.168.50.10",
                       env=isolated_env(tmp_path))
    assert missing_both.returncode == 2
    assert "--server-secret-key" in missing_both.stderr

    missing_dir = run("--plan", "--bind-ip", "192.168.50.10",
                      "--server-secret-key", str(secret),
                      env=isolated_env(tmp_path))
    assert missing_dir.returncode == 2
    assert "--client-keys-dir" in missing_dir.stderr

    missing_secret = run("--plan", "--bind-ip", "192.168.50.10",
                         "--client-keys-dir", str(keys),
                         env=isolated_env(tmp_path))
    assert missing_secret.returncode == 2
    assert "--server-secret-key" in missing_secret.stderr


def test_remote_bind_validates_the_curve_files_before_startup(tmp_path):
    secret, keys = make_curve_material(tmp_path)
    missing_secret = run("--plan", "--bind-ip", "192.168.50.10",
                         "--server-secret-key", str(tmp_path / "absent.key_secret"),
                         "--client-keys-dir", str(keys),
                         env=isolated_env(tmp_path))
    assert missing_secret.returncode == 2
    assert "readable file" in missing_secret.stderr

    missing_dir = run("--plan", "--bind-ip", "192.168.50.10",
                      "--server-secret-key", str(secret),
                      "--client-keys-dir", str(tmp_path / "absent-keys"),
                      env=isolated_env(tmp_path))
    assert missing_dir.returncode == 2
    assert "directory" in missing_dir.stderr

    empty_dir = tmp_path / "empty-keys"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").write_text("not a certificate\n", encoding="utf-8")
    no_certificates = run("--plan", "--bind-ip", "192.168.50.10",
                          "--server-secret-key", str(secret),
                          "--client-keys-dir", str(empty_dir),
                          env=isolated_env(tmp_path))
    assert no_certificates.returncode == 2
    assert "*.key" in no_certificates.stderr


def test_ipv6_remote_bind_brackets_the_robot_endpoint(tmp_path):
    secret, keys = make_curve_material(tmp_path)
    resolved = plan("--bind-ip", "fd00::5", "--server-secret-key", str(secret),
                    "--client-keys-dir", str(keys), env=isolated_env(tmp_path))
    assert resolved["remote"] is True
    assert resolved["robot_endpoint"] == "tcp://[fd00::5]:5561"


def test_key_material_never_reaches_stdout_or_stderr(tmp_path):
    secret, keys = make_curve_material(tmp_path)
    remote = ("--bind-ip", "192.168.50.10",
              "--server-secret-key", str(secret),
              "--client-keys-dir", str(keys))

    planned = run("--plan", *remote, env=isolated_env(tmp_path))
    assert planned.returncode == 0
    assert SECRET_MARKER not in planned.stdout + planned.stderr
    assert CLIENT_MARKER not in planned.stdout + planned.stderr

    # Starting without a prepared environment stops at the interpreter check,
    # so nothing executes the server and nothing echoes key contents.
    refused = run(*remote, env=isolated_env(tmp_path))
    assert refused.returncode == 2
    assert "no model interpreter" in refused.stderr
    assert SECRET_MARKER not in refused.stdout + refused.stderr
    assert CLIENT_MARKER not in refused.stdout + refused.stderr


def test_remote_check_reports_the_robot_contract_without_key_material(tmp_path):
    secret, keys = make_curve_material(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    stub = tmp_path / "python"
    stub.write_text("#!/bin/sh\n"
                    "if [ \"$#\" -eq 1 ] && [ \"$1\" = \"-\" ]; then cat >/dev/null; fi\n"
                    "exit 0\n", encoding="utf-8")
    stub.chmod(0o755)

    result = run("--check", "--python", str(stub), "--checkpoint", str(checkpoint),
                 "--bind-ip", "192.168.50.10", "--server-secret-key", str(secret),
                 "--client-keys-dir", str(keys), env=isolated_env(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "preflight OK" in result.stdout
    assert "endpoint=tcp://192.168.50.10:5561" in result.stderr
    assert "client_certificate=" in result.stderr
    assert "server_public_key=" in result.stderr
    assert SECRET_MARKER not in result.stdout + result.stderr
    assert CLIENT_MARKER not in result.stdout + result.stderr


def test_help_documents_the_remote_contract(tmp_path):
    result = run("--help", env=isolated_env(tmp_path))
    assert result.returncode == 0
    for needle in ("--bind-ip", "--server-secret-key", "--client-keys-dir",
                   "client_certificate", "server_public_key", "127.0.0.1"):
        assert needle in result.stderr
