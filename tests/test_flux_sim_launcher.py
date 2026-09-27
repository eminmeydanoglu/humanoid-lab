"""Focused checks for the flux-sim launcher's canonical defaults and options.

Everything here runs the launcher's `config` plan: it resolves paths, the
command mode and the prerequisite checks without touching docker, the
simulator or the model server.  The full e2e run itself is exercised by
`./dev.sh flux-sim e2e`, not by these tests.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/flux-sim.sh"
CANONICAL_PROFILE = "configs/profiles/isaac-g1-flux-dex3-pickapple.json"
CANONICAL_MOTOR_CONFIG = "configs/flux/flux-dex3-sim-motor-config.json"


def plan(*args, env_overrides=None):
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("FLUX_ISAAC_PROFILE", "FLUX_MOTOR_CONFIG", "FLUX_DURATION")}
    if env_overrides:
        environment.update(env_overrides)
    result = subprocess.run(["bash", str(SCRIPT), "config", *args],
                            capture_output=True, text=True, cwd=REPO, env=environment)
    if result.returncode != 0:
        pytest.fail("config failed: %s" % result.stderr)
    return json.loads(result.stdout)


def test_canonical_run_selects_fixed_root_scene_and_motor_config():
    resolved = plan()
    assert resolved["profile"] == CANONICAL_PROFILE
    assert resolved["profile_exists"] is True
    assert resolved["motor_config"] == CANONICAL_MOTOR_CONFIG
    assert resolved["motor_config_exists"] is True
    assert resolved["motor_commands"] is True
    assert resolved["mode"] == "livestream"
    # The explicit ports: the server default 5557 belongs to the SONIC state
    # port on this host, and the camera feed is the simulator's ego_view port.
    assert resolved["model_endpoint"] == "tcp://127.0.0.1:5561"
    assert resolved["camera_endpoint"] == "tcp://127.0.0.1:5555"
    assert resolved["sonic_process"] is False


def test_display_modes_are_explicit_and_only_one_is_allowed():
    # The canonical run is watched through Isaac's WebRTC livestream; --gui and
    # --headless remain available, and a contradictory pair must fail before
    # anything starts.
    assert plan("--livestream")["mode"] == "livestream"
    assert plan("--gui")["mode"] == "gui"
    assert plan("--headless")["mode"] == "headless"
    assert plan(env_overrides={"FLUX_ISAAC_MODE": "headless"})["mode"] == "headless"
    # An explicit flag wins over the environment.
    assert plan("--gui", env_overrides={"FLUX_ISAAC_MODE": "headless"})["mode"] == "gui"

    for args in (["--gui", "--headless"], ["--livestream", "--gui"]):
        result = subprocess.run(["bash", str(SCRIPT), "config", *args],
                                capture_output=True, text=True, cwd=REPO)
        assert result.returncode == 2, args
        assert "only one of" in result.stderr

    invalid = subprocess.run(["bash", str(SCRIPT), "config"], capture_output=True,
                             text=True, cwd=REPO,
                             env={**os.environ, "FLUX_ISAAC_MODE": "window"})
    assert invalid.returncode == 2
    assert "display mode must be" in invalid.stderr


def test_scene_and_actuation_files_are_repository_files_not_aliases():
    profile = json.loads((REPO / CANONICAL_PROFILE).read_text(encoding="utf-8"))
    # The canonical scene carries the fixed-root Flux controller the launcher
    # depends on; a swapped default must fail here, not in a live run.
    assert profile["controller"]["provider"] == "flux_dds"
    assert profile["robot"]["fixed_base"] is True
    motor = json.loads((REPO / CANONICAL_MOTOR_CONFIG).read_text(encoding="utf-8"))
    assert len(motor["joint_limits_rad"]) == 28
    assert motor["control_authority_confirmed"] is True


def test_no_motor_commands_is_the_dry_run_opt_out():
    resolved = plan("--no-motor-commands")
    assert resolved["motor_commands"] is False
    assert resolved["motor_config"] == ""
    assert resolved["motor_config_exists"] is None
    assert resolved["profile"] == CANONICAL_PROFILE  # the scene stays the same


def test_no_motor_commands_conflicts_with_an_explicit_config():
    result = subprocess.run(
        ["bash", str(SCRIPT), "config", "--no-motor-commands",
         "--motor-output-config", CANONICAL_MOTOR_CONFIG],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 2
    assert "conflicts" in result.stderr


def test_cli_and_environment_overrides_win():
    override = plan("--profile", "configs/profiles/isaac-g1-sonic-fixed-base-dex3.json",
                    "--motor-output-config", "configs/flux/missing.json")
    assert override["profile"] == "configs/profiles/isaac-g1-sonic-fixed-base-dex3.json"
    assert override["motor_config"] == "configs/flux/missing.json"
    assert override["motor_config_exists"] is False

    from_env = plan(env_overrides={"FLUX_MOTOR_CONFIG": "configs/flux/missing.json"})
    assert from_env["motor_config"] == "configs/flux/missing.json"
    assert from_env["motor_commands"] is True


def test_non_numeric_options_fail_before_anything_starts():
    for args in (["--duration", "soon"], ["--run-s", ""], ["--port", "55x1"]):
        result = subprocess.run(["bash", str(SCRIPT), "config", *args],
                                capture_output=True, text=True, cwd=REPO)
        assert result.returncode == 2, args
        assert "must be a number" in result.stderr


def test_up_log_follow_and_detach_are_explicit():
    assert plan("--detach")["profile"] == CANONICAL_PROFILE
    source = SCRIPT.read_text()
    assert 'up) cmd_up; [ "$follow_logs" -eq 0 ] || cmd_logs' in source
    assert 'logs) cmd_logs' in source
    assert 'cmd_up || up_rc=$?' in source  # e2e stays bounded and does not follow logs
    assert '"$run_dir/isaac.log" "$run_dir/ros-launch.log"' in source
    assert 'model-server-$port.log' in source


def test_simulation_age_limit_is_bounded_and_does_not_change_robot_default():
    import importlib.util

    base = REPO / "third_party/flux/flux-inference/ros2"
    launch = (base / "flux_sim_camera/launch/flux_sim.launch.py").read_text()
    node = (base / "flux_dex3/flux_dex3/node.py").read_text()
    executor_path = base / "flux_dex3/flux_dex3/executor.py"
    spec = importlib.util.spec_from_file_location("flux_test_executor", executor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ChunkExecutor().max_chunk_age_s == 1.2
    assert 'DeclareLaunchArgument("max_chunk_age_s", default_value="1.6")' in launch
    assert '"max_chunk_age_s": max_chunk_age_s' in launch
    assert 'self.declare_parameter("max_chunk_age_s", 1.2)' in node
    assert 'ChunkExecutor(max_chunk_age_s=max_chunk_age_s)' in node
    assert 'max_chunk_age_s <= 2.0' in node
    accepted = module.ChunkExecutor(max_chunk_age_s=1.6)
    accepted.start("sim")
    import numpy as np
    accepted.accept("sim", 0, 1.6, np.zeros((32, 28), dtype=np.float32))
    rejected = module.ChunkExecutor(max_chunk_age_s=1.6)
    rejected.start("sim")
    with pytest.raises(ValueError, match="stale observation"):
        rejected.accept("sim", 0, 1.601, np.zeros((32, 28), dtype=np.float32))


def test_help_documents_defaults_prerequisites_and_the_opt_out():
    result = subprocess.run(["bash", str(SCRIPT), "--help"],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0
    assert CANONICAL_PROFILE in result.stderr + result.stdout
    assert CANONICAL_MOTOR_CONFIG in result.stderr + result.stdout
    assert "--no-motor-commands" in result.stderr + result.stdout
    assert "checkpoint_identity" in result.stderr + result.stdout
    assert "SONIC" in result.stderr + result.stdout
