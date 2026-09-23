#!/usr/bin/env python3
"""Run pinned SONIC VLA with opt-in simulation-clock action timing."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import tempfile
from pathlib import Path

UPSTREAM = Path("/opt/src/sonic/gear_sonic/scripts/run_vla_inference.py")


def transform_source(source: str) -> str:
    replacements = (
        (
            "import time\n\nimport numpy as np",
            "import time\nimport os\n\nfrom humanoid_lab.psi0_bridge.policy_clock import PolicyClock, PolicyTime\n"
            "from humanoid_lab.psi0_bridge.sim_clock import SimulationClockPacer\n"
            "from humanoid_lab.psi0_bridge.groot_clock_runtime import advance_chunk_index\n"
            "from humanoid_lab.psi0_bridge.groot_hand_contract import (\n"
            "    left_hand_action_to_live, left_hand_observation_to_robot_model_actuated,\n"
            ")\n"
            "from humanoid_lab.psi0_bridge.groot_capture_runtime import (\n"
            "    bind_processed_action, capture_get_action, chunk_installed, row_executed,\n"
            ")\n\n"
            "_POLICY_CLOCK = PolicyClock.from_environment()\n\nimport numpy as np",
        ),
        (
            "        action, _info = policy.get_action(observation)\n",
            "        action, _info = capture_get_action(policy, observation)\n",
        ),
        (
            "    # Copy index finger data to middle finger (hardware coupling)\n"
            "    state_msg[\"left_hand_q\"][5] = state_msg[\"left_hand_q\"][3]\n"
            "    state_msg[\"left_hand_q\"][6] = state_msg[\"left_hand_q\"][4]\n",
            "    state_msg[\"left_hand_q\"] = left_hand_observation_to_robot_model_actuated(\n"
            "        state_msg[\"left_hand_q\"], os.environ[\"HUMANOID_GROOT_LEFT_HAND_CONTRACT\"]\n"
            "    )\n",
        ),
        (
            "        processed_action = concat_action(robot_model, action)\n",
            "        processed_action = concat_action(robot_model, action)\n"
            "        left_hand_action_to_live(\n"
            "            processed_action, os.environ[\"HUMANOID_GROOT_LEFT_HAND_CONTRACT\"]\n"
            "        )\n"
            "        bind_processed_action(action, processed_action)\n",
        ),
        (
            "                inference_start_time = time.monotonic()\n",
            "                inference_start_time = _POLICY_CLOCK.now()\n",
        ),
        (
            "    loop_rate = config.action_publish_rate\n    loop_period = 1.0 / loop_rate\n",
            "    loop_rate = config.action_publish_rate\n    loop_period = 1.0 / loop_rate\n"
            "    policy_clock = _POLICY_CLOCK\n"
            "    action_pacer = SimulationClockPacer(config.action_publish_rate)\n"
            "    print(f'[policy-clock] mode={policy_clock.mode} file={policy_clock.path}', flush=True)\n",
        ),
        (
            "    last_inference_time = 0.0\n",
            "    last_inference_time: PolicyTime | None = None\n",
        ),
        (
            "            t_start = time.monotonic()\n            check_keyboard_input()\n",
            "            t_start = time.monotonic()\n            policy_time = policy_clock.now()\n            check_keyboard_input()\n",
        ),
        (
            "                inference_delay = time.monotonic() - inference_start_time\n",
            "                inference_delay = policy_clock.latency_seconds(inference_start_time)\n",
        ),
        (
            "                last_inference_time = time.monotonic()\n",
            "                last_inference_time = policy_time\n"
            "                chunk_installed(processed_action, action_chunk_index, inference_delay)\n",
        ),
        (
            "                time_since_last_inference=(time.monotonic() - last_inference_time),\n",
            "                time_since_last_inference=(\n"
            "                    float('inf') if last_inference_time is None\n"
            "                    else policy_clock.latency_seconds(last_inference_time, policy_time)\n"
            "                ),\n",
        ),
        (
            "                    zmq_socket.send(zmq_message)\n                    last_sent_motion_token = motion_token.copy()\n",
            "                    zmq_socket.send(zmq_message)\n"
            "                    row_executed(processed_action, current_idx, int(frame_index[0]))\n"
            "                    last_sent_motion_token = motion_token.copy()\n",
        ),
        (
            "            with telemetry.timer(\"total_loop\"):\n",
            "            rows_due = 1\n"
            "            if policy_clock.mode == 'simulation':\n"
            "                clock_update = action_pacer.update(policy_time.seconds, policy_time.generation)\n"
            "                rows_due = clock_update.rows_due\n"
            "                action_chunk_index, should_publish = advance_chunk_index(\n"
            "                    action_chunk_index, rows_due, config.action_horizon\n"
            "                )\n"
            "                if not should_publish:\n"
            "                    time.sleep(0.001)\n"
            "                    continue\n\n"
            "            with telemetry.timer(\"total_loop\"):\n",
        ),
    )
    transformed = source
    for old, new in replacements:
        count = transformed.count(old)
        if count != 1:
            raise RuntimeError(f"upstream SONIC timing anchor matched {count} times: {old[:60]!r}")
        transformed = transformed.replace(old, new)
    return transformed


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy-clock", choices=("wall", "simulation"), default="simulation")
    parser.add_argument("--policy-clock-file", type=Path)
    parser.add_argument("--policy-clock-timeout-s", type=float, default=5.0)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--capture-max-requests", type=int, default=32)
    parser.add_argument(
        "--left-hand-contract",
        choices=("compatibility", "model-independent", "model-coupled"),
        default="model-independent",
    )
    return parser.parse_known_args()


def main() -> None:
    args, upstream_args = parse_args()
    if args.policy_clock == "simulation" and args.policy_clock_file is None:
        raise SystemExit("--policy-clock simulation requires --policy-clock-file")
    if args.policy_clock == "wall" and args.policy_clock_file is not None:
        raise SystemExit("--policy-clock-file is only valid with --policy-clock simulation")
    if args.policy_clock_timeout_s <= 0:
        raise SystemExit("--policy-clock-timeout-s must be positive")
    if args.capture_max_requests <= 0:
        raise SystemExit("--capture-max-requests must be positive")

    os.environ["HUMANOID_POLICY_CLOCK"] = args.policy_clock
    os.environ["HUMANOID_POLICY_CLOCK_TIMEOUT_S"] = str(args.policy_clock_timeout_s)
    os.environ["HUMANOID_GROOT_LEFT_HAND_CONTRACT"] = args.left_hand_contract
    if args.policy_clock_file is not None:
        os.environ["HUMANOID_POLICY_CLOCK_FILE"] = str(args.policy_clock_file)
    if args.capture_dir is not None:
        os.environ["HUMANOID_GROOT_CAPTURE_DIR"] = str(args.capture_dir)
        os.environ["HUMANOID_GROOT_CAPTURE_MAX_REQUESTS"] = str(args.capture_max_requests)

    transformed = transform_source(UPSTREAM.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="humanoid-groot-vla-") as directory:
        patched = Path(directory) / "run_vla_inference.py"
        patched.write_text(transformed, encoding="utf-8")
        print(
            f"[policy-clock-wrapper] upstream={UPSTREAM} mode={args.policy_clock} "
            f"clock_file={args.policy_clock_file} left_hand_contract={args.left_hand_contract} "
            f"patched={patched}",
            flush=True,
        )
        sys.argv = [str(patched), *upstream_args]
        runpy.run_path(str(patched), run_name="__main__")


if __name__ == "__main__":
    main()
