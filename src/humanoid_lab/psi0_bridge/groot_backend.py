"""Lifecycle wrapper around NVIDIA's upstream GR00T server and SONIC VLA client."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


class GrootBackendError(RuntimeError):
    pass


def validate_groot_checkpoint(path: Path) -> None:
    if not path.is_dir():
        raise GrootBackendError(f"GR00T checkpoint does not exist: {path}")
    missing = [name for name in ("config.json", "processor_config.json", "statistics.json")
               if not (path / name).is_file()]
    if missing:
        raise GrootBackendError(f"GR00T checkpoint is missing: {', '.join(missing)}")


class GrootProcessGroup:
    """Own only the upstream GR00T policy server and VLA client."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        prompt: str,
        log_dir: Path,
        server_port: int = 5550,
        action_port: int = 5560,
        ready_timeout_s: float = 300.0,
        policy_clock: str = "simulation",
        policy_clock_file: Path | None = None,
        policy_clock_timeout_s: float = 5.0,
        capture_dir: Path | None = None,
        capture_max_requests: int = 32,
        left_hand_contract: str = "model-independent",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.prompt = prompt
        self.log_dir = Path(log_dir)
        self.server_port = int(server_port)
        self.action_port = int(action_port)
        self.ready_timeout_s = float(ready_timeout_s)
        if policy_clock not in ("wall", "simulation"):
            raise ValueError(f"unknown policy clock {policy_clock!r}")
        if policy_clock == "simulation" and policy_clock_file is None:
            raise ValueError("simulation policy clock requires policy_clock_file")
        self.policy_clock = policy_clock
        self.policy_clock_file = None if policy_clock_file is None else Path(policy_clock_file)
        self.policy_clock_timeout_s = float(policy_clock_timeout_s)
        if capture_max_requests <= 0:
            raise ValueError("capture_max_requests must be positive")
        self.capture_dir = None if capture_dir is None else Path(capture_dir)
        self.capture_max_requests = int(capture_max_requests)
        if left_hand_contract not in ("compatibility", "model-independent", "model-coupled"):
            raise ValueError(f"unknown GR00T left-hand contract {left_hand_contract!r}")
        self.left_hand_contract = left_hand_contract
        self._server: subprocess.Popen | None = None
        self._vla: subprocess.Popen | None = None
        self._logs: list[Any] = []

    @property
    def alive(self) -> bool:
        return self._server is not None and self._server.poll() is None and self._vla is not None and self._vla.poll() is None

    @property
    def pid(self) -> int | None:
        return None if self._server is None else self._server.pid

    def start(self) -> None:
        if self.alive:
            return
        validate_groot_checkpoint(self.checkpoint)
        self.stop()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        server_cmd = (
            "source /opt/humanoid-lab/entrypoint.sh && use-groot && "
            "cd /opt/src/isaac-groot && exec python gr00t/eval/run_gr00t_server.py "
            f"--model-path={shlex.quote(str(self.checkpoint))} "
            "--embodiment-tag=UNITREE_G1_SONIC --device=cuda:0 "
            f"--host=0.0.0.0 --port={self.server_port} --strict"
        )
        self._server = self._spawn(server_cmd, self.log_dir / "groot-server.log")
        self._wait_server()
        vla_cmd = (
            "source /opt/humanoid-lab/entrypoint.sh && use-sonic-sim && "
            "export PYTHONPATH=/workspace/humanoid-lab/src:/opt/src/sonic:/opt/src/isaac-groot && "
            f"export HUMANOID_GROOT_CHECKPOINT={shlex.quote(str(self.checkpoint))} && "
            "cd /workspace/humanoid-lab && exec python scripts/run-groot-vla.py "
            f"--policy-clock={self.policy_clock} "
            + (f"--policy-clock-file={shlex.quote(str(self.policy_clock_file))} "
               if self.policy_clock_file is not None else "")
            + f"--policy-clock-timeout-s={self.policy_clock_timeout_s:g} "
            + (f"--left-hand-contract={self.left_hand_contract} "
               if self.left_hand_contract != "compatibility" else "")
            + (f"--capture-dir={shlex.quote(str(self.capture_dir))} "
               f"--capture-max-requests={self.capture_max_requests} "
               if self.capture_dir is not None else "")
            + f"--host=127.0.0.1 --port={self.server_port} "
            "--camera-host=127.0.0.1 --camera-port=5555 "
            "--state-zmq-host=127.0.0.1 --state-zmq-port=5557 "
            f"--action-zmq-host=0.0.0.0 --action-zmq-port={self.action_port} "
            "--keyboard-zmq-host=127.0.0.1 --keyboard-zmq-port=5580 "
            "--embodiment-tag=unitree_g1_sonic "
            f"--prompt={shlex.quote(self.prompt)}"
        )
        self._vla = self._spawn(vla_cmd, self.log_dir / "sonic-vla.log")
        time.sleep(4.0)
        if self._vla.poll() is not None:
            code = self._vla.returncode
            self.stop()
            raise GrootBackendError(f"NVIDIA SONIC VLA client exited during startup ({code})")

    def stop(self) -> None:
        for process in (self._vla, self._server):
            self._terminate(process)
        self._vla = None
        self._server = None
        for handle in self._logs:
            handle.close()
        self._logs.clear()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "groot",
            "checkpoint": str(self.checkpoint),
            "server_pid": None if self._server is None else self._server.pid,
            "vla_pid": None if self._vla is None else self._vla.pid,
            "alive": self.alive,
        }

    def _spawn(self, command: str, log_path: Path) -> subprocess.Popen:
        handle = open(log_path, "ab", buffering=0)
        self._logs.append(handle)
        try:
            return subprocess.Popen(
                ["bash", "-lc", command], stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ),
            )
        except OSError as exc:
            raise GrootBackendError(f"cannot start upstream GR00T process: {exc}") from exc

    def _wait_server(self) -> None:
        probe = (
            "source /opt/humanoid-lab/entrypoint.sh && use-sonic-sim && "
            "export PYTHONPATH=/opt/src/isaac-groot && python -c "
            "'from gr00t.policy.server_client import PolicyClient; "
            f"raise SystemExit(0 if PolicyClient(host=\"127.0.0.1\", port={self.server_port}, timeout_ms=2000).ping() else 1)'"
        )
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            if self._server is None or self._server.poll() is not None:
                code = None if self._server is None else self._server.returncode
                self.stop()
                raise GrootBackendError(f"GR00T PolicyServer exited during startup ({code})")
            if subprocess.run(["bash", "-lc", probe], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, check=False).returncode == 0:
                return
            time.sleep(1.0)
        self.stop()
        raise GrootBackendError("GR00T PolicyServer did not become ready")

    @staticmethod
    def _terminate(process: subprocess.Popen | None) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
