"""The policy server as a process the bridge owns and can restart.

``serve_psi0_sonic`` is started as a child of this bridge, so a checkpoint
switch is a controlled restart: the child is terminated (TERM, then KILL after a
bounded wait), a new one is started on the same ``:8014`` port, and the caller
verifies the served ``/info`` identity before anything is marked ready.

Only the child's own pid is signalled -- never the process group -- so the
bridge can stop and replace its server without touching the launcher (dev.sh and
the container-side lease/watchdog wrappers own *this* process, not the policy
server).  The child is spawned in the bridge's process group, so if the bridge
is torn down by the launcher lease the child dies with it even when the bridge's
own cleanup never runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

#: The canonical deployment invocation (same flags dev.sh used when the launcher
#: owned the server): ``--action_exec_horizon 30`` matches the bridge's 30 Hz
#: Protocol v4 stream and ``--rtc`` enables the replan path verified at startup.
SERVE_EXECUTABLE = "serve_psi0_sonic"

DEFAULT_STOP_TIMEOUT_S = 15.0
DEFAULT_KILL_WAIT_S = 10.0


class PolicyServerError(RuntimeError):
    """The policy server could not be started, stopped or reached."""


def serve_command(run_dir: Path, step: int, port: int,
                  executable: str = SERVE_EXECUTABLE) -> list[str]:
    """Build the canonical ``serve_psi0_sonic`` command line for one checkpoint."""
    return [
        executable,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--action_exec_horizon", "30",
        "--policy", "psi",
        "--rtc",
        "--run-dir", str(run_dir),
        "--ckpt-step", str(step),
    ]


def probe_info(host: str, port: int, *, timeout_s: float = 2.0) -> Optional[dict[str, Any]]:
    """Return the decoded ``/info`` body if anything serves the port, else ``None``.

    Only called to refuse a *foreign* server before we spawn our own; identity is
    still enforced later through the full ``validate_info`` path.
    """
    url = f"http://{host}:{port}/info"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


class PolicyServerProcess:
    """Owns one ``serve_psi0_sonic`` child for the lifetime of the bridge."""

    def __init__(
        self,
        *,
        port: int,
        log_path: Optional[Path] = None,
        stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
        kill_wait_s: float = DEFAULT_KILL_WAIT_S,
    ) -> None:
        self.port = int(port)
        self.log_path = Path(log_path) if log_path is not None else None
        self.stop_timeout_s = float(stop_timeout_s)
        self.kill_wait_s = float(kill_wait_s)
        self._process: Optional[subprocess.Popen] = None
        self._run_dir: Optional[Path] = None
        self._step: Optional[int] = None
        self._log_file = None

    # -- description -------------------------------------------------------

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def run_dir(self) -> Optional[Path]:
        return self._run_dir

    @property
    def step(self) -> Optional[int]:
        return self._step

    def command(self, run_dir: Path, step: int) -> list[str]:
        """The exact command line to spawn; tests override this seam."""
        return serve_command(run_dir, step, self.port)

    # -- lifecycle ---------------------------------------------------------

    def start(self, run_dir: Path, step: int) -> None:
        if self.alive:
            raise PolicyServerError(
                f"a policy server (pid {self.pid}) is already running for "
                f"{self._run_dir} step {self._step}"
            )
        run_dir = Path(run_dir)
        if not (run_dir / "checkpoints" / f"ckpt_{step}").is_dir():
            raise PolicyServerError(
                f"cannot start the policy server: {run_dir} has no checkpoints/ckpt_{step}"
            )
        if probe_info("127.0.0.1", self.port) is not None:
            raise PolicyServerError(
                f"something already serves :{self.port} on localhost; refusing to start a "
                "second policy server"
            )
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "ab", buffering=0)
            stdout = self._log_file
        else:
            stdout = None
        try:
            # No start_new_session: the child stays in the bridge's process
            # group, so the launcher's lease teardown reaps it too.
            self._process = subprocess.Popen(
                self.command(run_dir, step),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.STDOUT if stdout is not None else None,
                env=dict(os.environ),
            )
        except OSError as exc:
            self._close_log()
            raise PolicyServerError(
                f"cannot start {' '.join(self.command(run_dir, step))}: {exc}"
            ) from exc
        self._run_dir = run_dir
        self._step = int(step)

    def stop(self, *, timeout_s: Optional[float] = None) -> None:
        """Terminate and reap the child; idempotent and safe when it never started."""
        process, self._process = self._process, None
        self._close_log()
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            deadline = time.monotonic() + (timeout_s if timeout_s is not None else self.stop_timeout_s)
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if process.poll() is None:
                process.kill()
                kill_deadline = time.monotonic() + self.kill_wait_s
                while process.poll() is None and time.monotonic() < kill_deadline:
                    time.sleep(0.1)
                if process.poll() is None:
                    raise PolicyServerError(
                        f"policy server pid {process.pid} did not exit after TERM/KILL"
                    )
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - kill path only
            pass
        self._run_dir = None
        self._step = None

    def close(self) -> None:
        self.stop()

    # -- helpers -----------------------------------------------------------

    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            finally:
                self._log_file = None
