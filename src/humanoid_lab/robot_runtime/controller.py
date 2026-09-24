"""One operator session for the real robot, using the existing model backends."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Model:
    id: str
    kind: str  # psi or groot
    path: Path
    step: int | None = None


def load_tasks(path: Path) -> dict[str, str]:
    tasks = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["tasks"]
    if not isinstance(tasks, dict) or len(tasks) != 13 or any(
        not isinstance(key, str) or not isinstance(text, str) or not text
        for key, text in tasks.items()
    ):
        raise ValueError("expected the 13 canonical Unitree task prompts")
    return tasks


class RobotSessionError(RuntimeError):
    pass


class RobotController:
    """Switch policies only in planner; never equate a UI command with robot state."""

    def __init__(self, *, monitor: Any, sonic: Any, router: Any, camera_bridge: Any,
                 psi_session: Any, psi_server: Any, groot: Any,
                 models: list[Model], tasks: dict[str, str],
                 command_builder: Any | None = None) -> None:
        if not models or not tasks:
            raise ValueError("models and canonical tasks are required")
        self.monitor, self.sonic, self.router = monitor, sonic, router
        self.camera_bridge = camera_bridge
        self.psi_session, self.psi_server, self.groot = psi_session, psi_server, groot
        self.models = {entry.id: entry for entry in models}
        if len(self.models) != len(models) or any(e.kind not in ("psi", "groot") for e in models):
            raise ValueError("duplicate or unsupported model")
        self.tasks = tasks
        self.model_id = models[0].id
        self.task_id = next(iter(tasks))
        self.state = "IDLE"
        self.error: str | None = None
        self._lock = threading.RLock()
        self._control_lock = threading.Lock()
        self._cancel_start = threading.Event()
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None
        self._groot_playing = False
        self._session_sent_base = 0
        self._command_builder = command_builder

    def open(self) -> None:
        self.monitor.start()
        self.sonic.start()
        self.camera_bridge.start()
        self._stop.clear()
        self._watcher = threading.Thread(target=self._watch, name="robot-health", daemon=True)
        self._watcher.start()

    def close(self) -> None:
        self.stop()
        self._stop.set()
        if self._watcher is not None:
            self._watcher.join(timeout=2)
        self.groot.stop()
        self.psi_server.stop()
        self.psi_session.close()
        self.camera_bridge.close()
        self.sonic.close()
        self.monitor.stop()
        self.router.close()

    def select(self, model_id: str, task_id: str) -> dict[str, Any]:
        with self._lock:
            if self.state not in ("IDLE", "STOPPED"):
                raise RobotSessionError("model and task may change only while idle")
            if model_id not in self.models or task_id not in self.tasks:
                raise RobotSessionError("unknown model or canonical task")
            reported = self.sonic.latest()
            if reported is not None and reported.fresh() and reported.mode != "PLANNER":
                raise RobotSessionError("robot does not report PLANNER mode")
            self.model_id, self.task_id = model_id, task_id
            return self.status()

    def _command(self, planner: bool) -> None:
        builder = self._command_builder
        if builder is None:
            from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message
            builder = build_command_message
        packet = builder(start=True, stop=False, planner=planner)
        for _ in range(3):
            self.router.send_control(packet)
            time.sleep(0.05)

    def _await(self, predicate, timeout_s: float, reason: str) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._cancel_start.is_set():
                raise RobotSessionError("Start was cancelled by Durdur or Sıfırla")
            if predicate():
                return
            time.sleep(0.025)
        raise RobotSessionError(reason)

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.state not in ("IDLE", "STOPPED"):
                raise RobotSessionError(f"cannot start from {self.state}")
            if self.monitor.frame(max_age_s=0.5) is None:
                raise RobotSessionError("no fresh robot RGB camera frame")
            self.state, self.error = "STARTING", None
            self._cancel_start.clear()
            entry = self.models[self.model_id]
            prompt = self.tasks[self.task_id]
            try:
                self._command(planner=True)
                self._await(lambda: self.monitor.state(max_age_s=0.5) is not None,
                            8.0, "no fresh 43D g1_debug state in planner")
                self._await(lambda: (s := self.sonic.latest()) is not None and s.fresh()
                            and s.mode == "PLANNER", 2.0, "robot did not report PLANNER")
                with self._control_lock:
                    if self._cancel_start.is_set():
                        raise RobotSessionError("Start was cancelled")
                    self.router.select(entry.kind)
                    self.router.resume()
                first_count = self.router.status()["sent"]
                if entry.kind == "psi":
                    self.psi_session.config = replace(self.psi_session.config, instruction=prompt)
                    self.psi_server.start(entry.path, entry.step)
                    result = self.psi_session.start()
                    if result["state"] != "RUNNING":
                        raise RobotSessionError(result.get("error") or "Psi0 did not start")
                else:
                    self.groot.prompt = prompt
                    self.groot.start()
                    self._groot_playing = True
                    self._send_groot_key("p")
                self._await(lambda: self.router.status()["sent"] >= first_count + 2,
                            20.0, "model did not publish two valid actions")
                last = self.router.status()["last_time"]
                if last is None or time.time() - last > 0.3:
                    raise RobotSessionError("model action stream is too slow for the robot watchdog")
                if self.monitor.frame(max_age_s=0.5) is None or self.monitor.state(max_age_s=0.5) is None:
                    raise RobotSessionError("camera or state became stale during model startup")
                with self._control_lock:
                    if self._cancel_start.is_set():
                        raise RobotSessionError("Start was cancelled")
                    self._command(planner=False)
                self._await(lambda: (s := self.sonic.latest()) is not None and s.fresh()
                            and s.mode == "STREAMED_MOTION" and
                            s.valid_token_age_ms is not None and s.valid_token_age_ms <= 300,
                            1.0, "robot did not confirm fresh streamed motion")
                if self._cancel_start.is_set():
                    raise RobotSessionError("Start was cancelled")
                self.state = "RUNNING"
            except Exception as exc:
                self._halt(f"{type(exc).__name__}: {exc}")
                raise RobotSessionError(self.error) from exc
            return self.status()

    def _send_groot_key(self, key: str) -> None:
        # The existing GR00T client accepts pause/resume over its keyboard ZMQ.
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind("tcp://127.0.0.1:5580")
            time.sleep(0.2)
            socket.send_string(key)
            time.sleep(0.05)
        finally:
            socket.close(linger=0)
            context.term()

    def _halt(self, error: str | None = None) -> None:
        if error:
            self.state, self.error = "ERROR", error
        report = self.sonic.latest()
        needs_planner = self.state in ("STARTING", "RUNNING", "ERROR") or (
            report is not None and report.fresh() and report.mode == "STREAMED_MOTION"
        )
        self.router.halt()
        self._groot_playing = False
        self.psi_session.stop()
        if needs_planner:
            try:
                self._command(planner=True)
            except Exception as exc:
                error = error or f"planner command failed: {exc}"
        self.groot.stop()
        self.psi_server.stop()
        self.state = "ERROR" if error else "STOPPED"
        self.error = error

    def stop(self) -> dict[str, Any]:
        with self._control_lock:
            self._cancel_start.set()
            self.router.halt()
            report = self.sonic.latest()
            if self.state in ("STARTING", "RUNNING") or (
                report is not None and report.fresh() and report.mode == "STREAMED_MOTION"
            ):
                self._command(planner=True)
        with self._lock:
            self._halt()
            return self.status()

    def reset(self) -> dict[str, Any]:
        self.stop()
        with self._lock:
            self.monitor.invalidate()
            self.router.reset_fault()
            self._session_sent_base = self.router.status()["sent"]
            self.state, self.error = "IDLE", None
            return self.status()

    def _watch(self) -> None:
        while not self._stop.wait(0.05):
            with self._lock:
                if self.state != "RUNNING":
                    continue
                report = self.sonic.latest()
                router = self.router.status()
                reason = None
                if report is None or not report.fresh() or report.mode != "STREAMED_MOTION":
                    reason = "SONIC mode report lost or returned to planner"
                elif self.monitor.frame(max_age_s=0.5) is None or self.monitor.state(max_age_s=0.5) is None:
                    reason = "robot camera or state became stale"
                elif router.get("error"):
                    reason = f"action router: {router['error']}"
                elif router["last_time"] is None or time.time() - router["last_time"] > 0.3:
                    reason = "model action stream stopped"
                if reason:
                    self._halt(reason)

    def status(self) -> dict[str, Any]:
        report = self.sonic.latest()
        router = self.router.status()
        return {
            "session": self.state, "error": self.error,
            "selected_model": self.model_id, "selected_task": self.task_id,
            "sonic": {
                "mode": report.mode if report and report.fresh() else "UNKNOWN",
                "report_age_s": None if report is None else time.monotonic() - report.received_at,
                "valid_token_age_ms": None if report is None else report.valid_token_age_ms,
            },
            "camera": self.monitor.status()["camera"],
            "state": self.monitor.status()["state"],
            "action": {
                "sent_this_session": router["sent"] - self._session_sent_base,
                "last_age_s": None if router["last_time"] is None else time.time() - router["last_time"],
                "error": router.get("error"),
            },
        }
