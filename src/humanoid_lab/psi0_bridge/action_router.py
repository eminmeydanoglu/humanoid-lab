"""One persistent SONIC action output shared by PSI and GR00T."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any


class RouterError(RuntimeError):
    pass


class PsiSink:
    """Publisher-compatible PSI input whose socket is owned by ActionRouter."""

    def __init__(self, router: "ActionRouter") -> None:
        self.router = router
        self._halted = True

    @property
    def halted(self) -> bool:
        return self._halted

    def publish(self, payload: bytes) -> bool:
        if self._halted:
            return False
        return self.router.submit_psi(payload)

    def halt(self) -> None:
        self._halted = True

    def resume(self) -> None:
        self._halted = False

    def close(self) -> None:
        self._halted = True


class ActionRouter:
    """Forward only the selected policy source to SONIC's public action port."""

    def __init__(
        self,
        *,
        public_endpoint: str = "tcp://*:5556",
        groot_endpoint: str = "tcp://127.0.0.1:5560",
    ) -> None:
        self.public_endpoint = public_endpoint
        self.groot_endpoint = groot_endpoint
        self._lock = threading.Lock()
        self._source = "psi"
        self._gate = False
        self._closed = False
        self._error: str | None = None
        self._last_time: float | None = None
        self._sent = 0
        self._generation = 0
        self._psi: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=2)
        self._control: queue.Queue[bytes] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="action-router", daemon=True)
        self._thread.start()
        if not self._ready.wait(3.0):
            raise RouterError("action router did not start")
        if self._error:
            raise RouterError(self._error)
        self.psi_sink = PsiSink(self)

    def select(self, source: str) -> None:
        if source not in {"psi", "groot"}:
            raise ValueError(f"unknown action source: {source}")
        with self._lock:
            self._source = source
            self._gate = False
            self._generation += 1
        self._drain_psi()

    def resume(self) -> None:
        with self._lock:
            if self._closed:
                raise RouterError("action router is closed")
            self._gate = True

    def halt(self) -> None:
        with self._lock:
            self._gate = False
            self._generation += 1
        self._drain_psi()

    def submit_psi(self, payload: bytes) -> bool:
        with self._lock:
            accepted = self._gate and self._source == "psi" and not self._closed
            generation = self._generation
        if not accepted:
            return False
        try:
            self._psi.put_nowait((generation, bytes(payload)))
        except queue.Full:
            try:
                self._psi.get_nowait()
            except queue.Empty:
                pass
            self._psi.put_nowait((generation, bytes(payload)))
        return True

    def send_control(self, payload: bytes) -> None:
        """Send a SONIC manager command even while policy forwarding is gated."""
        try:
            self._control.put_nowait(bytes(payload))
        except queue.Full:
            self._control.get_nowait()
            self._control.put_nowait(bytes(payload))

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "endpoint": self.public_endpoint,
                "source": self._source,
                "gate_open": self._gate,
                "last_time": self._last_time,
                "sent": self._sent,
                "error": self._error,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._gate = False
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _serve(self) -> None:
        import zmq

        context = zmq.Context()
        output = context.socket(zmq.PUB)
        groot = context.socket(zmq.SUB)
        output.setsockopt(zmq.LINGER, 0)
        groot.setsockopt(zmq.LINGER, 0)
        groot.setsockopt(zmq.SUBSCRIBE, b"")
        try:
            output.bind(self.public_endpoint)
            groot.connect(self.groot_endpoint)
        except zmq.ZMQError as exc:
            with self._lock:
                self._error = f"cannot start action router: {exc}"
            self._ready.set()
            output.close()
            groot.close()
            context.term()
            return
        self._ready.set()
        poller = zmq.Poller()
        poller.register(groot, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                events = dict(poller.poll(10))
                if groot in events:
                    payload = groot.recv()
                    if self._accept("groot"):
                        output.send(payload)
                        self._record()
                self._forward_control(output)
                self._forward_psi(output)
        finally:
            output.close()
            groot.close()
            context.term()

    def _forward_control(self, output: Any) -> None:
        while True:
            try:
                payload = self._control.get_nowait()
            except queue.Empty:
                return
            output.send(payload)

    def _forward_psi(self, output: Any) -> None:
        while True:
            try:
                generation, payload = self._psi.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                accepted = self._gate and self._source == "psi" and generation == self._generation
            if accepted:
                output.send(payload)
                self._record()

    def _accept(self, source: str) -> bool:
        with self._lock:
            return self._gate and self._source == source and not self._closed

    def _record(self) -> None:
        with self._lock:
            self._last_time = time.time()
            self._sent += 1

    def _drain_psi(self) -> None:
        while True:
            try:
                self._psi.get_nowait()
            except queue.Empty:
                return
