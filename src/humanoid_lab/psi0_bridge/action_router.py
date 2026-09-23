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
    """Forward only the selected policy source to SONIC's public action port.

    Four sources can reach that port: ``psi`` (this bridge's own client), the
    upstream VLA client's stream on ``groot``, ``warmstart`` (opt-in, the
    demonstration-token warm start) and ``initial_pose`` (opt-in, the VLA
    client's own initial-pose command republished over the settle).  The
    selection is the only switch, so a hand-off between them is one atomic
    generation change and the SONIC deployment is never reset by it.
    """

    #: Everything that may be selected; a source that is not selected has its
    #: messages dropped before they can reach the socket.
    SOURCES = ("psi", "groot", "warmstart", "initial_pose")

    def __init__(
        self,
        *,
        public_endpoint: str = "tcp://*:5556",
        groot_endpoint: str = "tcp://127.0.0.1:5560",
        telemetry: Any | None = None,
    ) -> None:
        self.public_endpoint = public_endpoint
        self.groot_endpoint = groot_endpoint
        #: Opt-in recorder of every message that reaches the SONIC action port,
        #: for both sources (see :mod:`humanoid_lab.psi0_bridge.telemetry`).
        self.telemetry = telemetry
        self._lock = threading.Lock()
        self._source = "psi"
        self._gate = False
        self._closed = False
        self._error: str | None = None
        self._telemetry_error: str | None = None
        self._last_time: float | None = None
        self._sent = 0
        self._generation = 0
        self._psi: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=2)
        self._warmstart: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=2)
        self._initial_pose: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=2)
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
        if source not in self.SOURCES:
            raise ValueError(f"unknown action source: {source}")
        with self._lock:
            self._source = source
            self._gate = False
            self._generation += 1
        self._drain_psi()
        self._drain_warmstart()
        self._drain_initial_pose()

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
        self._drain_warmstart()
        self._drain_initial_pose()

    def submit_psi(self, payload: bytes) -> bool:
        return self._submit(self._psi, "psi", payload)

    def submit_warmstart(self, payload: bytes) -> bool:
        """Offer one message of the opt-in demonstration-token warm start.

        Accepted only while ``warmstart`` is the selected source and the gate is
        open, so a warm-start publisher that is still running after ``Start``
        cannot put a token between the policy's own messages.
        """
        return self._submit(self._warmstart, "warmstart", payload)

    def submit_initial_pose(self, payload: bytes) -> bool:
        """Offer one message of the opt-in initial-pose command stream.

        Same gate as every other source: accepted only while ``initial_pose`` is
        selected, so a handshake that is still running when the policy starts
        cannot put an initial-pose token into the GR00T stream.
        """
        return self._submit(self._initial_pose, "initial_pose", payload)

    def _submit(self, queue_: "queue.Queue[tuple[int, bytes]]", source: str, payload: bytes) -> bool:
        with self._lock:
            accepted = self._gate and self._source == source and not self._closed
            generation = self._generation
        if not accepted:
            return False
        try:
            queue_.put_nowait((generation, bytes(payload)))
        except queue.Full:
            try:
                queue_.get_nowait()
            except queue.Empty:
                pass
            queue_.put_nowait((generation, bytes(payload)))
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
                "telemetry_error": self._telemetry_error,
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
                        self._record("groot", payload)
                self._forward_control(output)
                self._forward_psi(output)
                self._forward_warmstart(output)
                self._forward_initial_pose(output)
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
                self._record("psi", payload)

    def _forward_warmstart(self, output: Any) -> None:
        while True:
            try:
                generation, payload = self._warmstart.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                accepted = (
                    self._gate and self._source == "warmstart"
                    and generation == self._generation
                )
            if accepted:
                output.send(payload)
                self._record("warmstart", payload)

    def _forward_initial_pose(self, output: Any) -> None:
        while True:
            try:
                generation, payload = self._initial_pose.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                accepted = (
                    self._gate and self._source == "initial_pose"
                    and generation == self._generation
                )
            if accepted:
                output.send(payload)
                self._record("initial_pose", payload)

    def _accept(self, source: str) -> bool:
        with self._lock:
            return self._gate and self._source == source and not self._closed

    def _record(self, source: str, payload: bytes) -> None:
        with self._lock:
            self._last_time = time.time()
            self._sent += 1
        if self.telemetry is None:
            return
        # Recording is best effort and must never stop the action path: an
        # exception here once killed this router thread, which silently left a
        # GR00T episode with no commands at all.
        try:
            if source == "groot":
                # NVIDIA's VLA client owns this stream; the bridge can only
                # report the camera snapshot it holds next to the action.
                self.telemetry.sample_frame("groot")
            self.telemetry.applied_action(source, payload)
        except Exception as exc:  # noqa: BLE001 - the router outlives the recorder
            if self._telemetry_error is None:
                self._telemetry_error = f"{type(exc).__name__}: {exc}"
                print(f"[action-router] telemetry disabled: {self._telemetry_error}", flush=True)

    def _drain_psi(self) -> None:
        while True:
            try:
                self._psi.get_nowait()
            except queue.Empty:
                return

    def _drain_warmstart(self) -> None:
        while True:
            try:
                self._warmstart.get_nowait()
            except queue.Empty:
                return

    def _drain_initial_pose(self) -> None:
        while True:
            try:
                self._initial_pose.get_nowait()
            except queue.Empty:
                return
