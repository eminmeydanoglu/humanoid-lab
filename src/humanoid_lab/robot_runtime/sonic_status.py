"""Read the robot controller's own mode and valid-token age heartbeat."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

TOPIC = b"sonic_status "
MODES = {"PLANNER", "STREAMED_MOTION"}


@dataclass(frozen=True)
class SonicStatus:
    mode: str
    valid_token_age_ms: int | None
    sequence: int
    received_at: float

    def fresh(self, max_age_s: float = 0.5) -> bool:
        return time.monotonic() - self.received_at <= max_age_s


def decode_status(payload: bytes, *, received_at: float | None = None) -> SonicStatus:
    if not payload.startswith(TOPIC):
        raise ValueError("wrong SONIC status topic")
    try:
        data = json.loads(payload[len(TOPIC):])
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("malformed SONIC status JSON") from exc
    if not isinstance(data, dict) or data.get("mode") not in MODES:
        raise ValueError("invalid SONIC mode")
    age = data.get("valid_token_age_ms")
    if age is not None and (type(age) is not int or age < 0):
        raise ValueError("invalid valid-token age")
    sequence = data.get("sequence")
    if type(sequence) is not int or sequence < 1:
        raise ValueError("invalid SONIC status sequence")
    return SonicStatus(data["mode"], age, sequence,
                       time.monotonic() if received_at is None else received_at)


class SonicStatusSubscriber:
    def __init__(self, endpoint: str = "tcp://100.84.117.123:5561") -> None:
        import zmq

        self.endpoint = endpoint
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self.last: SonicStatus | None = None

    def poll(self) -> SonicStatus | None:
        import zmq

        try:
            payload = self.socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return self.last
        status = decode_status(payload)
        if self.last is None or status.sequence > self.last.sequence:
            self.last = status
        return self.last

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class SonicStatusMonitor:
    """Own the SUB socket in one thread; serve cached robot reports to the UI."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: SonicStatus | None = None
        self.error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="sonic-status", daemon=True)
        self._thread.start()

    def latest(self) -> SonicStatus | None:
        with self._lock:
            return self._last

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _serve(self) -> None:
        subscriber = SonicStatusSubscriber(self.endpoint)
        try:
            while not self._stop.is_set():
                try:
                    current = subscriber.poll()
                    if current is not None:
                        with self._lock:
                            self._last = current
                except ValueError as exc:
                    self.error = str(exc)
                self._stop.wait(0.02)
        finally:
            subscriber.close()
