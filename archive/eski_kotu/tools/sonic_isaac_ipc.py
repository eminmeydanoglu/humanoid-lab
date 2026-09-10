"""Loopback framing between the Isaac runner and the DDS bridge process.

The runner has no DDS dependency: CycloneDDS is loaded only by the bridge, in
the SONIC simulation environment.  The two talk over a length-prefixed TCP
stream on loopback, carrying exactly the state SONIC needs and the lowcmd it
produces.

Both frame kinds are fixed-layout and self-describing so a truncated, corrupt
or mismatched frame is refused rather than partially applied.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import threading

from sonic_isaac_contract import BODY_JOINT_COUNT, DDS_MOTOR_ARRAY_SIZE

__all__ = [
    "CMD_MAGIC",
    "CMD_STRUCT",
    "CMD_VALUE_COUNT",
    "DEFAULT_IPC_PORT",
    "FRAME_HEADER",
    "HandStateFrame",
    "LowCmdFrame",
    "MAX_PAYLOAD",
    "StateFrame",
    "StateLink",
    "STATE_MAGIC",
    "STATE_STRUCT",
    "decode_cmd",
    "decode_frame",
    "decode_state",
    "encode_cmd",
    "encode_state",
    "pack_frame",
]

DEFAULT_IPC_PORT = 49052
MAX_PAYLOAD = 1 << 20

FRAME_HEADER = struct.Struct("!I")  # length prefix, payload bytes
STATE_MAGIC = b"SISA"
CMD_MAGIC = b"SICM"
_VERSION = 1

# tick + floating base (pos/quat/lin-vel/ang-vel/acc) + torso imu + body arrays
STATE_STRUCT = struct.Struct(
    "!4sBQ"
    "3f" "4f" "3f" "3f" "3f"
    "4f" "3f"
    + f"{BODY_JOINT_COUNT}f" * 4
)
assert STATE_STRUCT.size == 4 + 1 + 8 + (3 + 4 + 3 + 3 + 3) * 4 + 7 * 4 + BODY_JOINT_COUNT * 4 * 4

_CMD_FIELDS = BODY_JOINT_COUNT * 5
CMD_VALUE_COUNT = _CMD_FIELDS
CMD_STRUCT = struct.Struct("!4sBQ" + f"{_CMD_FIELDS}f")


@dataclass(frozen=True)
class StateFrame:
    tick_us: int
    root_pos: tuple[float, float, float]
    root_quat_wxyz: tuple[float, float, float, float]
    root_lin_vel: tuple[float, float, float]
    root_ang_vel: tuple[float, float, float]
    root_acc: tuple[float, float, float]
    torso_quat_wxyz: tuple[float, float, float, float]
    torso_gyro: tuple[float, float, float]
    body_q: tuple[float, ...]
    body_dq: tuple[float, ...]
    body_ddq: tuple[float, ...]
    body_tau_est: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("body_q", "body_dq", "body_ddq", "body_tau_est"):
            if len(getattr(self, name)) != BODY_JOINT_COUNT:
                raise ValueError(f"{name} must hold {BODY_JOINT_COUNT} values")


@dataclass(frozen=True)
class LowCmdFrame:
    sequence: int
    q: tuple[float, ...]
    dq: tuple[float, ...]
    tau: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("q", "dq", "tau", "kp", "kd"):
            if len(getattr(self, name)) != BODY_JOINT_COUNT:
                raise ValueError(f"{name} must hold {BODY_JOINT_COUNT} values")


def pack_frame(payload: bytes) -> bytes:
    return FRAME_HEADER.pack(len(payload)) + payload


def encode_state(frame: StateFrame) -> bytes:
    return STATE_STRUCT.pack(
        STATE_MAGIC,
        _VERSION,
        int(frame.tick_us),
        *frame.root_pos,
        *frame.root_quat_wxyz,
        *frame.root_lin_vel,
        *frame.root_ang_vel,
        *frame.root_acc,
        *frame.torso_quat_wxyz,
        *frame.torso_gyro,
        *frame.body_q,
        *frame.body_dq,
        *frame.body_ddq,
        *frame.body_tau_est,
    )


def encode_cmd(frame: LowCmdFrame) -> bytes:
    values: list[float] = []
    for name in ("q", "dq", "tau", "kp", "kd"):
        values.extend(getattr(frame, name))
    return CMD_STRUCT.pack(CMD_MAGIC, _VERSION, int(frame.sequence), *values)


def decode_frame(payload: bytes):
    """Decode a payload, or return ``None`` when it is not a valid frame."""
    if len(payload) < 5:
        return None
    magic = payload[:4]
    if magic == STATE_MAGIC:
        return decode_state(payload)
    if magic == CMD_MAGIC:
        return decode_cmd(payload)
    return None


def decode_state(payload: bytes) -> StateFrame | None:
    if len(payload) != STATE_STRUCT.size:
        return None
    (
        magic, version, tick_us, *values
    ) = STATE_STRUCT.unpack(payload)
    if magic != STATE_MAGIC or version != _VERSION:
        return None
    cursor = 0

    def take(count: int):
        nonlocal cursor
        chunk = tuple(values[cursor : cursor + count])
        cursor += count
        return chunk

    return StateFrame(
        tick_us=tick_us,
        root_pos=take(3),
        root_quat_wxyz=take(4),
        root_lin_vel=take(3),
        root_ang_vel=take(3),
        root_acc=take(3),
        torso_quat_wxyz=take(4),
        torso_gyro=take(3),
        body_q=take(BODY_JOINT_COUNT),
        body_dq=take(BODY_JOINT_COUNT),
        body_ddq=take(BODY_JOINT_COUNT),
        body_tau_est=take(BODY_JOINT_COUNT),
    )


def decode_cmd(payload: bytes) -> LowCmdFrame | None:
    if len(payload) != CMD_STRUCT.size:
        return None
    magic, version, sequence, *values = CMD_STRUCT.unpack(payload)
    if magic != CMD_MAGIC or version != _VERSION:
        return None
    stride = BODY_JOINT_COUNT
    groups = [
        tuple(values[offset * stride : (offset + 1) * stride]) for offset in range(5)
    ]
    return LowCmdFrame(sequence=sequence, q=groups[0], dq=groups[1], tau=groups[2],
                       kp=groups[3], kd=groups[4])


class HandStateFrame:
    """Hand joint state published for the *uncontrolled* fingers.

    The hands are never driven by SONIC body output; the runner reports them so
    the topics exist and the evidence can prove they stayed at the open target.
    """

    def __init__(self, joint_count: int, q: tuple[float, ...], dq: tuple[float, ...]) -> None:
        if len(q) != joint_count or len(dq) != joint_count:
            raise ValueError(f"hand state must hold {joint_count} joints")
        self.joint_count = joint_count
        self.q = q
        self.dq = dq


class StateLink:
    """Non-blocking loopback server the DDS bridge connects back to.

    The runner is the listener because it starts first: the scene must be
    paused and publishing the initial pose before SONIC is allowed to start.
    Dropped or unparseable command frames are counted, never applied.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_IPC_PORT) -> None:
        self.host = host
        self.port = port
        self._listener: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._buffer = bytearray()
        self._accepting = False
        self._lock = threading.Lock()
        self.state_frames_sent = 0
        self.command_frames_received = 0
        self.corrupt_frames = 0
        self.connection_refused = 0

    # -- lifecycle --------------------------------------------------------- #

    def open(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(1)
        listener.setblocking(False)
        # Record the port actually bound: port 0 asks the kernel to choose.
        self.port = int(listener.getsockname()[1])
        self._listener = listener
        self._accepting = True

    def close(self) -> None:
        self._accepting = False
        with self._lock:
            for stream in (self._connection, self._listener):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self._connection = None
            self._listener = None

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connection is not None

    def accept(self) -> bool:
        """Accept a pending bridge connection; returns True once connected."""
        with self._lock:
            if self._connection is not None or self._listener is None:
                return self._connection is not None
            try:
                connection, _ = self._listener.accept()
            except BlockingIOError:
                return False
            except OSError:
                return False
            connection.setblocking(False)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._connection = connection
            return True

    # -- publishing -------------------------------------------------------- #

    def publish(self, state: StateFrame) -> bool:
        payload = pack_frame(encode_state(state))
        with self._lock:
            connection = self._connection
            if connection is None:
                return False
            try:
                connection.sendall(payload)
            except OSError:
                self.corrupt_frames += 1
                return False
            self.state_frames_sent += 1
            return True

    # -- consuming --------------------------------------------------------- #

    def take_command(self) -> LowCmdFrame | None:
        """Newest complete, valid command frame, or None."""
        with self._lock:
            connection = self._connection
            if connection is None:
                return None
            while True:
                try:
                    block = connection.recv(65536)
                except BlockingIOError:
                    break
                except OSError:
                    break
                if not block:
                    self._connection = None
                    break
                self._buffer.extend(block)

            newest: LowCmdFrame | None = None
            while True:
                if len(self._buffer) < FRAME_HEADER.size:
                    break
                (length,) = FRAME_HEADER.unpack(bytes(self._buffer[: FRAME_HEADER.size]))
                if length <= 0 or length > MAX_PAYLOAD:
                    self.corrupt_frames += 1
                    self._buffer.clear()
                    break
                if len(self._buffer) < FRAME_HEADER.size + length:
                    break
                payload = bytes(self._buffer[FRAME_HEADER.size : FRAME_HEADER.size + length])
                del self._buffer[: FRAME_HEADER.size + length]
                frame = decode_frame(payload)
                if isinstance(frame, LowCmdFrame):
                    self.command_frames_received += 1
                    newest = frame
                else:
                    self.corrupt_frames += 1
            return newest

    def snapshot(self) -> dict:
        return {
            "connected": self.connected,
            "state_frames_sent": self.state_frames_sent,
            "command_frames_received": self.command_frames_received,
            "corrupt_frames": self.corrupt_frames,
        }
