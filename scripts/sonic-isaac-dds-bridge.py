#!/usr/bin/env python3
"""Publish Isaac state to SONIC over the Unitree sim DDS contract.

This process runs in the SONIC simulation environment (``sonic-sim``), which is
the only place CycloneDDS and ``unitree_sdk2py`` are installed.  Isaac Sim
bundles a different DDS implementation, so the runner speaks to this bridge over
a loopback TCP stream instead of loading a second DDS stack in-process.

Topics are exactly the simulation contract:

    Isaac -> SONIC   rt/lowstate, rt/odostate, rt/secondary_imu
    SONIC -> Isaac   rt/lowcmd

The bridge refuses to open DDS unless the domain and interface are the
loopback-only simulation profile, so it can never join a physical robot's
domain by accident.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from sonic_isaac_contract import (  # noqa: E402
    BODY_JOINT_COUNT,
    SIM_DDS_DOMAIN_ID,
    SIM_DDS_INTERFACE,
    ContractError,
)
from sonic_isaac_ipc import (  # noqa: E402
    DEFAULT_IPC_PORT,
    FRAME_HEADER,
    LowCmdFrame,
    decode_frame,
    encode_cmd,
    pack_frame,
)

MAX_PAYLOAD = 1 << 20


def _recv_exact(connection: socket.socket, size: int, *, idle_timeout_s: float = 120.0):
    """Read exactly ``size`` bytes, tolerating idle gaps.

    Returns None on a real EOF or once the link has been idle beyond the
    timeout. A short socket timeout must not be treated as a closed link, or
    the bridge would exit during a pause in the runner's publishing.
    """
    chunks = bytearray()
    deadline = time.monotonic() + float(idle_timeout_s)
    while len(chunks) < size:
        try:
            block = connection.recv(size - len(chunks))
        except (TimeoutError, socket.timeout):
            if time.monotonic() >= deadline:
                return None
            continue
        if not block:
            return None
        chunks.extend(block)
        deadline = time.monotonic() + float(idle_timeout_s)
    return bytes(chunks)


class UnitreeDds:
    """Thin adapter over the Unitree SDK2 publisher/subscriber set."""

    def __init__(self, *, domain_id: int, interface: str, hand_joint_count: int = 0) -> None:
        if int(domain_id) != SIM_DDS_DOMAIN_ID:
            raise ContractError(f"bridge refuses DDS domain {domain_id}; expected {SIM_DDS_DOMAIN_ID}")
        if str(interface) != SIM_DDS_INTERFACE:
            raise ContractError(f"bridge refuses DDS interface {interface!r}; expected {SIM_DDS_INTERFACE!r}")

        from unitree_sdk2py.core.channel import (  # noqa: PLC0415
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (  # noqa: PLC0415
            unitree_hg_msg_dds__IMUState_ as IMUState_default,
            unitree_hg_msg_dds__LowCmd_ as LowCmd_default,
            unitree_hg_msg_dds__LowState_ as LowState_default,
            unitree_hg_msg_dds__OdoState_ as OdoState_default,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # noqa: PLC0415
            IMUState_,
            LowCmd_,
            LowState_,
            OdoState_,
        )

        ChannelFactoryInitialize(int(domain_id), str(interface))

        self._lock = threading.Lock()
        self._latest_cmd = None
        self._received = 0
        self._sequence = 0

        self.low_state = LowState_default()
        self.odo_state = OdoState_default()
        self.torso_imu_state = IMUState_default()

        self.low_state_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_publisher.Init()
        self.odo_state_publisher = ChannelPublisher("rt/odostate", OdoState_)
        self.odo_state_publisher.Init()
        self.torso_imu_publisher = ChannelPublisher("rt/secondary_imu", IMUState_)
        self.torso_imu_publisher.Init()

        self.low_cmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_subscriber.Init(self._on_low_cmd, 1)

        self.hand_joint_count = int(hand_joint_count)
        self._hand_publishers = []
        if self.hand_joint_count:
            from unitree_sdk2py.idl.default import (  # noqa: PLC0415
                unitree_hg_msg_dds__HandState_ as HandState_default,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_  # noqa: PLC0415

            for side in ("left", "right"):
                publisher = ChannelPublisher(f"rt/dex3/{side}/state", HandState_)
                publisher.Init()
                self._hand_publishers.append((publisher, HandState_default()))

    # -- DDS callbacks ----------------------------------------------------- #

    def _on_low_cmd(self, msg) -> None:
        with self._lock:
            self._latest_cmd = msg
            self._received += 1

    @property
    def received(self) -> int:
        with self._lock:
            return self._received

    def take_cmd(self) -> LowCmdFrame | None:
        """Return the newest lowcmd converted to the body contract, or None."""
        with self._lock:
            msg = self._latest_cmd
            if msg is None:
                return None
            self._sequence += 1
            sequence = self._sequence
            self._latest_cmd = None

        def column(field: str) -> tuple[float, ...]:
            return tuple(float(getattr(msg.motor_cmd[i], field)) for i in range(BODY_JOINT_COUNT))

        return LowCmdFrame(
            sequence=sequence,
            q=column("q"),
            dq=column("dq"),
            tau=column("tau"),
            kp=column("kp"),
            kd=column("kd"),
        )

    # -- publishing -------------------------------------------------------- #

    def publish(self, state, *, hand_q: tuple[float, ...] = ()) -> None:
        for index in range(BODY_JOINT_COUNT):
            self.low_state.motor_state[index].q = state.body_q[index]
            self.low_state.motor_state[index].dq = state.body_dq[index]
            self.low_state.motor_state[index].ddq = state.body_ddq[index]
            self.low_state.motor_state[index].tau_est = state.body_tau_est[index]

        # Quaternions are w, x, y, z on the wire.
        self.odo_state.position[:] = state.root_pos
        self.odo_state.linear_velocity[:] = state.root_lin_vel
        self.odo_state.orientation[:] = state.root_quat_wxyz
        self.odo_state.angular_velocity[:] = state.root_ang_vel

        self.low_state.imu_state.quaternion[:] = state.root_quat_wxyz
        self.low_state.imu_state.gyroscope[:] = state.root_ang_vel
        self.low_state.imu_state.accelerometer[:] = state.root_acc

        self.torso_imu_state.quaternion[:] = state.torso_quat_wxyz
        self.torso_imu_state.gyroscope[:] = state.torso_gyro

        self.low_state.tick = int(state.tick_us // 1000) & 0xFFFFFFFF
        self.odo_state.tick = self.low_state.tick

        self.low_state_publisher.Write(self.low_state)
        self.odo_state_publisher.Write(self.odo_state)
        self.torso_imu_publisher.Write(self.torso_imu_state)

        if self._hand_publishers and hand_q:
            for publisher, message in self._hand_publishers:
                for index in range(min(self.hand_joint_count, len(hand_q))):
                    message.motor_state[index].q = hand_q[index]
                publisher.Write(message)


def serve(*, host: str, port: int, dds, report_path: Path | None = None,
          connect_timeout_s: float = 120.0) -> int:
    """Connect to the runner's loopback link and pump frames until it closes.

    The runner is the listener: it starts first so it can publish the initial
    pose while the scene is still paused. The bridge is therefore a client.
    """
    deadline = time.monotonic() + float(connect_timeout_s)
    connection = None
    while connection is None:
        try:
            connection = socket.create_connection((host, port), timeout=5.0)
        except OSError:
            if time.monotonic() >= deadline:
                print(
                    json.dumps({"event": "bridge_connect_failed", "host": host, "port": port}),
                    file=sys.stderr, flush=True,
                )
                return 2
            time.sleep(0.5)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(json.dumps({"event": "bridge_connected", "host": host, "port": port}), flush=True)

    published = 0
    forwarded = 0
    started = time.monotonic()
    status = "closed"
    try:
        while True:
            header = _recv_exact(connection, FRAME_HEADER.size)
            if header is None:
                status = "runner_closed"
                break
            (length,) = FRAME_HEADER.unpack(header)
            if length <= 0 or length > MAX_PAYLOAD:
                status = "bad_frame_length"
                break
            payload = _recv_exact(connection, length)
            if payload is None:
                status = "runner_closed"
                break
            frame = decode_frame(payload)
            if frame is None:
                status = "corrupt_frame"
                break
            dds.publish(frame)
            published += 1

            command = dds.take_cmd()
            if command is not None:
                connection.sendall(pack_frame(encode_cmd(command)))
                forwarded += 1
    finally:
        connection.close()

    summary = {
        "event": "bridge_stopped",
        "status": status,
        "state_frames_published": published,
        "lowcmd_forwarded": forwarded,
        "lowcmd_received": dds.received,
        "seconds": time.monotonic() - started,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if status == "runner_closed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_IPC_PORT)
    parser.add_argument("--domain-id", type=int, default=SIM_DDS_DOMAIN_ID)
    parser.add_argument("--interface", default=SIM_DDS_INTERFACE)
    parser.add_argument("--hand-joint-count", type=int, default=0)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()

    try:
        dds = UnitreeDds(
            domain_id=args.domain_id,
            interface=args.interface,
            hand_joint_count=args.hand_joint_count,
        )
    except ContractError as exc:
        print(json.dumps({"event": "bridge_refused", "reason": str(exc)}), file=sys.stderr, flush=True)
        return 2
    return serve(host=args.host, port=args.port, dds=dds, report_path=args.report_path)


if __name__ == "__main__":
    raise SystemExit(main())
