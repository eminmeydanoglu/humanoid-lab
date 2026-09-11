"""Unitree DDS bridge: the simulator acting as the robot.

The official SONIC deployment never talks to a simulator directly.  It talks to
the Unitree DDS topics a G1 robot would expose, and the official MuJoCo loop
stands in for the robot.  This module is the same stand-in for Isaac: it
publishes robot state and applies the deployment's motor commands, so the
unmodified binary runs against Isaac exactly as it runs against MuJoCo.

Only the topics the deployment consumes are created.  Odometer and
wireless-controller topics belong to the gamepad input path, which this profile
does not use.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Mapping

from ..contracts.commands import (
    BODY_COMMAND_SCHEMA,
    DEX3_COMMAND_SCHEMA,
    CompleteRobotCommand,
    JointCommand,
)
from .base import ControllerInterface, RobotStateSample
from .sonic import (
    BODY_EFFORT_LIMIT_NM,
    BODY_JOINT_ORDER,
    HAND_EFFORT_LIMIT_NM,
    LEFT_HAND_COMMAND_TOPIC,
    LEFT_HAND_STATE_TOPIC,
    LOW_COMMAND_TOPIC,
    LOW_STATE_TOPIC,
    RIGHT_HAND_COMMAND_TOPIC,
    RIGHT_HAND_EFFORT_LIMIT_NM,
    RIGHT_HAND_STATE_TOPIC,
    SECONDARY_IMU_TOPIC,
    hand_joint_names,
)


READ_BATCH = 256


def interface(config: Mapping[str, Any]) -> ControllerInterface:
    """The joint order and effort limits the SONIC deployment speaks."""
    return ControllerInterface(
        body_joint_names=BODY_JOINT_ORDER,
        body_effort_limits_nm=BODY_EFFORT_LIMIT_NM,
        hand_kind="dex3",
        left_hand_joint_names=hand_joint_names("left"),
        left_hand_effort_limits_nm=HAND_EFFORT_LIMIT_NM,
        right_hand_joint_names=hand_joint_names("right"),
        right_hand_effort_limits_nm=RIGHT_HAND_EFFORT_LIMIT_NM,
        notes={
            "domain_id": int(config.get("domain_id", 0)),
            "interface": str(config.get("interface", "lo")),
        },
    )


def _take_newest(reader: Any) -> Any:
    """Drain a topic and return its newest valid sample without ever blocking.

    ``take`` never waits, so a topic with no publisher costs nothing.  The
    SDK's ``Read`` helper is deliberately not used here: it waits forever when
    its timeout is zero, which would hang the simulation loop whenever the
    deployment is not running.
    """
    from cyclonedds.internal import InvalidSample

    samples = reader.take(READ_BATCH)
    valid = [sample for sample in samples if not isinstance(sample, InvalidSample)]
    return valid[-1] if valid else None


class SonicDdsController:
    """Apply the pinned SONIC deployment's commands to the Isaac simulator."""

    kind = "sonic_dds"
    # The official MuJoCo loop publishes state every physics step. The
    # deployment treats state older than 500 ms as a lost robot and stops, so
    # the stream is driven by its own thread at that same steady rate rather
    # than from the simulation loop, whose jitter must not reach the robot side.
    PUBLISH_HZ = 200.0

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        physics_dt: float,
        ttl_s: float,
    ) -> None:
        # Imported lazily: the DDS bindings only exist in the runtime image, and
        # only this controller needs them.
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandState_ as HandStateDefault,
            unitree_hg_msg_dds__IMUState_ as ImuStateDefault,
            unitree_hg_msg_dds__LowState_ as LowStateDefault,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, IMUState_, LowState_

        self._physics_dt = float(physics_dt)
        self._ttl_ticks = max(1, int(round(float(ttl_s) / self._physics_dt)))
        self._domain_id = int(config.get("domain_id", 0))
        self._interface = str(config.get("interface", "lo"))
        self._body_joint_names = BODY_JOINT_ORDER
        self._left_hand_names = hand_joint_names("left")
        self._right_hand_names = hand_joint_names("right")

        self._lock = threading.Lock()
        self._low_command: Any = None
        self._left_hand_command: Any = None
        self._right_hand_command: Any = None
        self._episode_id = 0
        self._latest_tick = 0
        self._command_tick = -1
        self._sequence = 0
        self._received = 0
        self._applied = 0
        self._stale = 0
        self._gap_reported = False
        self._gap_report: dict[str, Any] | None = None

        self._state_message = LowStateDefault()
        self._imu_message = ImuStateDefault()
        self._left_hand_state = HandStateDefault()
        self._right_hand_state = HandStateDefault()

        ChannelFactoryInitialize(self._domain_id, self._interface)
        self._low_state_publisher = ChannelPublisher(LOW_STATE_TOPIC, LowState_)
        self._low_state_publisher.Init()
        self._imu_publisher = ChannelPublisher(SECONDARY_IMU_TOPIC, IMUState_)
        self._imu_publisher.Init()
        self._left_hand_state_publisher = ChannelPublisher(LEFT_HAND_STATE_TOPIC, HandState_)
        self._left_hand_state_publisher.Init()
        self._right_hand_state_publisher = ChannelPublisher(RIGHT_HAND_STATE_TOPIC, HandState_)
        self._right_hand_state_publisher.Init()

        # Commands are polled from the simulation loop rather than delivered to
        # an SDK callback thread. A callback thread competes for the interpreter
        # lock with a Kit process that is already saturating it, and was
        # observed to go quiet for seconds at a time; polling keeps delivery in
        # the hands of the loop that needs the data.
        self._readers = self._open_command_readers()
        self._pending_state: RobotStateSample | None = None
        self._publisher_stop = threading.Event()
        self._published = 0
        self._publisher = threading.Thread(
            target=self._publish_loop, name="sonic_dds_publisher", daemon=True
        )
        self._publisher.start()

    def _open_command_readers(self) -> dict[str, Any]:
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, LowCmd_

        # The SDK already created this domain with its loopback-only profile;
        # a second Domain object for the same id would be rejected, so the
        # reader only joins the existing one.
        participant = DomainParticipant(self._domain_id)
        self._reader_participant = participant
        return {
            "low_command": DataReader(participant, Topic(participant, LOW_COMMAND_TOPIC, LowCmd_)),
            "left_hand": DataReader(participant, Topic(participant, LEFT_HAND_COMMAND_TOPIC, HandCmd_)),
            "right_hand": DataReader(participant, Topic(participant, RIGHT_HAND_COMMAND_TOPIC, HandCmd_)),
        }

    # ----------------------------------------------------------------- state

    def publish_state(self, state: RobotStateSample) -> None:
        """Hand the newest observation to the publisher thread."""
        with self._lock:
            if state.episode_id != self._episode_id:
                # A new episode invalidates every command produced under the old
                # one; nothing carries over into the reset simulation.
                self._episode_id = state.episode_id
                self._low_command = None
                self._left_hand_command = None
                self._right_hand_command = None
                self._command_tick = -1
            self._latest_tick = state.physics_tick
            self._pending_state = state

    def _publish_loop(self) -> None:
        period = 1.0 / self.PUBLISH_HZ
        while not self._publisher_stop.is_set():
            started = time.monotonic()
            with self._lock:
                state = self._pending_state
            if state is not None:
                self._write_state(state)
                self._published += 1
            delay = period - (time.monotonic() - started)
            if delay > 0.0:
                self._publisher_stop.wait(delay)

    def _write_state(self, state: RobotStateSample) -> None:
        message = self._state_message
        for index, value in enumerate(state.body_q):
            motor = message.motor_state[index]
            motor.q = float(value)
            motor.dq = float(state.body_dq[index])
            # The deployment reads q, dq, motor temperature and motor status,
            # plus the IMU fields. The MuJoCo loop also publishes joint
            # acceleration and estimated torque, which the deployment never
            # consumes; they stay zero here rather than being approximated.
            motor.ddq = 0.0
            motor.tau_est = 0.0
        message.imu_state.quaternion[:] = state.root_quaternion_wxyz
        message.imu_state.gyroscope[:] = state.root_angular_velocity_rps
        # The MuJoCo loop forwards the MuJoCo base acceleration here. Isaac does
        # not expose that directly and the deployment does not consume it.
        message.imu_state.accelerometer[:] = (0.0, 0.0, 0.0)
        message.tick = int(state.simulated_time_s * 1e3)
        self._low_state_publisher.Write(message)

        imu = self._imu_message
        imu.quaternion[:] = state.torso_quaternion_wxyz
        imu.gyroscope[:] = state.torso_angular_velocity_rps
        self._imu_publisher.Write(imu)

        for index in range(len(state.left_hand_q)):
            self._left_hand_state.motor_state[index].q = float(state.left_hand_q[index])
            self._left_hand_state.motor_state[index].dq = float(state.left_hand_dq[index])
            self._right_hand_state.motor_state[index].q = float(state.right_hand_q[index])
            self._right_hand_state.motor_state[index].dq = float(state.right_hand_dq[index])
        self._left_hand_state_publisher.Write(self._left_hand_state)
        self._right_hand_state_publisher.Write(self._right_hand_state)

    # -------------------------------------------------------------- commands

    def _receive_commands(self) -> None:
        """Take the newest command from each topic, if the deployment sent one."""
        low_command = _take_newest(self._readers["low_command"])
        left = _take_newest(self._readers["left_hand"])
        right = _take_newest(self._readers["right_hand"])
        with self._lock:
            if low_command is not None:
                self._low_command = low_command
                self._received += 1
                self._command_tick = self._latest_tick
            if left is not None:
                self._left_hand_command = left
            if right is not None:
                self._right_hand_command = right

    def poll(self, tick: int) -> CompleteRobotCommand | None:
        self._receive_commands()
        with self._lock:
            low_command = self._low_command
            if low_command is None or self._command_tick < 0:
                self._stale += 1
                return None
            received_tick = self._command_tick
            left_command = self._left_hand_command
            right_command = self._right_hand_command
            self._sequence += 1
            sequence = self._sequence
            self._maybe_report_gap(tick, received_tick)

        size = len(self._body_joint_names)
        body = JointCommand.build(
            schema=BODY_COMMAND_SCHEMA,
            sequence=sequence,
            joint_names=self._body_joint_names,
            q=[low_command.motor_cmd[index].q for index in range(size)],
            dq=[low_command.motor_cmd[index].dq for index in range(size)],
            tau=[low_command.motor_cmd[index].tau for index in range(size)],
            kp=[low_command.motor_cmd[index].kp for index in range(size)],
            kd=[low_command.motor_cmd[index].kd for index in range(size)],
            valid_until_tick=received_tick + self._ttl_ticks,
        )
        command = CompleteRobotCommand(
            episode_id=self._episode_id,
            body=body,
            left_hand=self._hand_command(left_command, "left", sequence, received_tick),
            right_hand=self._hand_command(right_command, "right", sequence, received_tick),
        )
        if command.is_valid_at(tick):
            self._applied += 1
        else:
            self._stale += 1
        return command

    def _maybe_report_gap(self, tick: int, received_tick: int) -> None:
        """Announce once, and visibly, when a live command stream goes quiet."""
        gap = tick - received_tick
        gap_ticks = self._ttl_ticks + int(2.0 / self._physics_dt)
        if gap < gap_ticks or self._gap_reported:
            return
        self._gap_reported = True
        self._gap_report = {
            "event": "sonic_command_gap",
            "physics_tick": tick,
            "age_ticks": gap,
            "commands_received": self._received,
            "commands_applied": self._applied,
            "polls": self._sequence,
            "last_command_tick": received_tick,
            "state_published": self._published,
        }
        print(json.dumps(self._gap_report), flush=True)

    def _hand_command(
        self, message: Any, side: str, sequence: int, received_tick: int
    ) -> JointCommand | None:
        if message is None:
            return None
        names = self._left_hand_names if side == "left" else self._right_hand_names
        return JointCommand.build(
            schema=DEX3_COMMAND_SCHEMA,
            sequence=sequence,
            joint_names=names,
            q=[message.motor_cmd[index].q for index in range(len(names))],
            dq=[message.motor_cmd[index].dq for index in range(len(names))],
            tau=[message.motor_cmd[index].tau for index in range(len(names))],
            kp=[message.motor_cmd[index].kp for index in range(len(names))],
            kd=[message.motor_cmd[index].kd for index in range(len(names))],
            valid_until_tick=received_tick + self._ttl_ticks,
        )

    # ---------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        with self._lock:
            command_tick = self._command_tick
            latest_tick = self._latest_tick
            applied = self._applied
            status = {
                "kind": self.kind,
                "state": "controlled" if command_tick >= 0 and applied else "waiting",
                "domain_id": self._domain_id,
                "interface": self._interface,
                "commands_received": self._received,
                "commands_applied": applied,
                "stale_polls": self._stale,
                "last_sequence": self._sequence,
                "command_age_ticks": (latest_tick - command_tick) if command_tick >= 0 else None,
                "ttl_ticks": self._ttl_ticks,
                "state_published": self._published,
                "gap": self._gap_report,
            }
        return status

    def close(self) -> None:
        self._publisher_stop.set()
        self._publisher.join(timeout=1.0)
        with self._lock:
            self._low_command = None
            self._left_hand_command = None
            self._right_hand_command = None
        for reader in self._readers.values():
            del reader
