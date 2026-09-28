"""Opt-in Flux Dex3 DDS bridge: the simulator as the fixed-base evaluation robot.

The Foxy ``flux_dex3`` node is a position-command client: it publishes arm
targets on ``/arm_sdk`` and one command per Dex3 hand, and reads ``/lowstate``
plus one state per hand.  It was written for the real robot and is not modified
for the simulator; this provider is the robot side of that contract:

* it publishes the three state topics the node's freshness gates read, and
* it turns the newest arm and hand commands into the simulator's joint command,
  applying the 14 arm joints and the two 7-joint hands **only**.

The legs are passive.  The three waist joints hold their zero-angle standing
pose with the pinned deployment's position and damping gains while arm commands
are fresh.  This keeps the torso aligned with the fixed pelvis under arm loads.
A stream that stops arriving expires through the standard ``command_ttl_s``
window and returns the whole robot to passive.

Arm and hand positions are clamped to the joint ranges of the pinned physical
model before they become the simulator's torque command.  That clamp is the
second half of a two-limit contract: the node's ``joint_limits_rad`` is the
*acceptance envelope* for the model's predicted targets, while physics applies
the *physical* range -- the simulated fingers cannot exceed the stroke the
shipped asset actually has.  The trained corpus commands the real fingers up to
20 degrees past the nominal model limits, so an accepted target can sit outside
the physical range; it is applied at the limit, explicitly, rather than being
left to PhysX to intercept.  The first clamp is reported as a
``flux_command_clipped`` event and counted in ``status()``.

Topics are raw Unitree DDS names (``rt/...``) because the simulator's DDS
stack is the same one the SONIC bridge uses.  A ROS 2 node publishing
``/arm_sdk`` under ``rmw_cyclonedds_cpp`` -- the Foxy robot's middleware -- is
the same DDS topic; the names are declared by the profile rather than assumed,
and a ROS-style ``/name`` here is refused instead of silently opening a topic
nothing publishes to.  The node's ``motor_cmd[29].q`` arm-SDK weight and every
``mode`` field are not interpreted: this simulator is the exclusive command
owner of the arms and hands, so there is no second controller to switch to.
"""

from __future__ import annotations

import json
import math
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts.commands import (
    BODY_COMMAND_SCHEMA,
    DEX3_COMMAND_SCHEMA,
    CommandError,
    CompleteRobotCommand,
    JointCommand,
)
from .base import ControllerInterface, RobotStateSample
from .sonic import (
    BODY_EFFORT_LIMIT_NM,
    BODY_JOINT_ORDER,
    DEFAULT_INTERFACE,
    deploy_gains,
    HAND_EFFORT_LIMIT_NM,
    RIGHT_HAND_EFFORT_LIMIT_NM,
    hand_joint_names,
)

READ_BATCH = 256

#: The arm block of the Unitree hardware order the Flux node commands
#: (``flux_dex3.mapping.ARM_MOTORS``): motors 15..28 of the 35-slot LowCmd.
#: ``BODY_JOINT_ORDER`` shares that hardware order, so the same indices select
#: the arm joints in the simulator's command vectors.
ARM_MOTOR_SLOTS: tuple[int, ...] = tuple(range(15, 29))
WAIST_MOTOR_SLOTS: tuple[int, ...] = tuple(range(12, 15))

#: Motor slots of one 35-slot LowCmd, checked instead of assumed: a different
#: layout on the wire would otherwise distribute arm targets into hand slots.
LOW_CMD_MOTOR_SLOTS = 35
DEX3_MOTOR_SLOTS = 7

#: Raw DDS name of each topic the Foxy node uses, mapped from its ROS 2 names
#: (``/arm_sdk``, ``/dex3/left/cmd``, ``/dex3/right/cmd``, ``/lowstate``,
#: ``/dex3/left/state``, ``/dex3/right/state``) by the ``rt/`` prefix every
#: ROS 2 middleware applies.  They are the same names the Unitree SDK examples
#: and the SONIC bridge use.
DEFAULT_TOPICS: dict[str, str] = {
    "arm_command": "rt/arm_sdk",
    "left_hand_command": "rt/dex3/left/cmd",
    "right_hand_command": "rt/dex3/right/cmd",
    "low_state": "rt/lowstate",
    "left_hand_state": "rt/dex3/left/state",
    "right_hand_state": "rt/dex3/right/state",
}

#: Machine variant the published LowState declares and the node echoes back in
#: its LowCmd.  Unitree's variant table gives ``g1_29dof_with_hand_rev_1_0`` --
#: the pinned asset's own name -- ``mode_machine`` 5, and the G1 29DoF teleop
#: config of the pinned stack declares the same value.  It is profile-overridable
#: because this simulator is the side that declares it.
DEFAULT_MODE_MACHINE = 5
DEFAULT_MODE_PR = 0

#: The pinned physical model the simulator's asset was built from.  The shipped
#: USD's own revolute limits were read back and equal these values on all 43
#: limited joints (arms and hands included), so this one file is what
#: "physical" means on both sides of the contract: the motor config's numbers
#: are derived from it, and the sim-side clamp below is applied with it.
PINNED_LIMITS_PATH = "configs/datasets/sonic/g1_joint_limits.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def pinned_position_limits() -> dict[str, tuple[float, float]]:
    """Position range of every joint of the pinned model, keyed by joint name."""
    path = REPO_ROOT / PINNED_LIMITS_PATH
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        joints = document["joints"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise CommandError(f"cannot read the pinned joint limits {path}: {error}") from error
    limits: dict[str, tuple[float, float]] = {}
    for name, entry in joints.items():
        try:
            lower, upper = float(entry["lower"]), float(entry["upper"])
        except (KeyError, TypeError, ValueError) as error:
            raise CommandError(f"pinned joint limits for {name!r} are malformed: {entry!r}") from error
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise CommandError(f"pinned joint limits for {name!r} are invalid: {entry!r}")
        limits[str(name)] = (lower, upper)
    return limits


def _limits_for(
    names: Sequence[str], limits: Mapping[str, tuple[float, float]]
) -> tuple[tuple[float, float], ...]:
    missing = sorted(set(names) - set(limits))
    if missing:
        raise CommandError(f"the pinned joint limits declare no range for {missing}")
    return tuple(limits[name] for name in names)


def clip_positions(
    values: Sequence[float],
    limits: Sequence[tuple[float, float]],
    names: Sequence[str],
) -> tuple[tuple[float, ...], list[tuple[str, float, float]]]:
    """Clamp commanded positions to their physical range.

    Returns the clamped positions and one ``(joint, requested, applied)`` entry
    for every value that moved, so a caller can report the clamp instead of
    letting it pass unnoticed.  Non-finite values are left untouched: they are
    refused by the command contract itself, in one place, rather than being
    silently turned into a limit value here.
    """
    if not (len(values) == len(limits) == len(names)):
        raise CommandError(
            f"clip needs one range and one name per value, got "
            f"{len(values)} values, {len(limits)} ranges, {len(names)} names"
        )
    clipped: list[tuple[str, float, float]] = []
    result: list[float] = []
    for name, value, (lower, upper) in zip(names, values, limits):
        value = float(value)
        applied = value if not math.isfinite(value) else min(max(value, lower), upper)
        if applied != value:
            clipped.append((name, value, applied))
        result.append(applied)
    return tuple(result), clipped


def _require_domain_id(config: Mapping[str, Any]) -> int:
    """The DDS domain, never defaulted: domain 0 is a physical G1's domain."""
    if "domain_id" not in config:
        raise CommandError("controller.domain_id is required for the flux_dds provider")
    try:
        domain_id = int(config["domain_id"])
    except (TypeError, ValueError) as error:
        raise CommandError(f"controller.domain_id is not an integer: {config['domain_id']!r}") from error
    if not 0 <= domain_id <= 232:
        raise CommandError(f"controller.domain_id {domain_id} is outside the DDS domain range")
    return domain_id


def _topic_map(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve the six raw DDS topic names, refusing ROS-style spelling."""
    override = config.get("topics") or {}
    if not isinstance(override, Mapping):
        raise CommandError("controller.topics must be an object mapping role to a raw DDS topic name")
    unknown = sorted(set(override) - set(DEFAULT_TOPICS))
    if unknown:
        raise CommandError(f"controller.topics names unknown roles: {unknown}")
    topics = dict(DEFAULT_TOPICS)
    for role, value in override.items():
        name = str(value)
        if not name or name.startswith("/") or any(character.isspace() for character in name):
            raise CommandError(
                f"controller.topics.{role} must be a raw DDS topic name such as "
                f"{DEFAULT_TOPICS[role]!r}, got {name!r}"
            )
        topics[role] = name
    return topics


def _mode(config: Mapping[str, Any], key: str, default: int) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError) as error:
        raise CommandError(f"controller.{key} is not an integer: {config.get(key)!r}") from error
    if not 0 <= value <= 255:
        raise CommandError(f"controller.{key} {value} does not fit the message field")
    return value


def body_command_vectors(
    motor_cmd: Sequence[Any],
) -> tuple[tuple[float, ...], ...]:
    """The 29-joint ``q``/``dq``/``tau``/``kp``/``kd`` vectors for one LowCmd.

    Arm targets come from the command slots the Flux node writes (15..28).
    The waist holds its zero-angle standing pose with deployment gains; the
    legs remain passive under the simulator's per-joint torque law.
    """
    if len(motor_cmd) != LOW_CMD_MOTOR_SLOTS:
        raise CommandError(
            f"LowCmd carries {len(motor_cmd)} motor slots, expected {LOW_CMD_MOTOR_SLOTS}"
        )
    q = [0.0] * len(BODY_JOINT_ORDER)
    dq = [0.0] * len(BODY_JOINT_ORDER)
    tau = [0.0] * len(BODY_JOINT_ORDER)
    kp = [0.0] * len(BODY_JOINT_ORDER)
    kd = [0.0] * len(BODY_JOINT_ORDER)
    deployed_kp, deployed_kd = deploy_gains()
    for slot in WAIST_MOTOR_SLOTS:
        kp[slot] = deployed_kp[slot]
        kd[slot] = deployed_kd[slot]
    for slot in ARM_MOTOR_SLOTS:
        motor = motor_cmd[slot]
        q[slot] = float(motor.q)
        dq[slot] = float(motor.dq)
        tau[slot] = float(motor.tau)
        kp[slot] = float(motor.kp)
        kd[slot] = float(motor.kd)
    return tuple(q), tuple(dq), tuple(tau), tuple(kp), tuple(kd)


def hand_command_vectors(motor_cmd: Sequence[Any]) -> tuple[tuple[float, ...], ...]:
    """The 7-joint ``q``/``dq``/``tau``/``kp``/``kd`` vectors for one hand command."""
    if len(motor_cmd) != DEX3_MOTOR_SLOTS:
        raise CommandError(
            f"hand command carries {len(motor_cmd)} motor slots, expected {DEX3_MOTOR_SLOTS}"
        )
    q = tuple(float(motor.q) for motor in motor_cmd)
    dq = tuple(float(motor.dq) for motor in motor_cmd)
    tau = tuple(float(motor.tau) for motor in motor_cmd)
    kp = tuple(float(motor.kp) for motor in motor_cmd)
    kd = tuple(float(motor.kd) for motor in motor_cmd)
    return q, dq, tau, kp, kd


def interface(config: Mapping[str, Any]) -> ControllerInterface:
    """The joint order and effort limits the Flux deployment speaks.

    The body vocabulary is the full 29-joint hardware order the asset is built
    from, not only the commanded arm block: the simulator resolves the declared
    order against the articulation once, and the command vectors above decide
    which of those joints receive a drive.
    """
    topics = _topic_map(config)
    return ControllerInterface(
        body_joint_names=BODY_JOINT_ORDER,
        body_effort_limits_nm=BODY_EFFORT_LIMIT_NM,
        hand_kind="dex3",
        left_hand_joint_names=hand_joint_names("left"),
        left_hand_effort_limits_nm=HAND_EFFORT_LIMIT_NM,
        right_hand_joint_names=hand_joint_names("right"),
        right_hand_effort_limits_nm=RIGHT_HAND_EFFORT_LIMIT_NM,
        notes={
            "domain_id": _require_domain_id(config),
            "interface": str(config.get("interface", DEFAULT_INTERFACE)),
            "topics": topics,
            "arm_motor_slots": [ARM_MOTOR_SLOTS[0], ARM_MOTOR_SLOTS[-1]],
            "mode_machine": _mode(config, "mode_machine", DEFAULT_MODE_MACHINE),
            "command_owner": "flux",
        },
    )


def _take_newest(reader: Any) -> Any:
    """Drain a topic and return its newest valid sample without ever blocking.

    ``take`` never waits, so a topic with no publisher costs nothing.  The
    SDK's ``Read`` helper is deliberately not used here: it waits forever when
    its timeout is zero, which would hang the simulation loop whenever the node
    is not running.
    """
    from cyclonedds.internal import InvalidSample

    samples = reader.take(READ_BATCH)
    valid = [sample for sample in samples if not isinstance(sample, InvalidSample)]
    return valid[-1] if valid else None


class FluxDdsController:
    """Apply the Foxy Flux node's arm and hand commands to the Isaac simulator."""

    kind = "flux_dds"
    # The node gates its own observations on freshness_s = 0.25 s, and the node
    # treats state older than that as a lost robot.  State is driven by its own
    # thread at the same steady cadence the Unitree robot publishes with, so the
    # simulation loop's jitter never reaches the node's gates.
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
            unitree_hg_msg_dds__LowState_ as LowStateDefault,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

        self.topics = _topic_map(config)
        limits = pinned_position_limits()
        self._body_limits = _limits_for(BODY_JOINT_ORDER, limits)
        self._hand_limits = {
            side: _limits_for(hand_joint_names(side), limits) for side in ("left", "right")
        }
        self._physics_dt = float(physics_dt)
        self._ttl_ticks = max(1, int(round(float(ttl_s) / self._physics_dt)))
        self._domain_id = _require_domain_id(config)
        self._interface = str(config.get("interface", DEFAULT_INTERFACE))
        self._mode_machine = _mode(config, "mode_machine", DEFAULT_MODE_MACHINE)
        self._mode_pr = _mode(config, "mode_pr", DEFAULT_MODE_PR)

        self._lock = threading.Lock()
        self._arm_command: Any = None
        self._left_hand_command: Any = None
        self._right_hand_command: Any = None
        self._episode_id = 0
        self._latest_tick = 0
        self._command_tick = -1
        self._sequence = 0
        self._received = 0
        self._applied = 0
        self._stale = 0
        self._clipped_polls = 0
        self._clipped_values = 0
        self._clip_report: dict[str, Any] | None = None
        self._gap_reported = False
        self._gap_report: dict[str, Any] | None = None

        self._state_message = LowStateDefault()
        self._state_message.mode_pr = self._mode_pr
        self._state_message.mode_machine = self._mode_machine
        self._left_hand_state = HandStateDefault()
        self._right_hand_state = HandStateDefault()

        ChannelFactoryInitialize(self._domain_id, self._interface)
        self._low_state_publisher = ChannelPublisher(self.topics["low_state"], LowState_)
        self._low_state_publisher.Init()
        self._left_hand_state_publisher = ChannelPublisher(self.topics["left_hand_state"], HandState_)
        self._left_hand_state_publisher.Init()
        self._right_hand_state_publisher = ChannelPublisher(self.topics["right_hand_state"], HandState_)
        self._right_hand_state_publisher.Init()

        # Commands are polled from the simulation loop rather than delivered to
        # an SDK callback thread.  A callback thread competes for the interpreter
        # lock with a Kit process that is already saturating it; polling keeps
        # delivery in the hands of the loop that needs the data.
        self._readers = self._open_command_readers()
        self._pending_state: RobotStateSample | None = None
        self._publisher_stop = threading.Event()
        self._published = 0
        self._publisher = threading.Thread(
            target=self._publish_loop, name="flux_dds_publisher", daemon=True
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
            "arm_command": DataReader(participant, Topic(participant, self.topics["arm_command"], LowCmd_)),
            "left_hand": DataReader(participant, Topic(participant, self.topics["left_hand_command"], HandCmd_)),
            "right_hand": DataReader(participant, Topic(participant, self.topics["right_hand_command"], HandCmd_)),
        }

    # ----------------------------------------------------------------- state

    def publish_state(self, state: RobotStateSample) -> None:
        """Hand the newest observation to the publisher thread."""
        with self._lock:
            if state.episode_id != self._episode_id:
                # A new episode invalidates every command produced under the old
                # one; nothing carries over into the reset simulation.
                self._episode_id = state.episode_id
                self._arm_command = None
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
            # The node reads q and, through the shared measured-state helper,
            # nothing else from these slots.  Joint acceleration and estimated
            # torque stay zero rather than being approximated.
            motor.ddq = 0.0
            motor.tau_est = 0.0
        message.imu_state.quaternion[:] = state.root_quaternion_wxyz
        message.imu_state.gyroscope[:] = state.root_angular_velocity_rps
        # The node does not consume the accelerometer; Isaac does not expose the
        # MuJoCo base acceleration the real robot forwards here.
        message.imu_state.accelerometer[:] = (0.0, 0.0, 0.0)
        message.tick = int(state.simulated_time_s * 1e3)
        self._low_state_publisher.Write(message)

        for index, value in enumerate(state.left_hand_q):
            self._left_hand_state.motor_state[index].q = float(value)
            self._left_hand_state.motor_state[index].dq = float(state.left_hand_dq[index])
        for index, value in enumerate(state.right_hand_q):
            self._right_hand_state.motor_state[index].q = float(value)
            self._right_hand_state.motor_state[index].dq = float(state.right_hand_dq[index])
        self._left_hand_state_publisher.Write(self._left_hand_state)
        self._right_hand_state_publisher.Write(self._right_hand_state)

    # -------------------------------------------------------------- commands

    def _receive_commands(self) -> None:
        """Take the newest command from each topic, if the node sent one."""
        arm = _take_newest(self._readers["arm_command"])
        left = _take_newest(self._readers["left_hand"])
        right = _take_newest(self._readers["right_hand"])
        with self._lock:
            if arm is not None:
                self._arm_command = arm
                self._received += 1
                self._command_tick = self._latest_tick
            if left is not None:
                self._left_hand_command = left
            if right is not None:
                self._right_hand_command = right

    def poll(self, tick: int) -> CompleteRobotCommand | None:
        self._receive_commands()
        with self._lock:
            arm_command = self._arm_command
            if arm_command is None or self._command_tick < 0:
                self._stale += 1
                return None
            received_tick = self._command_tick
            left_command = self._left_hand_command
            right_command = self._right_hand_command
            self._sequence += 1
            sequence = self._sequence
            self._maybe_report_gap(tick, received_tick)

        q, dq, tau, kp, kd = body_command_vectors(arm_command.motor_cmd)
        q, clipped = clip_positions(q, self._body_limits, BODY_JOINT_ORDER)
        body = JointCommand.build(
            schema=BODY_COMMAND_SCHEMA,
            sequence=sequence,
            joint_names=BODY_JOINT_ORDER,
            q=q,
            dq=dq,
            tau=tau,
            kp=kp,
            kd=kd,
            valid_until_tick=received_tick + self._ttl_ticks,
        )
        left_hand, left_clipped = self._hand_command(left_command, "left", sequence, received_tick)
        right_hand, right_clipped = self._hand_command(right_command, "right", sequence, received_tick)
        clipped = clipped + left_clipped + right_clipped
        if clipped:
            self._note_clip(clipped, tick)
        command = CompleteRobotCommand(
            episode_id=self._episode_id,
            body=body,
            left_hand=left_hand,
            right_hand=right_hand,
        )
        if command.is_valid_at(tick):
            self._applied += 1
        else:
            self._stale += 1
        return command

    def _hand_command(
        self, message: Any, side: str, sequence: int, received_tick: int
    ) -> tuple[JointCommand | None, list[tuple[str, float, float]]]:
        """One hand's command, or None for the profile's declared hand fallback.

        Positions are clamped to the physical range of that hand, and every
        clamped value is returned so the caller can report it.
        """
        if message is None:
            return None, []
        names = hand_joint_names(side)
        q, dq, tau, kp, kd = hand_command_vectors(message.motor_cmd)
        q, clipped = clip_positions(q, self._hand_limits[side], names)
        command = JointCommand.build(
            schema=DEX3_COMMAND_SCHEMA,
            sequence=sequence,
            joint_names=names,
            q=q,
            dq=dq,
            tau=tau,
            kp=kp,
            kd=kd,
            valid_until_tick=received_tick + self._ttl_ticks,
        )
        return command, clipped

    def _note_clip(self, clipped: Sequence[tuple[str, float, float]], tick: int) -> None:
        """Count a clamped command and announce the first one, once."""
        report = None
        with self._lock:
            self._clipped_polls += 1
            self._clipped_values += len(clipped)
            if self._clip_report is None:
                name, requested, applied = clipped[0]
                self._clip_report = {
                    "event": "flux_command_clipped",
                    "physics_tick": tick,
                    "joint": name,
                    "requested_rad": round(float(requested), 6),
                    "applied_rad": round(float(applied), 6),
                    "clipped_values": self._clipped_values,
                }
                report = self._clip_report
        if report is not None:
            print(json.dumps(report), flush=True)

    def _maybe_report_gap(self, tick: int, received_tick: int) -> None:
        """Announce once, and visibly, when a live command stream goes quiet."""
        gap = tick - received_tick
        gap_ticks = self._ttl_ticks + int(2.0 / self._physics_dt)
        if gap < gap_ticks or self._gap_reported:
            return
        self._gap_reported = True
        self._gap_report = {
            "event": "flux_command_gap",
            "physics_tick": tick,
            "age_ticks": gap,
            "commands_received": self._received,
            "commands_applied": self._applied,
            "polls": self._sequence,
            "last_command_tick": received_tick,
            "state_published": self._published,
        }
        print(json.dumps(self._gap_report), flush=True)

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
                "topics": dict(self.topics),
                "mode_machine": self._mode_machine,
                "commands_received": self._received,
                "commands_applied": applied,
                "stale_polls": self._stale,
                "clipped_polls": self._clipped_polls,
                "clipped_values": self._clipped_values,
                "clip": self._clip_report,
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
            self._arm_command = None
            self._left_hand_command = None
            self._right_hand_command = None
        for reader in self._readers.values():
            del reader
