"""Focused contracts for the opt-in Flux Dex3 provider and its configuration.

Everything here runs without a simulator and without a DDS stack: the message
path is exercised through plain objects that carry the same fields as the
Unitree messages the runtime actually binds, and the motor config is checked
against the Flux node's own ``load_motor_config`` so the sim-side values cannot
drift away from the contract the node enforces.  Live endpoint interoperability
between the Foxy node and the simulator's raw DDS topics is deliberately not
claimed here; that check is the integration owner's wire probe.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/flux/flux-inference/ros2/flux_dex3"))

import flux_dex3.command_output as command_output  # noqa: E402

from humanoid_lab.controllers import flux_dds, sonic  # noqa: E402
from humanoid_lab.contracts.commands import CommandError  # noqa: E402
from humanoid_lab.simulators.isaac.contracts import RunProfile  # noqa: E402

PROFILES = ROOT / "configs" / "profiles"
MOTOR_CONFIG = ROOT / "configs" / "flux" / "flux-dex3-sim-motor-config.json"
JOINT_LIMITS = ROOT / "configs" / "datasets" / "sonic" / "g1_joint_limits.json"

FLUX_PROFILES = (
    "pick-gum-askida.json",
    "pick-apple-askida.json",
)

#: The declared raw-DDS contract, in the role names the provider uses.
EXPECTED_TOPICS = {
    "arm_command": "rt/arm_sdk",
    "left_hand_command": "rt/dex3/left/cmd",
    "right_hand_command": "rt/dex3/right/cmd",
    "low_state": "rt/lowstate",
    "left_hand_state": "rt/dex3/left/state",
    "right_hand_state": "rt/dex3/right/state",
}

#: Real model overshoots observed in the successful closed-loop diagnostic run.
REAL_MODEL_OVERSHOOT_RAD = {
    "right_hand_index_1_joint": -0.4410,
    "right_hand_middle_1_joint": -0.3286,
    "right_hand_index_0_joint": -0.3045,
    "right_hand_middle_0_joint": -0.2984,
    "left_hand_index_1_joint": 0.2872,
    "left_hand_middle_1_joint": 0.2088,
    "left_hand_middle_0_joint": 0.1437,
    "right_hand_thumb_0_joint": -1.1406,
}


# -- the Flux node's own message shapes, as plain objects --------------------


class FakeMotor:
    def __init__(self):
        self.mode = 0
        self.reserve = 0
        self.q = self.dq = self.tau = self.kp = self.kd = 0.0


class FakeMotorState(FakeMotor):
    def __init__(self):
        super().__init__()
        self.ddq = 0.0
        self.tau_est = 0.0


class FakeImu:
    def __init__(self):
        self.quaternion = [0.0] * 4
        self.gyroscope = [0.0] * 3
        self.accelerometer = [0.0] * 3


class FakeLowState:
    def __init__(self):
        self.version = [0, 0]
        self.mode_pr = self.mode_machine = self.tick = self.crc = 0
        self.imu_state = FakeImu()
        self.motor_state = [FakeMotorState() for _ in range(35)]
        self.reserve = [0] * 4


class FakeHandState:
    def __init__(self):
        self.motor_state = [FakeMotorState() for _ in range(7)]
        self.imu_state = FakeImu()


class FakeLowCmd:
    def __init__(self):
        self.mode_pr = self.mode_machine = self.crc = 0
        self.motor_cmd = [FakeMotor() for _ in range(35)]
        self.reserve = [0] * 4


class FakeHandCmd:
    def __init__(self, slots=7):
        self.motor_cmd = [FakeMotor() for _ in range(slots)]


def arm_low_command() -> FakeLowCmd:
    """A command with distinctive values in the 14 arm slots."""
    message = FakeLowCmd()
    message.mode_machine = 5
    for offset, slot in enumerate(flux_dds.ARM_MOTOR_SLOTS):
        motor = message.motor_cmd[slot]
        motor.q = 0.1 * (offset + 1)
        motor.dq = 0.2 * (offset + 1)
        motor.tau = 0.3 * (offset + 1)
        motor.kp = 10.0 + offset
        motor.kd = 1.0 + 0.1 * offset
    message.motor_cmd[29].q = 1.0
    return message


def hand_command(values) -> FakeHandCmd:
    """A hand command with the given seven positions and the configured gains."""
    message = FakeHandCmd()
    for index, value in enumerate(values):
        motor = message.motor_cmd[index]
        motor.q = float(value)
        motor.kp = 1.5
        motor.kd = 0.1
    return message


#: Seven positions per hand that sit inside every physical hand range, in the
#: training order of each side (left: thumb0/1/2, middle0/1, index0/1;
#: right: thumb0/1/2, index0/1, middle0/1).
LEFT_INSIDE = (0.0, -0.3, 0.5, -0.4, -0.6, -0.5, -0.8)
RIGHT_INSIDE = (0.3, 0.2, -0.5, 0.4, 0.6, 0.5, 0.8)


def robot_state(episode_id: int = 0, tick: int = 100):
    from humanoid_lab.controllers.base import RobotStateSample

    return RobotStateSample(
        episode_id=episode_id,
        physics_tick=tick,
        simulated_time_s=tick * 0.005,
        root_position_m=(0.0, 0.0, 0.792563),
        root_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        root_linear_velocity_mps=(0.0, 0.0, 0.0),
        root_angular_velocity_rps=(0.0, 0.0, 0.0),
        body_q=tuple(0.01 * index for index in range(29)),
        body_dq=tuple(0.001 * index for index in range(29)),
        body_tau=(0.0,) * 29,
        torso_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        torso_angular_velocity_rps=(0.0, 0.0, 0.0),
        left_hand_q=tuple(0.1 * index for index in range(7)),
        left_hand_dq=(0.0,) * 7,
        right_hand_q=tuple(0.2 * index for index in range(7)),
        right_hand_dq=(0.0,) * 7,
    )


class FakePublisher:
    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.messages = []
        self.initialized = 0

    def Init(self):
        self.initialized += 1

    def Write(self, message):
        self.messages.append(message)


class FakeReader:
    def __init__(self, name):
        self.name = name
        self.queue = []
        self.reads = 0

    def take(self, batch):
        self.reads += 1
        samples = self.queue[:batch]
        del self.queue[:batch]
        return samples


class FakeDds:
    """The SDK surface the provider touches, in-process and assertable."""

    def __init__(self):
        self.publishers = {}
        self.readers = {}
        self.initialized = None
        self.domains = []
        self._modules = {}

    def _publisher(self, topic, message_type):
        publisher = FakePublisher(topic, message_type)
        self.publishers[topic] = publisher
        return publisher

    def _channel_module(self):
        channel = type(sys)("unitree_sdk2py.core.channel")
        channel.ChannelFactoryInitialize = self._initialize
        channel.ChannelConfigHasInterface = "<CycloneDDS><Domain id='any'></Domain></CycloneDDS>"
        channel.ChannelPublisher = self._publisher
        return channel

    def _initialize(self, domain_id, interface):
        self.initialized = (domain_id, interface)

    def _domain_participant(self, domain_id):
        self.domains.append(domain_id)
        return object()

    def _reader(self, participant, topic):
        reader = FakeReader(topic.name)
        self.readers[topic.name] = reader
        return reader

    def _topic(self, participant, name, message_type):
        return type("TopicStub", (), {"name": name, "message_type": message_type})()

    def __enter__(self):
        def module(name, **attributes):
            created = type(sys)(name)
            for key, value in attributes.items():
                setattr(created, key, value)
            return created

        self._modules = {
            "unitree_sdk2py.core.channel": self._channel_module(),
            "unitree_sdk2py.idl.default": module(
                "unitree_sdk2py.idl.default",
                unitree_hg_msg_dds__HandState_=FakeHandState,
                unitree_hg_msg_dds__LowState_=FakeLowState,
            ),
            "unitree_sdk2py.idl.unitree_hg.msg.dds_": module(
                "unitree_sdk2py.idl.unitree_hg.msg.dds_",
                LowCmd_=FakeLowCmd,
                HandCmd_=FakeHandCmd,
                LowState_=FakeLowState,
                HandState_=FakeHandState,
            ),
            "cyclonedds.domain": module("cyclonedds.domain", DomainParticipant=self._domain_participant),
            "cyclonedds.sub": module("cyclonedds.sub", DataReader=self._reader),
            "cyclonedds.topic": module("cyclonedds.topic", Topic=self._topic),
            "cyclonedds.internal": module("cyclonedds.internal", InvalidSample=type("InvalidSample", (), {})),
        }
        self._patch = unittest.mock.patch.dict(sys.modules, self._modules)
        self._patch.start()
        return self

    def __exit__(self, *exc_info):
        self._patch.stop()
        return False


def controller_config(**overrides):
    config = {
        "provider": "flux_dds",
        "domain_id": 42,
        "interface": "lo",
        "mode_machine": 5,
        "torso_link": "torso_link",
        "command_ttl_s": 0.25,
        "hand_fallback": "passive",
    }
    config.update(overrides)
    return config


class MotorConfigTests(unittest.TestCase):
    """The JSON the node loads; every value re-derived from its own evidence."""

    def setUp(self) -> None:
        self.config = command_output.load_motor_config(MOTOR_CONFIG)

    def test_arm_gains_are_the_deployment_gains_for_slots_15_to_28(self) -> None:
        kp, kd = sonic.deploy_gains()
        self.assertEqual(self.config["arm_kp"], [kp[index] for index in range(15, 29)])
        self.assertEqual(self.config["arm_kd"], [kd[index] for index in range(15, 29)])
        # The arm block is not one uniform family: wrist pitch/yaw use the 4010
        # armature, so a single copied value would be wrong for four joints.
        self.assertNotEqual(self.config["arm_kp"][0], self.config["arm_kp"][5])

    def test_arm_limits_are_the_strict_pinned_model_limits(self) -> None:
        pinned = json.loads(JOINT_LIMITS.read_text())["joints"]
        arm_names = list(sonic.BODY_JOINT_ORDER[15:29])
        self.assertEqual(arm_names[0], "left_shoulder_pitch_joint")
        self.assertEqual(arm_names[13], "right_wrist_yaw_joint")
        self.assertEqual(
            self.config["joint_limits_rad"][:14],
            [[pinned[name]["lower"], pinned[name]["upper"]] for name in arm_names],
        )

    def test_hand_limits_match_the_pinned_physical_range_and_order(self) -> None:
        pinned = json.loads(JOINT_LIMITS.read_text())["joints"]
        hand_names = list(sonic.hand_joint_names("left")) + list(sonic.hand_joint_names("right"))
        self.assertEqual(
            self.config["joint_limits_rad"][14:],
            [[pinned[name]["lower"], pinned[name]["upper"]] for name in hand_names],
        )

    def test_hand_gains_mode_and_bound_are_the_declared_values(self) -> None:
        self.assertEqual(self.config["hand_kp"], [1.5] * 14)
        self.assertEqual(self.config["hand_kd"], [0.1] * 14)
        self.assertIs(self.config["hand_timeout_enabled"], True)
        self.assertEqual(self.config["arm_mode_machine"], 5)
        self.assertEqual(self.config["max_tracking_error_rad"], 2.5)

    def test_simulation_config_scope_is_documented(self) -> None:
        readme = (ROOT / "configs" / "flux" / "README.md").read_text()
        self.assertIn("simulation settings do not establish hardware", readme)
        self.assertIn("Use the robot-specific", readme)


class PhysicalLimitGateTests(unittest.TestCase):
    """The node schedules out-of-range predictions at physical joint limits."""

    TRAINING_ORDER = (
        list(sonic.BODY_JOINT_ORDER[15:29])
        + list(sonic.hand_joint_names("left"))
        + list(sonic.hand_joint_names("right"))
    )

    def setUp(self) -> None:
        self.limits = command_output.load_motor_config(MOTOR_CONFIG)["joint_limits_rad"]

    def test_real_model_overshoots_are_clipped_and_task_continues(self) -> None:
        import numpy as np
        from flux_dex3.executor import ChunkExecutor

        executor = ChunkExecutor(limits=self.limits)
        executor.start("session")
        actions = np.zeros((32, 28), dtype=np.float32)
        for joint, value in REAL_MODEL_OVERSHOOT_RAD.items():
            actions[:, self.TRAINING_ORDER.index(joint)] = value
        executor.accept("session", 0, 0.1, actions, now=10.0)
        first = executor.tick(now=10.0)
        for joint, value in REAL_MODEL_OVERSHOOT_RAD.items():
            index = self.TRAINING_ORDER.index(joint)
            lower, upper = self.limits[index]
            self.assertAlmostEqual(first[index], min(max(value, lower), upper), places=6)
        self.assertEqual(executor.session, "session")
        self.assertEqual(executor.last_seq, 0)
        self.assertAlmostEqual(actions[0, self.TRAINING_ORDER.index("right_hand_index_1_joint")], -0.4410)
        executor.accept("session", 1, 0.1, actions * 10, now=10.2)
        self.assertIsNotNone(executor.next_chunk)
        self.assertTrue(np.all(executor.next_chunk >= np.asarray(self.limits)[:, 0]))
        self.assertTrue(np.all(executor.next_chunk <= np.asarray(self.limits)[:, 1]))

    def test_adapter_uses_the_same_physical_limits(self) -> None:
        pinned = flux_dds.pinned_position_limits()
        for name, pair in zip(self.TRAINING_ORDER, self.limits):
            self.assertEqual(pinned[name], tuple(pair))


class FluxProfileTests(unittest.TestCase):
    def test_flux_profiles_are_fixed_root_task_scenes(self) -> None:
        diagnostic = RunProfile.load(PROFILES / "isaac-g1-sonic-fixed-base-dex3.json")
        self.assertIsNone(diagnostic.scene)  # the no-table diagnostic this is not
        for name in FLUX_PROFILES:
            with self.subTest(profile=name):
                profile = RunProfile.load(PROFILES / name)
                self.assertTrue(profile.robot.fixed_base)
                self.assertTrue(profile.robot.disable_gravity)
                self.assertIsNone(profile.support)
                self.assertEqual(profile.initial_pose, "sonic_standing")
                expected_x = 0.08 if name == "pick-apple-askida.json" else 0.0
                self.assertEqual(profile.robot.initial_position_m, (expected_x, 0.0, sonic.STANDING_ROOT_HEIGHT_M))
                scene = profile.scene
                self.assertIsNotNone(scene)
                self.assertEqual(scene.target.kind, "plate")
                self.assertIsNotNone(scene.object)
                self.assertTrue(scene.camera_enabled)
                controller = profile.controller
                self.assertIsNotNone(controller)
                self.assertEqual(controller["provider"], "flux_dds")
                self.assertEqual(controller["domain_id"], 42)
                self.assertEqual(controller["interface"], "lo")
                self.assertEqual(controller["mode_machine"], 5)
                self.assertEqual(controller["command_ttl_s"], 0.25)
                self.assertEqual(controller["hand_fallback"], "passive")

    def test_apple_spawn_faces_the_table(self) -> None:
        apple = RunProfile.load(PROFILES / "pick-apple-askida.json")
        self.assertEqual(apple.robot.initial_position_m[:2], (0.08, 0.0))
        self.assertEqual(apple.robot.initial_rotation_wxyz, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(apple.robot.initial_joint_positions_rad, {
            "left_elbow_joint": -0.3222726578,
            "right_elbow_joint": -0.3222726578,
        })
        self.assertEqual(apple.camera,
                         RunProfile.load(PROFILES / "isaac-g1-sonic-pickapple-dex3.json").camera)

    def test_flux_profiles_declare_the_raw_dds_topic_contract(self) -> None:
        for name in FLUX_PROFILES:
            with self.subTest(profile=name):
                profile = RunProfile.load(PROFILES / name)
                self.assertEqual(profile.controller["topics"], EXPECTED_TOPICS)
                interface = flux_dds.interface(profile.controller)
                self.assertEqual(interface.notes["topics"], EXPECTED_TOPICS)
                self.assertEqual(interface.notes["domain_id"], 42)
                self.assertEqual(interface.notes["command_owner"], "flux")

    def test_the_profiles_keep_the_inherited_task_table(self) -> None:
        for name, base in (
            ("pick-gum-askida.json", "isaac-g1-sonic-pickgum-dex3.json"),
            ("pick-apple-askida.json", "isaac-g1-sonic-pickapple-dex3.json"),
        ):
            with self.subTest(profile=name):
                flux = RunProfile.load(PROFILES / name)
                inherited = RunProfile.load(PROFILES / base)
                self.assertEqual(flux.scene.table, inherited.scene.table)
                self.assertEqual(flux.scene.target, inherited.scene.target)
                self.assertEqual(flux.scene.object, inherited.scene.object)
                self.assertEqual(flux.camera, inherited.camera)

    def test_a_ros_style_topic_name_is_refused(self) -> None:
        with self.assertRaisesRegex(CommandError, "raw DDS topic name"):
            flux_dds.interface(controller_config(topics={"arm_command": "/arm_sdk"}))
        with self.assertRaisesRegex(CommandError, "unknown roles"):
            flux_dds.interface(controller_config(topics={"front_camera": "rt/camera"}))
        with self.assertRaisesRegex(CommandError, "must be an object"):
            flux_dds.interface(controller_config(topics="rt/arm_sdk"))

    def test_the_domain_is_never_defaulted(self) -> None:
        with self.assertRaisesRegex(CommandError, "domain_id is required"):
            flux_dds.interface({"provider": "flux_dds", "torso_link": "torso_link"})
        with self.assertRaisesRegex(CommandError, "outside the DDS domain range"):
            flux_dds.interface(controller_config(domain_id=999))
        with self.assertRaisesRegex(CommandError, "does not fit the message field"):
            flux_dds.interface(controller_config(mode_machine=300))


class CommandVectorTests(unittest.TestCase):
    def test_arm_targets_come_from_slots_15_to_28_only(self) -> None:
        message = arm_low_command()
        # A slot outside the arm block, written by some other source: the
        # provider must not pick it up.
        message.motor_cmd[0].q = 99.0
        message.motor_cmd[0].kp = 99.0
        q, dq, tau, kp, kd = flux_dds.body_command_vectors(message.motor_cmd)
        self.assertEqual(len(q), 29)
        for offset, slot in enumerate(flux_dds.ARM_MOTOR_SLOTS):
            self.assertAlmostEqual(q[slot], 0.1 * (offset + 1))
            self.assertAlmostEqual(dq[slot], 0.2 * (offset + 1))
            self.assertAlmostEqual(tau[slot], 0.3 * (offset + 1))
            self.assertAlmostEqual(kp[slot], 10.0 + offset)
            self.assertAlmostEqual(kd[slot], 1.0 + 0.1 * offset)
        for slot in range(12):
            self.assertEqual((q[slot], dq[slot], tau[slot], kp[slot], kd[slot]), (0.0,) * 5)
        deployed_kp, deployed_kd = sonic.deploy_gains()
        for slot in flux_dds.WAIST_MOTOR_SLOTS:
            self.assertEqual((q[slot], dq[slot], tau[slot]), (0.0,) * 3)
            self.assertEqual((kp[slot], kd[slot]), (deployed_kp[slot], deployed_kd[slot]))
        # The command vectors are the 29 body joints: the arm-SDK weight slot
        # (29) and the unused tail of the 35-slot LowCmd are not joints.
        for values in (q, dq, tau, kp, kd):
            self.assertEqual(len(values), len(sonic.BODY_JOINT_ORDER))

    def test_legs_are_passive_and_waist_holds_the_torso(self) -> None:
        q, dq, tau, kp, kd = flux_dds.body_command_vectors(arm_low_command().motor_cmd)
        measured = [0.7 + 0.11 * index for index in range(29)]
        measured_velocity = [-0.3 + 0.02 * index for index in range(29)]
        for slot in range(15):
            torque = tau[slot] + kp[slot] * (q[slot] - measured[slot]) + kd[slot] * (
                dq[slot] - measured_velocity[slot]
            )
            if slot in flux_dds.WAIST_MOTOR_SLOTS:
                self.assertLess(torque, 0.0)
            else:
                self.assertEqual(torque, 0.0)

    def test_hand_vectors_cover_exactly_seven_motors(self) -> None:
        q, dq, tau, kp, kd = flux_dds.hand_command_vectors(
            hand_command(tuple(0.5 + index for index in range(7))).motor_cmd
        )
        self.assertEqual(q, tuple(0.5 + index for index in range(7)))
        self.assertEqual(kp, (1.5,) * 7)
        self.assertEqual(kd, (0.1,) * 7)
        self.assertEqual((dq, tau), ((0.0,) * 7, (0.0,) * 7))
        with self.assertRaisesRegex(CommandError, "expected 7"):
            flux_dds.hand_command_vectors(FakeHandCmd(slots=6).motor_cmd)
        with self.assertRaisesRegex(CommandError, "expected 35"):
            flux_dds.body_command_vectors(FakeLowCmd().motor_cmd[:34])


class ControllerTests(unittest.TestCase):
    """The provider end to end against the fake SDK surface."""

    def test_state_topics_are_the_declared_publishers_and_carry_the_slices(self) -> None:
        with FakeDds() as dds:
            controller = flux_dds.FluxDdsController(
                controller_config(), physics_dt=0.005, ttl_s=0.25
            )
            try:
                self.assertEqual(dds.initialized, (42, "lo"))
                self.assertEqual(
                    sorted(dds.publishers),
                    ["rt/dex3/left/state", "rt/dex3/right/state", "rt/lowstate"],
                )
                self.assertEqual(
                    sorted(dds.readers),
                    ["rt/arm_sdk", "rt/dex3/left/cmd", "rt/dex3/right/cmd"],
                )
                state = robot_state()
                controller.publish_state(state)
                controller._write_state(state)
                low = dds.publishers["rt/lowstate"].messages[-1]
                self.assertEqual(low.mode_machine, 5)
                self.assertEqual(low.motor_state[15].q, state.body_q[15])
                self.assertEqual(low.motor_state[28].dq, state.body_dq[28])
                self.assertEqual(low.imu_state.quaternion, list(state.root_quaternion_wxyz))
                self.assertEqual(low.tick, int(state.simulated_time_s * 1e3))
                left = dds.publishers["rt/dex3/left/state"].messages[-1]
                right = dds.publishers["rt/dex3/right/state"].messages[-1]
                self.assertEqual([motor.q for motor in left.motor_state], list(state.left_hand_q))
                self.assertEqual([motor.q for motor in right.motor_state], list(state.right_hand_q))
            finally:
                controller.close()

    def test_commands_apply_arms_and_hands_and_expire_through_the_ttl(self) -> None:
        with FakeDds() as dds:
            controller = flux_dds.FluxDdsController(
                controller_config(), physics_dt=0.005, ttl_s=0.25
            )
            try:
                controller.publish_state(robot_state(tick=100))
                # No command yet: the loop must stay passive.
                self.assertIsNone(controller.poll(100))

                dds.readers["rt/arm_sdk"].queue.append(arm_low_command())
                dds.readers["rt/dex3/left/cmd"].queue.append(hand_command(LEFT_INSIDE))
                dds.readers["rt/dex3/right/cmd"].queue.append(hand_command(RIGHT_INSIDE))
                command = controller.poll(101)
                self.assertIsNotNone(command)
                self.assertEqual(command.body.joint_names, sonic.BODY_JOINT_ORDER)
                self.assertEqual(command.body.q[15], 0.1)
                self.assertEqual(command.body.kp[28], 10.0 + 13)
                self.assertEqual(command.body.q[14], 0.0)  # waist holds the standing angle
                self.assertGreater(command.body.kp[14], 0.0)
                self.assertIsNotNone(command.left_hand)
                self.assertEqual(command.left_hand.q, LEFT_INSIDE)
                self.assertEqual(command.right_hand.q, RIGHT_INSIDE)
                self.assertTrue(command.is_valid_at(101))
                # ttl_s / physics_dt = 50 ticks after the receiving tick.
                self.assertTrue(command.is_valid_at(150))
                self.assertFalse(command.is_valid_at(151))
                self.assertEqual(controller.status()["state"], "controlled")

                # A new episode invalidates the command before any new one lands.
                controller.publish_state(robot_state(episode_id=1, tick=0))
                self.assertIsNone(controller.poll(0))
                status = controller.status()
                self.assertEqual(status["state"], "waiting")
                self.assertEqual(status["topics"], EXPECTED_TOPICS)
            finally:
                controller.close()

    def test_out_of_physical_hand_targets_are_clipped_and_reported(self) -> None:
        with FakeDds() as dds:
            controller = flux_dds.FluxDdsController(
                controller_config(), physics_dt=0.005, ttl_s=0.25
            )
            try:
                controller.publish_state(robot_state(tick=100))
                # Every commanded hand joint sits at zero (inside every physical
                # range) except slot 4, which is middle_1 on the left (training
                # order) and index_1 on the right: both are commanded past the
                # physical stroke.
                left = hand_command((0.0,) * 7)
                right = hand_command((0.0,) * 7)
                left.motor_cmd[4].q = -2.0
                right.motor_cmd[4].q = 2.0
                dds.readers["rt/arm_sdk"].queue.append(arm_low_command())
                dds.readers["rt/dex3/left/cmd"].queue.append(left)
                dds.readers["rt/dex3/right/cmd"].queue.append(right)

                command = controller.poll(101)
                pinned = flux_dds.pinned_position_limits()
                expected_left = [0.0] * 7
                expected_left[4] = pinned["left_hand_middle_1_joint"][0]
                expected_right = [0.0] * 7
                expected_right[4] = pinned["right_hand_index_1_joint"][1]
                self.assertEqual(list(command.left_hand.q), expected_left)
                self.assertEqual(list(command.right_hand.q), expected_right)
                self.assertEqual(command.left_hand.q[3], 0.0)
                self.assertEqual(command.right_hand.q[5], 0.0)
                status = controller.status()
                self.assertEqual(status["clipped_polls"], 1)
                self.assertEqual(status["clipped_values"], 2)
                self.assertEqual(status["clip"]["event"], "flux_command_clipped")
                self.assertIn(status["clip"]["joint"], {"left_hand_middle_1_joint", "right_hand_index_1_joint"})
                self.assertNotEqual(status["clip"]["requested_rad"], status["clip"]["applied_rad"])
            finally:
                controller.close()

    def test_arm_targets_outside_the_physical_range_are_clipped_too(self) -> None:
        with FakeDds() as dds:
            controller = flux_dds.FluxDdsController(
                controller_config(), physics_dt=0.005, ttl_s=0.25
            )
            try:
                controller.publish_state(robot_state(tick=100))
                arm = arm_low_command()
                arm.motor_cmd[15].q = -3.5   # left shoulder pitch, physical -3.0892
                arm.motor_cmd[28].q = 2.0    # right wrist yaw, physical 1.61443
                dds.readers["rt/arm_sdk"].queue.append(arm)
                command = controller.poll(101)
                pinned = flux_dds.pinned_position_limits()
                self.assertEqual(command.body.q[15], pinned["left_shoulder_pitch_joint"][0])
                self.assertEqual(command.body.q[28], pinned["right_wrist_yaw_joint"][1])
                self.assertEqual(controller.status()["clipped_values"], 2)
            finally:
                controller.close()

    def test_targets_inside_the_physical_range_are_not_flagged(self) -> None:
        with FakeDds() as dds:
            controller = flux_dds.FluxDdsController(
                controller_config(), physics_dt=0.005, ttl_s=0.25
            )
            try:
                controller.publish_state(robot_state(tick=100))
                arm = arm_low_command()
                for slot in flux_dds.ARM_MOTOR_SLOTS:
                    arm.motor_cmd[slot].q = 0.0
                left = hand_command((0.0,) * 7)
                right = hand_command((0.0,) * 7)
                dds.readers["rt/arm_sdk"].queue.append(arm)
                dds.readers["rt/dex3/left/cmd"].queue.append(left)
                dds.readers["rt/dex3/right/cmd"].queue.append(right)
                command = controller.poll(101)
                self.assertEqual(command.left_hand.q, (0.0,) * 7)
                self.assertEqual(command.right_hand.q, (0.0,) * 7)
                status = controller.status()
                self.assertEqual((status["clipped_polls"], status["clipped_values"]), (0, 0))
                self.assertIsNone(status["clip"])
            finally:
                controller.close()

    def test_non_finite_targets_are_refused_not_clamped(self) -> None:
        with FakeDds() as dds:
            controller = flux_dds.FluxDdsController(
                controller_config(), physics_dt=0.005, ttl_s=0.25
            )
            try:
                controller.publish_state(robot_state(tick=100))
                right = hand_command((0.0,) * 7)
                right.motor_cmd[3].q = float("nan")
                dds.readers["rt/arm_sdk"].queue.append(arm_low_command())
                dds.readers["rt/dex3/right/cmd"].queue.append(right)
                with self.assertRaisesRegex(CommandError, "finite"):
                    controller.poll(101)
                status = controller.status()
                self.assertEqual((status["clipped_polls"], status["clipped_values"]), (0, 0))
                self.assertEqual(status["commands_applied"], 0)
            finally:
                controller.close()

    def test_the_factory_builds_the_provider(self) -> None:
        from humanoid_lab.controllers.factory import PROVIDERS, build_controller

        self.assertIn("flux_dds", PROVIDERS)
        with FakeDds():
            controller, interface = build_controller(
                controller_config(), provider_override=None, physics_dt=0.005, ttl_s=0.25
            )
            try:
                self.assertIsInstance(controller, flux_dds.FluxDdsController)
                self.assertEqual(interface.body_joint_names, sonic.BODY_JOINT_ORDER)
                self.assertEqual(interface.body_effort_limits_nm, sonic.BODY_EFFORT_LIMIT_NM)
                self.assertEqual(interface.hand_kind, "dex3")
                self.assertEqual(interface.left_hand_joint_names, sonic.hand_joint_names("left"))
                self.assertEqual(interface.right_hand_joint_names, sonic.hand_joint_names("right"))
            finally:
                controller.close()
        # The existing provider registry is untouched by the addition.
        self.assertEqual(PROVIDERS[:4], ("none", "scripted", "sonic_dds", "flux_dds"))




class TestLoopbackDiscovery(unittest.TestCase):
    def test_sdk_uses_well_known_ports_and_restores_config(self):
        module = type(sys)("unitree_sdk2py.core.channel")
        original = "<CycloneDDS><Domain id='any'></Domain></CycloneDDS>"
        module.ChannelConfigHasInterface = original
        seen = []
        module.ChannelFactoryInitialize = lambda domain, interface: seen.append(
            (domain, interface, module.ChannelConfigHasInterface))
        with unittest.mock.patch.dict(sys.modules, {module.__name__: module}):
            flux_dds._initialize_sdk_channel(42, "lo")
        self.assertEqual(seen[0][:2], (42, "lo"))
        self.assertIn("<ParticipantIndex>auto</ParticipantIndex>", seen[0][2])
        self.assertIn("<MaxAutoParticipantIndex>99</MaxAutoParticipantIndex>", seen[0][2])
        self.assertIn('address="127.0.0.1"', seen[0][2])
        self.assertEqual(module.ChannelConfigHasInterface, original)

    def test_nonloopback_interface_retains_sdk_config(self):
        module = type(sys)("unitree_sdk2py.core.channel")
        module.ChannelConfigHasInterface = "unchanged"
        module.ChannelFactoryInitialize = unittest.mock.Mock()
        with unittest.mock.patch.dict(sys.modules, {module.__name__: module}):
            flux_dds._initialize_sdk_channel(42, "eth0")
        module.ChannelFactoryInitialize.assert_called_once_with(42, "eth0")
        self.assertEqual(module.ChannelConfigHasInterface, "unchanged")

    def test_config_restored_when_sdk_initialization_fails(self):
        module = type(sys)("unitree_sdk2py.core.channel")
        original = "<CycloneDDS><Domain id='any'></Domain></CycloneDDS>"
        module.ChannelConfigHasInterface = original
        module.ChannelFactoryInitialize = unittest.mock.Mock(side_effect=RuntimeError("DDS failed"))
        with unittest.mock.patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(RuntimeError, "DDS failed"):
                flux_dds._initialize_sdk_channel(42, "lo")
        self.assertEqual(module.ChannelConfigHasInterface, original)


if __name__ == "__main__":
    unittest.main()
