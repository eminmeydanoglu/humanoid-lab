"""Node-level tests for the measured joint-state bridge.

Runs in the flux-ros container (rclpy + unitree_hg): synthetic state publishers
feed the real node and a ROS subscriber records what it publishes.  The
missing-input cases are the contract, not a preference: when state is absent or
has stopped, nothing is published and the log says which topic is silent --
never a message whose values were invented or refreshed under a new stamp.
"""

import time

import pytest
import rclpy
from flux_sim_viz.joint_state_bridge import JointStateBridge
from flux_sim_viz.joints import BODY_JOINT_ORDER, hand_joint_names
from rclpy.node import Node
from sensor_msgs.msg import JointState
from unitree_hg.msg import HandState, LowState, MotorState

BODY_TOPIC = "/flux_sim_viz/test/lowstate"
LEFT_TOPIC = "/flux_sim_viz/test/dex3/left/state"
RIGHT_TOPIC = "/flux_sim_viz/test/dex3/right/state"
STATE_TOPIC = "/flux_sim_viz/test/joint_states"
ALL_NAMES = list(BODY_JOINT_ORDER) + list(hand_joint_names("left")) + list(
    hand_joint_names("right")
)
HAND_SLOTS = 7


def body_message(offset=0.0):
    message = LowState()
    for index, motor in enumerate(message.motor_state):
        motor.q = offset + index * 0.01
    return message


def hand_message(offset=0.0):
    message = HandState()
    message.motor_state = [MotorState() for _ in range(HAND_SLOTS)]
    for index, motor in enumerate(message.motor_state):
        motor.q = offset + index * 0.01
    return message


class StatePublisher(Node):
    def __init__(self):
        super().__init__("flux_sim_viz_test_state_publisher")
        self.body = self.create_publisher(LowState, BODY_TOPIC, 10)
        self.left = self.create_publisher(HandState, LEFT_TOPIC, 10)
        self.right = self.create_publisher(HandState, RIGHT_TOPIC, 10)


class Collector(Node):
    def __init__(self):
        super().__init__("flux_sim_viz_test_collector")
        self.messages = []
        self.create_subscription(JointState, STATE_TOPIC, self.messages.append, 10)


def _spin(nodes, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)


def _init(timeout_s):
    rclpy.init(
        args=[
            "--ros-args",
            "-p", "body_topic:=" + BODY_TOPIC,
            "-p", "left_hand_topic:=" + LEFT_TOPIC,
            "-p", "right_hand_topic:=" + RIGHT_TOPIC,
            "-p", "joint_state_topic:=" + STATE_TOPIC,
            "-p", f"state_timeout_s:={timeout_s}",
        ]
    )


def _logged(capfd):
    captured = capfd.readouterr()
    return captured.err + captured.out


def test_absent_state_publishes_nothing_and_says_which_topic(capfd):
    _init(0.3)
    bridge = JointStateBridge()
    collector = Collector()
    try:
        _spin([bridge, collector], 1.0)

        assert collector.messages == []
        log = _logged(capfd)
        assert "no /lowstate" in log
        assert BODY_TOPIC in log
        assert "TF for the 29 body joints is missing" in log
        assert "no hand state" in log
    finally:
        bridge.destroy_node()
        collector.destroy_node()
        rclpy.shutdown()


def test_measured_state_publishes_all_joints_and_stops_with_the_stream(capfd):
    _init(0.4)
    bridge = JointStateBridge()
    collector = Collector()
    publisher = StatePublisher()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            publisher.body.publish(body_message(0.5))
            publisher.left.publish(hand_message(0.1))
            publisher.right.publish(hand_message(0.2))
            _spin([bridge, collector], 0.02)

        assert collector.messages, "fresh state must be published"
        assert list(collector.messages[-1].name) == ALL_NAMES
        # q crosses the wire as float32, so compare with tolerance.
        assert list(collector.messages[-1].position)[: len(BODY_JOINT_ORDER)] == pytest.approx(
            [0.5 + index * 0.01 for index in range(len(BODY_JOINT_ORDER))]
        )

        # Let the timeout pass after the last sample, then require silence:
        # a stopped stream must not be republished as if it were current.
        _spin([bridge, collector], 1.0)
        count = len(collector.messages)
        _spin([bridge, collector], 0.5)

        assert len(collector.messages) == count
        log = _logged(capfd)
        assert "stopped" in log
        assert "publishing stopped" in log
    finally:
        bridge.destroy_node()
        collector.destroy_node()
        publisher.destroy_node()
        rclpy.shutdown()
