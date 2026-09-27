"""Measured joint state for RViz, read from the simulator's Unitree topics.

Subscribes to the same three topics the robot's own node reads (`/lowstate` and
the two Dex3 hand states) and republishes them as one `sensor_msgs/JointState`
in URDF joint names, at a fixed rate.  Measured q in radians is passed through
unchanged: this node drives nothing and commands nothing.
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from unitree_hg.msg import HandState, LowState

from .joints import BODY_JOINT_ORDER, HAND_MOTOR_SLOTS, measured_joint_state


class JointStateBridge(Node):
    """One publisher fed by three measured inputs."""

    def __init__(self) -> None:
        super().__init__("flux_sim_joint_state_bridge")
        self.declare_parameter("body_topic", "/lowstate")
        self.declare_parameter("left_hand_topic", "/dex3/left/state")
        self.declare_parameter("right_hand_topic", "/dex3/right/state")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("publish_rate_hz", 50.0)

        self._body_q: list[float] | None = None
        self._hand_q: dict[str, list[float]] = {}

        self.create_subscription(
            LowState, self.get_parameter("body_topic").value, self._on_body, 10
        )
        self.create_subscription(
            HandState,
            self.get_parameter("left_hand_topic").value,
            lambda msg: self._on_hand("left", msg),
            10,
        )
        self.create_subscription(
            HandState,
            self.get_parameter("right_hand_topic").value,
            lambda msg: self._on_hand("right", msg),
            10,
        )
        self._publisher = self.create_publisher(
            JointState, self.get_parameter("joint_state_topic").value, 10
        )
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self.create_timer(1.0 / rate_hz, self._publish)
        self.get_logger().info(
            "joint state bridge: %s + %s/%s -> %s at %.0f Hz"
            % (
                self.get_parameter("body_topic").value,
                self.get_parameter("left_hand_topic").value,
                self.get_parameter("right_hand_topic").value,
                self.get_parameter("joint_state_topic").value,
                rate_hz,
            )
        )

    def _on_body(self, msg: LowState) -> None:
        values = [float(motor.q) for motor in msg.motor_state[: len(BODY_JOINT_ORDER)]]
        if len(values) != len(BODY_JOINT_ORDER) or not all(math.isfinite(v) for v in values):
            self.get_logger().warning(
                f"ignoring /lowstate with {len(msg.motor_state)} motor slots",
                once=True,
            )
            return
        self._body_q = values

    def _on_hand(self, side: str, msg: HandState) -> None:
        values = [float(motor.q) for motor in msg.motor_state[:HAND_MOTOR_SLOTS]]
        if len(values) != HAND_MOTOR_SLOTS or not all(math.isfinite(v) for v in values):
            self.get_logger().warning(
                f"ignoring {side} hand state with {len(msg.motor_state)} motor slots",
                once=True,
            )
            return
        self._hand_q[side] = values

    def _publish(self) -> None:
        if self._body_q is None:
            return
        names, positions = measured_joint_state(
            self._body_q, self._hand_q.get("left"), self._hand_q.get("right")
        )
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = names
        message.position = positions
        self._publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = JointStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
