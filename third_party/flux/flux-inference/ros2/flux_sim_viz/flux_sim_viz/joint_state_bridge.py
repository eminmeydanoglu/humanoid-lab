"""Measured joint state for RViz, read from the simulator's Unitree topics.

Subscribes to the same three topics the robot's own node reads (`/lowstate` and
the two Dex3 hand states) and republishes them as one `sensor_msgs/JointState`
in URDF joint names, at a fixed rate.  Measured q in radians is passed through
unchanged: this node drives nothing and commands nothing.

An input that has never arrived, or that has stopped arriving, is left out of
the message and said out loud -- never replaced by a default or by the last
sample under a fresh timestamp, because `robot_state_publisher` would broadcast
either one as if it were measured.
"""

import math
import time

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
        # An input silent for this long is an outage, not a slow sample: the
        # bridge stops publishing it rather than presenting stale values as the
        # current robot state.
        self.declare_parameter("state_timeout_s", 2.0)
        self.declare_parameter("warning_interval_s", 10.0)

        self._body_topic = str(self.get_parameter("body_topic").value)
        self._hand_topics = {
            "left": str(self.get_parameter("left_hand_topic").value),
            "right": str(self.get_parameter("right_hand_topic").value),
        }
        self._timeout = float(self.get_parameter("state_timeout_s").value)
        self._warning_interval = float(self.get_parameter("warning_interval_s").value)
        if self._timeout <= 0.0 or self._warning_interval <= 0.0:
            raise ValueError("state_timeout_s and warning_interval_s must be positive")

        self._body_q: list[float] | None = None
        self._hand_q: dict[str, list[float]] = {}
        self._body_rx: float | None = None
        self._hand_rx: dict[str, float | None] = {"left": None, "right": None}
        #: Outages already reported, by key, with the time of the last warning.
        self._reported: dict[str, float] = {}

        self.create_subscription(LowState, self._body_topic, self._on_body, 10)
        self.create_subscription(
            HandState,
            self._hand_topics["left"],
            lambda msg: self._on_hand("left", msg),
            10,
        )
        self.create_subscription(
            HandState,
            self._hand_topics["right"],
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
            "joint state bridge: %s + %s + %s -> %s at %.0f Hz"
            % (
                self._body_topic,
                self._hand_topics["left"],
                self._hand_topics["right"],
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
        self._body_rx = time.monotonic()

    def _on_hand(self, side: str, msg: HandState) -> None:
        values = [float(motor.q) for motor in msg.motor_state[:HAND_MOTOR_SLOTS]]
        if len(values) != HAND_MOTOR_SLOTS or not all(math.isfinite(v) for v in values):
            self.get_logger().warning(
                f"ignoring {side} hand state with {len(msg.motor_state)} motor slots",
                once=True,
            )
            return
        self._hand_q[side] = values
        self._hand_rx[side] = time.monotonic()

    def _report(self, key: str, message: str) -> None:
        """Warn about an outage at most once per interval; log recovery once."""
        now = time.monotonic()
        last = self._reported.get(key)
        if last is None or now - last >= self._warning_interval:
            self._reported[key] = now
            self.get_logger().warning(message)

    def _clear(self, key: str) -> None:
        if key in self._reported:
            del self._reported[key]
            self.get_logger().info(f"{key} state resumed")

    def _fresh_inputs(self, now: float) -> tuple[bool, dict[str, list[float]]]:
        """Body freshness plus the hand sides whose state is current."""
        hands: dict[str, list[float]] = {}
        for side in ("left", "right"):
            topic = self._hand_topics[side]
            received = self._hand_rx[side]
            if received is None:
                self._report(
                    side,
                    f"no hand state on {topic}; TF for the 7 {side} hand joints is missing",
                )
                continue
            hand_age = now - received
            if hand_age > self._timeout:
                self._report(
                    side,
                    f"hand state on {topic} stopped {hand_age:.1f}s ago; TF for the "
                    f"7 {side} hand joints is missing",
                )
                continue
            self._clear(side)
            hands[side] = self._hand_q[side]

        if self._body_q is None:
            self._report(
                "body",
                f"no /lowstate on {self._body_topic}; /joint_states is not published, "
                "so TF for the 29 body joints is missing",
            )
            return False, {}
        age = now - self._body_rx
        if age > self._timeout:
            self._report(
                "body",
                f"/lowstate on {self._body_topic} stopped {age:.1f}s ago "
                f"(timeout {self._timeout:.1f}s); publishing stopped so stale joint TF "
                "is not presented as live",
            )
            return False, {}
        self._clear("body")
        return True, hands

    def _publish(self) -> None:
        body_ok, hands = self._fresh_inputs(time.monotonic())
        if not body_ok:
            return
        names, positions = measured_joint_state(
            self._body_q, hands.get("left"), hands.get("right")
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
        # The signal that ended spin shuts the context down and can land again
        # during teardown; that is a normal stop, not a crash worth a traceback.
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass


if __name__ == "__main__":
    main()
