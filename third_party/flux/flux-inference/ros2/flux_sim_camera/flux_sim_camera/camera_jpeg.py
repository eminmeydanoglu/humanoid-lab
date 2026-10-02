"""Latest-only raw ROS camera -> JPEG preview, independent of model input."""

import threading
import time
from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from flux_sim_camera.jpeg import encode_jpeg, validate_parameters

WARNING_INTERVAL_S = 10.0


class CameraJpeg(Node):
    def __init__(self) -> None:
        super().__init__("flux_sim_camera_jpeg")
        self.declare_parameter("raw_topic", "/camera/color/image_raw")
        self.declare_parameter("compressed_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("max_fps", 15.0)
        self.declare_parameter("stale_s", 0.5)
        self.declare_parameter("quality", 75)
        self.max_fps = float(self.get_parameter("max_fps").value)
        self.stale_s = float(self.get_parameter("stale_s").value)
        self.quality = self.get_parameter("quality").value
        validate_parameters(self.max_fps, self.stale_s, self.quality)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._lock = threading.Lock()
        self._latest = None
        self._last_warning_at = None
        self.publisher = self.create_publisher(
            CompressedImage, str(self.get_parameter("compressed_topic").value), qos
        )
        self.subscription = self.create_subscription(
            Image, str(self.get_parameter("raw_topic").value), self._on_image, qos
        )
        self.timer = self.create_timer(1.0 / self.max_fps, self._publish_latest)

    def _on_image(self, message: Image) -> None:
        with self._lock:
            self._latest = (message, time.monotonic())

    def _is_stale(self, message: Image, received_at: float) -> bool:
        stamp = message.header.stamp
        source_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        age_s = (self.get_clock().now().nanoseconds - source_ns) / 1e9
        return age_s > self.stale_s or time.monotonic() - received_at > self.stale_s

    def _warn(self, reason: str) -> None:
        now = time.monotonic()
        if self._last_warning_at is None or now - self._last_warning_at >= WARNING_INTERVAL_S:
            self.get_logger().warn("Camera JPEG frame rejected: %s" % reason)
            self._last_warning_at = now

    def _publish_latest(self) -> None:
        with self._lock:
            latest, self._latest = self._latest, None
        if latest is None:
            return
        message, received_at = latest
        try:
            if self._is_stale(message, received_at):
                return
            jpeg = encode_jpeg(message.data, message.width, message.height, message.step,
                               message.encoding, self.quality)
            # Encoding must not make a fresh frame stale before publication.
            if self._is_stale(message, received_at):
                return
            compressed = CompressedImage()
            compressed.header = deepcopy(message.header)
            compressed.format = "%s; jpeg compressed bgr8" % message.encoding
            compressed.data = jpeg
            self.publisher.publish(compressed)
        except Exception as exc:
            self._warn(str(exc))


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = CameraJpeg()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass
