"""Foxy observation bridge; command publishing is deliberately disabled."""

import queue
import time
import uuid
from collections import deque

import numpy as np
import rclpy
from flux_dex3_interfaces.srv import GetStatus, StartTask
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from unitree_hg.msg import HandState, LowState

from flux_dex3.command_output import CommandOutput
from flux_dex3.executor import ChunkExecutor
from flux_dex3.mapping import measured_state, split_action
from flux_dex3.network import NetworkWorker

PROMPTS = frozenset((
    "stack three block", "camera packaging", "object placement", "pour water", "toasted bread",
    "Put the apple into the plate.", "Put the bottle into the plate.",
    "Put the charger into the plate.", "Put the doll into the plate.",
    "Put the gum into the plate.", "Put the snack into the plate.",
    "Put the tissue paper into the plate.",
))


def rgb_image(msg):
    if msg.encoding != "rgb8" or (msg.height, msg.width) not in ((480, 640), (192, 256)):
        raise ValueError("camera must provide RGB8 480x640 or 192x256")
    if msg.step < msg.width * 3 or len(msg.data) != msg.step * msg.height:
        raise ValueError("invalid camera byte layout")
    return np.ndarray((msg.height, msg.width, 3), dtype=np.uint8, buffer=bytes(msg.data),
                      strides=(msg.step, 3, 1)).copy(order="C")


class Dex3Node(Node):
    def __init__(self):
        super().__init__("flux_dex3")
        self.declare_parameter("endpoint", "tcp://127.0.0.1:5557")
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("network_timeout_s", 2.0)
        self.declare_parameter("freshness_s", 0.25)
        self.declare_parameter("pair_tolerance_s", 0.1)
        self.declare_parameter("client_certificate", "")
        self.declare_parameter("server_public_key", "")
        self.declare_parameter("enable_motor_commands", False)
        self.declare_parameter("motor_output_config", "")
        endpoint = self.get_parameter("endpoint").value
        certificate = self.get_parameter("client_certificate").value
        server = self.get_parameter("server_public_key").value
        public = secret = ""
        if certificate:
            from zmq.auth import load_certificate

            public_bytes, secret_bytes = load_certificate(certificate)
            if secret_bytes is None:
                raise ValueError("client certificate has no secret key")
            public, secret = public_bytes.decode("ascii"), secret_bytes.decode("ascii")
        if not endpoint.startswith("tcp://127.0.0.1:") and not all((public, secret, server)):
            raise ValueError("remote ZMQ requires CURVE client certificate and server key")
        self.freshness_s = float(self.get_parameter("freshness_s").value)
        self.pair_tolerance_s = float(self.get_parameter("pair_tolerance_s").value)
        if self.freshness_s <= 0 or self.pair_tolerance_s <= 0:
            raise ValueError("freshness_s and pair_tolerance_s must be positive")
        self.state_window_s = self.freshness_s + self.pair_tolerance_s
        self.latest = {}
        self.state_history = {key: deque() for key in ("low", "left", "right")}
        self.pair_offsets = {}
        self.events = queue.Queue()
        self.network = NetworkWorker(endpoint, self.events,
                                     float(self.get_parameter("network_timeout_s").value), public, secret, server)
        self.chunk_executor = ChunkExecutor()
        self.command_output = None
        if self.get_parameter("enable_motor_commands").value:
            config_path = self.get_parameter("motor_output_config").value
            if not config_path:
                raise ValueError("motor output requires a verified hardware config file")
            self.command_output = CommandOutput(self, config_path)
            self.chunk_executor.limits = self.command_output.config["joint_limits_rad"]
        self.server_status = "LOADING"
        self.checkpoint = ""
        self.reason = "waiting for GPU server"
        self.prompt = ""
        self.next_seq = 0
        self.inflight = False
        self.last_request_start = float("-inf")
        self.last_target = None
        self.last_action = None
        self.paused = False
        self.pause_target = None
        self.paused_session = None
        self.paused_request_pending = False
        self.hold_until_first_chunk = False
        self._last_status_log = None
        self._last_sensor_warning = None
        self._last_sensor_warning_at = 0.0
        self._last_heartbeat = 0.0
        self._last_chunk_start = None
        self._last_hold = None
        self.create_subscription(Image, self.get_parameter("camera_topic").value, self._image, 2)
        self.create_subscription(LowState, "/lowstate", lambda msg: self._store("low", msg), 10)
        self.create_subscription(HandState, "/dex3/left/state", lambda msg: self._store("left", msg), 10)
        self.create_subscription(HandState, "/dex3/right/state", lambda msg: self._store("right", msg), 10)
        self.create_service(StartTask, "~/start_task", self._start)
        self.create_service(Trigger, "~/pause_task", self._pause)
        self.create_service(Trigger, "~/stop_task", self._stop)
        self.create_service(GetStatus, "~/get_status", self._status)
        self.create_timer(1.0 / 30.0, self._tick)
        self.network.start()
        self.get_logger().info(
            "Dex3 node started: model=LOADING, endpoint=%s, camera=%s, command_publishing=%s"
            % (endpoint, self.get_parameter("camera_topic").value, self._output_mode())
        )
        self.get_logger().info("Waiting for GPU server readiness and fresh camera/joint observations")

    def _output_mode(self):
        return "enabled" if self.command_output is not None else "disabled"

    def _publish_target(self, action):
        if self.command_output is None:
            return
        now = time.monotonic()
        keys = ("low", "left", "right")
        if any(key not in self.latest or now - self.latest[key][1] > self.freshness_s for key in keys):
            raise ValueError("joint feedback unavailable/stale during command publication")
        measured = measured_state(*(self.latest[key][0] for key in keys))
        self.command_output.publish(action, measured)

    def _sensor_warning(self, reason):
        now = time.monotonic()
        if reason != self._last_sensor_warning or now - self._last_sensor_warning_at >= 10.0:
            self.get_logger().warn("Observation rejected: %s" % reason)
            self._last_sensor_warning, self._last_sensor_warning_at = reason, now

    def _abort(self, reason):
        session = self.chunk_executor.session
        self.chunk_executor.stop()
        self.inflight = False
        self.paused = False
        self.pause_target = None
        self.paused_session = None
        self.paused_request_pending = False
        self.hold_until_first_chunk = False
        self.last_action = None
        self.last_target = None
        self._last_hold = None
        self.reason = reason
        self.get_logger().error("Task output stopped: %s (session=%s); publishing ceased" % (reason, session))

    def _heartbeat(self):
        now = time.monotonic()
        if now - self._last_heartbeat < 10.0:
            return
        self._last_heartbeat = now
        ages = ", ".join(
            "%s=%s" % (key, "%.2fs" % (now - self.latest[key][1]) if key in self.latest else "missing")
            for key in ("image", "low", "left", "right")
        )
        self.get_logger().info(
            "STATUS model=%s task=%s request_in_flight=%s step=%s sensors=[%s] command_publishing=%s"
            % (self.server_status, "paused" if self.paused else "active" if self.chunk_executor.session else "idle",
               self.inflight, self.chunk_executor.index if self.chunk_executor.session else "-", ages,
               self._output_mode())
        )

    def _store(self, key, msg):
        now = time.monotonic()
        samples = self.state_history[key]
        if not samples:
            self.get_logger().info("First %s joint-state message received" % key)
        self.latest[key] = (msg, now)
        samples.append((now, msg))
        oldest = now - self.state_window_s
        while samples[0][0] < oldest:
            samples.popleft()

    def _image(self, msg):
        try:
            image = rgb_image(msg)
        except ValueError as exc:
            self.reason = str(exc)
            self._sensor_warning(self.reason)
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if stamp <= 0 or age < -0.05 or age > self.freshness_s:
            self.reason = "stale camera timestamp (age=%.3fs)" % age
            self._sensor_warning("stale camera timestamp")
            return
        if "image" not in self.latest:
            self.get_logger().info("First fresh RGB frame received: %dx%d" % (msg.width, msg.height))
        self.latest["image"] = (image, time.monotonic(), stamp)

    def _state_sample(self, key, frame):
        """Buffered sample whose arrival is closest to the frame instant, within the pairing tolerance."""
        samples = self.state_history[key]
        if not samples:
            return None
        arrival, msg = min(samples, key=lambda item: abs(item[0] - frame))
        offset = abs(arrival - frame)
        self.pair_offsets[key] = offset
        return msg if offset <= self.pair_tolerance_s else None

    def _snapshot(self):
        now = time.monotonic()
        if "image" not in self.latest or now - self.latest["image"][1] > self.freshness_s:
            raise ValueError("camera or joint state unavailable/stale")
        image, _, stamp = self.latest["image"]
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if age > self.freshness_s:
            raise ValueError("camera frame is stale")
        # The frame's capture instant on the clock the joint samples are recorded with.
        frame = now - age
        self.pair_offsets = {}
        samples = {key: self._state_sample(key, frame) for key in ("low", "left", "right")}
        missing = sorted(key for key, sample in samples.items() if sample is None)
        if missing:
            raise ValueError("no %s joint sample within %.3fs of the camera frame"
                             % ("/".join(missing), self.pair_tolerance_s))
        state = np.asarray(measured_state(*(samples[key] for key in ("low", "left", "right"))),
                           dtype=np.float32)
        return image, state, stamp

    def _start(self, request, response):
        response.accepted = False
        if self.chunk_executor.session is not None:
            response.reason = "task already running"
        elif self.server_status != "READY":
            response.reason = "model is not READY"
        elif request.prompt not in PROMPTS:
            response.reason = "prompt is absent from training manifest"
        elif self.paused and request.prompt != self.prompt:
            response.reason = "resume requires the paused task prompt; call StopTask to change tasks"
        elif self.paused and self.paused_request_pending:
            response.reason = "previous prediction still in flight; pause target remains active"
        else:
            try:
                self._snapshot()
            except ValueError as exc:
                response.reason = str(exc)
            else:
                resuming = self.paused
                self.chunk_executor.start(uuid.uuid4().hex)
                self.paused = False
                self.paused_session = None
                self.paused_request_pending = False
                self.hold_until_first_chunk = resuming
                if not resuming:
                    self.pause_target = None
                    self.last_action = None
                self.prompt = request.prompt
                self.next_seq = 0
                self.inflight = False
                self.last_request_start = float("-inf")
                self._request()
                response.accepted = self.chunk_executor.session is not None
                response.reason = (("resumed" if resuming else "started") +
                                   "; command publishing " + self._output_mode() if response.accepted else self.reason)
                if response.accepted:
                    self.get_logger().info(
                        "StartTask %s: session=%s prompt=%r; command_publishing=%s"
                        % ("resumed" if resuming else "accepted", self.chunk_executor.session,
                           self.prompt, self._output_mode())
                    )
        self.reason = response.reason
        if not response.accepted:
            self.get_logger().warn("StartTask rejected: %s" % response.reason)
        return response

    def _pause(self, request, response):
        if self.paused:
            response.success, response.message = True, "already paused; holding last target"
        elif self.chunk_executor.session is None or self.last_action is None:
            response.success, response.message = False, "no executed target available to hold"
        else:
            session = self.chunk_executor.session
            self.pause_target = self.last_action.copy()
            self.paused_session = session
            self.paused_request_pending = self.inflight
            self.chunk_executor.stop()
            self.inflight = False
            self.paused = True
            self.hold_until_first_chunk = False
            self.reason = "paused; repeating last target at 30 Hz; command publishing %s" % self._output_mode()
            response.success, response.message = True, self.reason
            self.get_logger().info(
                "PauseTask: session=%s last target latched; repeating at 30 Hz, command_publishing=%s"
                % (session, self._output_mode())
            )
        if not response.success:
            self.get_logger().warn("PauseTask rejected: %s" % response.message)
        return response

    def _stop(self, request, response):
        session = self.chunk_executor.session
        was_paused = self.paused
        self.chunk_executor.stop()
        self.inflight = False
        self.paused = False
        self.pause_target = None
        self.paused_session = None
        self.paused_request_pending = False
        self.hold_until_first_chunk = False
        self.last_action = None
        self.last_target = None
        self._last_hold = None
        self.reason = "stopped locally"
        self.get_logger().info(
            "StopTask: cleared targets for session=%s paused=%s; publishing ceased, no damping/hold transition sent"
            % (session or "none", was_paused)
        )
        response.success = True
        response.message = self.reason
        return response

    def _status(self, request, response):
        response.state = ("PAUSED" if self.paused else
                          ("RUNNING" if self.command_output is not None else "DRY_RUN")
                          if self.chunk_executor.session is not None else self.server_status)
        response.reason = self.reason
        response.checkpoint = self.checkpoint
        response.session_id = self.chunk_executor.session or ""
        return response

    def _request(self):
        if self.chunk_executor.session is None or self.inflight or self.server_status != "READY":
            return
        try:
            image, state, stamp = self._snapshot()
            seq = self.next_seq
            self.network.submit(self.chunk_executor.session, seq, stamp, self.prompt, image, state)
        except (ValueError, queue.Full) as exc:
            self._abort("observation/request failure: %s" % exc)
            return
        self.next_seq += 1
        self.inflight = True
        self.last_request_start = time.monotonic()
        pairing = ", ".join("%s=%.0fms" % (key, self.pair_offsets[key] * 1000)
                            for key in sorted(self.pair_offsets))
        self.get_logger().info("Prediction requested: session=%s seq=%d observation_stamp=%.3f pairing=[%s]" %
                               (self.chunk_executor.session, seq, stamp, pairing))

    def _tick(self):
        self._heartbeat()
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "status":
                self.server_status = event[1]["status"]
                self.checkpoint = event[1].get("checkpoint", "")
                status_log = (self.server_status, self.checkpoint, event[1].get("error", ""))
                if status_log != self._last_status_log:
                    level = self.get_logger().error if self.server_status == "ERROR" else self.get_logger().info
                    level("GPU model status: %s checkpoint=%s detail=%s" % status_log)
                    self._last_status_log = status_log
                if self.paused:
                    self.reason = (("paused; repeating last target at 30 Hz; command publishing %s" %
                                    self._output_mode()) if self.server_status == "READY" else
                                   "paused; GPU model %s: %s" %
                                   (self.server_status.lower(), event[1].get("error", "")))
                elif self.chunk_executor.session is None:
                    self.reason = event[1].get("error") or "model " + self.server_status.lower()
                if self.server_status != "READY" and self.chunk_executor.session is not None:
                    self._abort("model lost READY status")
            elif event[0] == "error":
                if self.paused and event[1] == self.paused_session:
                    self.paused_request_pending = False
                if event[1] is None:
                    self.server_status = "ERROR"
                    self.reason = event[3]
                    self.get_logger().error("GPU connection/status error: %s" % event[3])
                elif event[1] == self.chunk_executor.session:
                    self._abort("prediction failed: %s" % event[3])
                else:
                    self.get_logger().warn("Discarded error from inactive session=%s seq=%s" %
                                           (event[1], event[2]))
            elif event[0] == "actions":
                _, session, seq, stamp, actions, sent = event
                if self.paused and session == self.paused_session:
                    self.paused_request_pending = False
                if session != self.chunk_executor.session or not self.inflight or seq != self.next_seq - 1:
                    self.get_logger().warn("Discarded stale prediction: session=%s seq=%s" % (session, seq))
                    continue
                self.inflight = False
                try:
                    self._snapshot()
                    age = self.get_clock().now().nanoseconds * 1e-9 - stamp
                    self.chunk_executor.accept(session, seq, age, actions)
                    self.get_logger().info(
                        "Prediction accepted: session=%s seq=%d round_trip_ms=%.1f observation_age_s=%.3f queued=%s"
                        % (session, seq, (time.monotonic() - sent) * 1000, age,
                           self.chunk_executor.next_chunk is not None)
                    )
                except ValueError as exc:
                    self._abort("prediction rejected: %s" % exc)
        if self.paused:
            try:
                self._publish_target(self.pause_target)
            except ValueError as exc:
                self._abort("paused hold output failed: %s" % exc)
            return
        if self.chunk_executor.session is None:
            return
        try:
            self._snapshot()
        except ValueError as exc:
            self._abort("observation unavailable: %s" % exc)
            return
        target = self.chunk_executor.tick()
        if target is None:
            if self.hold_until_first_chunk:
                try:
                    self._publish_target(self.pause_target)
                except ValueError as exc:
                    self._abort("resuming hold output failed: %s" % exc)
            return
        if self.chunk_executor.held_since is not None:
            if self._last_hold is None:
                self._last_hold = time.monotonic()
                self.get_logger().warn(
                    "Chunk exhausted with a prediction in flight; repeating the last target at 30 Hz"
                )
        elif self._last_hold is not None:
            self.get_logger().warn("Hold ended after %.2fs; executing the new chunk from its first row"
                                   % (time.monotonic() - self._last_hold))
            self._last_hold = None
        self.hold_until_first_chunk = False
        self.pause_target = None
        self.last_action = target.copy()
        self.last_target = split_action(target)
        try:
            self._publish_target(target)
        except ValueError as exc:
            self._abort("motor command rejected: %s" % exc)
            return
        if self._last_chunk_start != self.chunk_executor.started:
            self._last_chunk_start = self.chunk_executor.started
            self.get_logger().info("Executing 32-step chunk at 30 Hz: session=%s command_publishing=%s" %
                                   (self.chunk_executor.session, self._output_mode()))
        if (not self.inflight and self.chunk_executor.next_chunk is None and
                time.monotonic() - self.chunk_executor.started >= 0.15 and
                self.last_request_start < self.chunk_executor.started):
            self._request()

    def destroy_node(self):
        self.network.close()
        self.network.join(timeout=3.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = Dex3Node()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
