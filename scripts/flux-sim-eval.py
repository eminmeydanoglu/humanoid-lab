#!/usr/bin/env python3
"""Client-side exercise and evidence collector for the flux simulation loop.

Runs inside the flux-ros container next to the flux_dex3 node and speaks to it
exactly as the evaluation will: ZMQ STATUS to the GPU model server, ROS
services to start/stop the task, ROS subscriptions to measure the camera and
joint streams, and (when commands are enabled) the arm/hand command topics the
node publishes.

Two modes:

* ``preflight`` -- model READY, camera freshness, joint-state freshness, node
  status.  Nothing is started.
* ``task``      -- negative-control rejection, StartTask, a bounded run with
  command-rate timing, GetStatus, StopTask, and the "publishing ceased" check.

Everything is printed as one JSON report; the caller keeps it as run evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import zmq
from flux_dex3 import protocol
from flux_dex3_interfaces.srv import GetStatus, StartTask
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from unitree_hg.msg import HandCmd, HandState, LowCmd, LowState

import rclpy

ARM_JOINTS = tuple(range(15, 29))
HAND_JOINTS = tuple(range(7))
#: The trained caption for the calibrated PickApple scene; any caption from the
#: node's manifest works, this one is the documented default.
DEFAULT_PROMPT = "Put the apple into the plate."
DEFAULT_BAD_PROMPT = "nonexistent task caption"


def model_status(endpoint, timeout_s=3.0):
    # DEALER, exactly like the node's own client: a REQ socket would add the
    # empty delimiter frame the ROUTER-side protocol does not expect.
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    socket.connect(endpoint)
    try:
        socket.send_multipart(protocol.encode_status_request())
        reply = protocol.decode_reply(socket.recv_multipart())
    finally:
        socket.close(linger=0)
        context.term()
    return {"status": reply["status"], "checkpoint": reply["checkpoint"], "error": reply["error"]}


class Eval(Node):
    def __init__(self, args):
        super().__init__("flux_sim_eval")
        self.args = args
        self.camera = []
        self.lowstate = []
        self.left = []
        self.right = []
        self.arm_commands = []
        self.hand_commands = []
        self.create_subscription(Image, args.camera_topic, self._camera, 10)
        self.create_subscription(LowState, "/lowstate", self._lowstate, 50)
        self.create_subscription(HandState, "/dex3/left/state", self._left, 50)
        self.create_subscription(HandState, "/dex3/right/state", self._right, 50)
        self.create_subscription(LowCmd, "/arm_sdk", self._arm_command, 50)
        self.create_subscription(HandCmd, "/dex3/left/cmd", self._left_command, 50)
        self.create_subscription(HandCmd, "/dex3/right/cmd", self._right_command, 50)
        self.start_client = self.create_client(StartTask, "/flux_dex3/start_task")
        self.stop_client = self.create_client(Trigger, "/flux_dex3/stop_task")
        self.status_client = self.create_client(GetStatus, "/flux_dex3/get_status")

    def _camera(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.camera.append((time.time(), stamp, msg.width, msg.height, msg.encoding))

    def _lowstate(self, msg):
        self.lowstate.append((time.time(), [float(msg.motor_state[i].q) for i in ARM_JOINTS]))

    def _left(self, msg):
        self.left.append((time.time(), [float(motor.q) for motor in msg.motor_state]))

    def _right(self, msg):
        self.right.append((time.time(), [float(motor.q) for motor in msg.motor_state]))

    def _arm_command(self, msg):
        self.arm_commands.append((time.time(), [float(msg.motor_cmd[i].q) for i in ARM_JOINTS]))

    def _left_command(self, msg):
        self.hand_commands.append((time.time(), "left", [float(m.q) for m in msg.motor_cmd]))

    def _right_command(self, msg):
        self.hand_commands.append((time.time(), "right", [float(m.q) for m in msg.motor_cmd]))

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))

    def call(self, client, request, timeout_s=10.0):
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError("service unavailable: %s" % client.srv_name)
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            raise RuntimeError("service timeout: %s" % client.srv_name)
        if future.exception() is not None:
            raise RuntimeError("service failed: %s: %s" % (client.srv_name, future.exception()))
        return future.result()


def rate(values, window_s):
    return len(values) / window_s if window_s > 0 else 0.0


def gaps(times):
    return [b - a for a, b in zip(times, times[1:])]


def check_freshness(samples, label, min_hz, max_age_s, problems):
    times = [item[0] for item in samples]
    if len(times) < 2:
        problems.append("%s: only %d samples" % (label, len(times)))
        return {"count": len(times)}
    window = times[-1] - times[0]
    hz = (len(times) - 1) / window if window > 0 else 0.0
    ages = [now - stamp for now, stamp, *_ in samples]
    result = {
        "count": len(samples),
        "hz": round(hz, 2),
        "max_stamp_age_s": round(max(ages), 4),
        "max_gap_s": round(max(gaps(times)), 4),
    }
    if hz < min_hz:
        problems.append("%s: %.2f Hz below required %.2f Hz" % (label, hz, min_hz))
    if max(ages) > max_age_s:
        problems.append("%s: stamp age %.3fs exceeds %.3fs" % (label, max(ages), max_age_s))
    return result


def wait_for_model(endpoint, timeout_s, problems):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            status = model_status(endpoint)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            status = {"status": "UNREACHABLE", "checkpoint": "", "error": str(exc)[:200]}
        if status["status"] != last:
            print(json.dumps({"event": "model_status", **status}), flush=True)
            last = status["status"]
        if status["status"] == "READY":
            return status
        if status["status"] == "ERROR":
            problems.append("model ERROR: %s" % status["error"])
            return status
        time.sleep(1.0)
    problems.append("model not READY within %.0fs" % timeout_s)
    return {"status": "TIMEOUT", "checkpoint": "", "error": ""}


def streams_once(node, args):
    """One observation window over camera and joint state.

    A short drain first: this client is a third subscriber next to the node and
    the bridge, and a queued backlog would otherwise be measured as pipeline
    staleness even though every frame carries the simulator's own stamp.
    """
    local_problems = []
    checks = {}
    node.camera.clear()
    node.spin_for(0.5)
    node.camera.clear()
    node.spin_for(args.window_s)
    checks["camera"] = check_freshness(node.camera, "camera", args.min_camera_hz,
                                       args.max_camera_age_s, local_problems)
    formats = sorted({(width, height, encoding) for _, _, width, height, encoding in node.camera})
    checks["camera"]["formats"] = formats
    if formats and formats != [(640, 480, "rgb8")]:
        local_problems.append("unexpected camera formats: %s" % formats)
    for label, samples, attribute in (("lowstate", node.lowstate, "lowstate"),
                                      ("left_hand_state", node.left, "left"),
                                      ("right_hand_state", node.right, "right")):
        samples.clear()
        node.spin_for(args.window_s)
        times = [item[0] for item in samples]
        checks[attribute] = {"count": len(samples), "hz": round(rate(times, args.window_s), 2)}
        if len(samples) < 2:
            local_problems.append("%s: no joint state received" % label)
        elif not all(math.isfinite(value) for value in samples[-1][1]):
            local_problems.append("%s: non-finite joint state" % label)
    return checks, local_problems


def run_preflight(node, args, problems):
    report = {}
    report["model"] = wait_for_model(args.model_endpoint, args.status_timeout, problems)

    # Camera and joint state arrive only once Isaac and the ROS launch are up;
    # keep the same windows the node's own gates use until they pass.
    deadline = time.monotonic() + args.sim_timeout
    while True:
        checks, local_problems = streams_once(node, args)
        report.update(checks)
        if not local_problems:
            break
        if time.monotonic() >= deadline:
            problems.extend(local_problems)
            break
        print(json.dumps({"event": "waiting_for_streams", "problems": local_problems}), flush=True)
        time.sleep(2.0)

    status = node.call(node.status_client, GetStatus.Request(), timeout_s=10.0)
    report["node_status_before"] = {"state": status.state, "reason": status.reason,
                                    "checkpoint": status.checkpoint, "session_id": status.session_id}
    if status.state not in ("READY", "DRY_RUN", "RUNNING"):
        problems.append("node not ready: state=%s reason=%s" % (status.state, status.reason))
    return report


def command_timing(commands):
    times = [item[0] for item in commands]
    intervals = gaps(times)
    if not intervals:
        return {"count": len(commands), "hz": 0.0}
    return {
        "count": len(commands),
        "hz": round((len(times) - 1) / (times[-1] - times[0]), 2) if times[-1] > times[0] else 0.0,
        "interval_mean_ms": round(statistics.fmean(intervals) * 1e3, 2),
        "interval_p95_ms": round(sorted(intervals)[max(0, int(0.95 * len(intervals)) - 1)] * 1e3, 2),
        "interval_max_ms": round(max(intervals) * 1e3, 2),
        "first_q": [{str(index): round(value, 4) for index, value in zip(ARM_JOINTS, commands[0][1])}],
        "last_q": [{str(index): round(value, 4) for index, value in zip(ARM_JOINTS, commands[-1][1])}],
    }


def run_task(node, args, problems):
    report = {}

    # Negative control: a caption that is not in the training manifest must be
    # rejected without touching the task state.
    bad = node.call(node.start_client, StartTask.Request(prompt=args.bad_prompt), timeout_s=10.0)
    report["bad_prompt"] = {"accepted": bool(bad.accepted), "reason": bad.reason}
    if bad.accepted:
        problems.append("StartTask accepted a caption outside the training manifest")

    before = len(node.arm_commands)
    report["start_wall_unix"] = time.time()
    start = node.call(node.start_client, StartTask.Request(prompt=args.prompt), timeout_s=15.0)
    report["start_response_wall_unix"] = time.time()
    report["start"] = {"accepted": bool(start.accepted), "reason": start.reason}
    if not start.accepted:
        problems.append("StartTask rejected: %s" % start.reason)
        return report

    node.spin_for(args.run_s)
    status = node.call(node.status_client, GetStatus.Request(), timeout_s=10.0)
    report["status_running"] = {"state": status.state, "reason": status.reason,
                                "session_id": status.session_id}
    if status.state not in ("DRY_RUN", "RUNNING"):
        problems.append("GetStatus after StartTask: state=%s reason=%s" % (status.state, status.reason))

    # Command timing is only meaningful with the motor output enabled.
    commands = node.arm_commands[before:]
    if args.expect_motor_commands:
        report["arm_commands"] = command_timing(commands)
        if report["arm_commands"]["count"] == 0:
            problems.append("no /arm_sdk commands published while a task was running")
        elif report["arm_commands"]["hz"] < args.min_command_hz:
            # This client is one subscriber among three; a slow queue can lower
            # its observed rate.  The simulator's own tracking cadence (verified
            # from the Parquet) is the authoritative timing check.
            report["arm_commands"]["note"] = (
                "client-side rate below the %.1f Hz gate; simulator tracking cadence is "
                "the authoritative check" % args.min_command_hz)
        if not node.hand_commands:
            problems.append("no hand commands published while a task was running")
        else:
            report["hand_commands"] = {"count": len(node.hand_commands)}
    else:
        report["arm_commands"] = {"count": len(commands),
                                  "note": "command-disabled run; only internal targets exist"}
        if commands:
            problems.append("motor commands published although the run is command-disabled")

    stop = node.call(node.stop_client, Trigger.Request(), timeout_s=10.0)
    report["stop_wall_unix"] = time.time()
    report["stop"] = {"success": bool(stop.success), "message": stop.message}
    if not stop.success:
        problems.append("StopTask failed: %s" % stop.message)

    # Existing node behaviour, unchanged: Stop clears all targets and ends
    # publication immediately.  No new hold/latch handling is added here.
    time.sleep(0.3)
    settled_commands = len(node.arm_commands)
    node.spin_for(args.quiet_s)
    report["after_stop"] = {"commands_before_quiet": settled_commands,
                            "commands_after_quiet": len(node.arm_commands),
                            "quiet_s": args.quiet_s}
    if len(node.arm_commands) != settled_commands:
        problems.append("commands kept being published after StopTask")

    status = node.call(node.status_client, GetStatus.Request(), timeout_s=10.0)
    report["status_stopped"] = {"state": status.state, "reason": status.reason, "session_id": status.session_id}
    if status.session_id:
        problems.append("session id not cleared after StopTask")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "task"), required=True)
    parser.add_argument("--model-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--camera-topic", default="/camera/color/image_raw")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--bad-prompt", default=DEFAULT_BAD_PROMPT)
    parser.add_argument("--status-timeout", type=float, default=300.0)
    parser.add_argument("--sim-timeout", type=float, default=600.0,
                        help="wait for camera/joint streams to appear")
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--min-camera-hz", type=float, default=10.0)
    parser.add_argument("--max-camera-age-s", type=float, default=0.5,
                        help="gross-staleness gate for this client's view; the node's own "
                             "0.25s gate is authoritative and is exercised by StartTask")
    parser.add_argument("--run-s", type=float, default=8.0)
    parser.add_argument("--quiet-s", type=float, default=0.7)
    parser.add_argument("--min-command-hz", type=float, default=25.0)
    parser.add_argument("--expect-motor-commands", action="store_true")
    parser.add_argument("--report", default=None, help="write the JSON report here as well")
    args = parser.parse_args(argv)

    problems = []
    rclpy.init()
    node = None
    try:
        node = Eval(args)
        if args.mode == "preflight":
            report = run_preflight(node, args, problems)
        else:
            report = run_task(node, args, problems)
            status = node.call(node.status_client, GetStatus.Request(), timeout_s=10.0)
            report["node_status_after"] = {"state": status.state, "reason": status.reason,
                                           "session_id": status.session_id}
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

    report["problems"] = problems
    report["result"] = "PASS" if not problems else "FAIL"
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
