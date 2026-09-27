#!/usr/bin/env python3
"""Wire-compatibility probe for the simulator's Unitree DDS topics.

One file, two roles, so a mismatch cannot hide behind duplicated type
definitions:

* ``--role sdk``  runs on the host with the isaac-sonic interpreter (the same
  ``unitree_sdk2py`` the simulator's DDS bridge uses): publishes LowState and
  dex3 HandState, subscribes to arm/hand command topics.
* ``--role ros``  runs in the flux-ros container with rclpy and the pinned
  ``unitree_hg`` messages: subscribes to the state topics and publishes LowCmd
  and HandCmd.

Both sides verify values, not just arrival, and print one JSON summary.  This
is how the "same topic, same type, same layout" claim is checked before a model
is ever connected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time

#: Distinctive, order-revealing state pattern.  Motor 15 is the first arm
#: motor in both orders, so seeing 1.5 there proves the arm slice is aligned.
ARM_BASE = 1.0
HAND_BASE = 2.0
STATE_PERIOD_S = 0.05


def _values(count, base):
    return [base + 0.01 * index for index in range(count)]


def _summary(role, state_received, commands_received, errors, checks):
    return {
        "role": role,
        "state_received": state_received,
        "commands_received": commands_received,
        "errors": errors,
        "checks": checks,
    }


def run_sdk(args):
    """Publish robot state and watch the command topics (host side)."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__HandState_ as HandStateDefault,
        unitree_hg_msg_dds__LowState_ as LowStateDefault,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic

    errors = []
    ChannelFactoryInitialize(args.domain_id, args.interface)
    low_pub = ChannelPublisher("rt/lowstate", LowState_)
    low_pub.Init()
    left_pub = ChannelPublisher("rt/dex3/left/state", HandState_)
    left_pub.Init()
    right_pub = ChannelPublisher("rt/dex3/right/state", HandState_)
    right_pub.Init()

    participant = DomainParticipant(args.domain_id)
    readers = {
        "arm_sdk": DataReader(participant, Topic(participant, "rt/arm_sdk", LowCmd_)),
        "left_cmd": DataReader(participant, Topic(participant, "rt/dex3/left/cmd", HandCmd_)),
        "right_cmd": DataReader(participant, Topic(participant, "rt/dex3/right/cmd", HandCmd_)),
    }
    received = {name: 0 for name in readers}
    last = {}
    lock = threading.Lock()
    stop = threading.Event()

    def watch():
        from cyclonedds.internal import InvalidSample

        while not stop.is_set():
            for name, reader in readers.items():
                try:
                    samples = reader.take(64)
                except Exception as exc:  # noqa: BLE001 - probe reports, not raises
                    errors.append("%s: %s" % (name, exc))
                    continue
                for sample in samples:
                    if isinstance(sample, InvalidSample):
                        continue
                    with lock:
                        received[name] += 1
                        last[name] = sample
            time.sleep(0.02)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    low = LowStateDefault()
    left = HandStateDefault()
    right = HandStateDefault()
    low.motor_state[15].q = ARM_BASE
    left.motor_state[0].q = HAND_BASE
    right.motor_state[0].q = HAND_BASE
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        # The pattern advances with time so a subscriber can tell fresh frames
        # from a stale cache.
        phase = (time.monotonic() * 10.0) % 1.0
        for index in range(15, 29):
            low.motor_state[index].q = ARM_BASE + 0.01 * (index - 15) + phase
        for index in range(7):
            left.motor_state[index].q = HAND_BASE + 0.01 * index + phase
            right.motor_state[index].q = HAND_BASE + 0.01 * index + phase
        low_pub.Write(low)
        left_pub.Write(left)
        right_pub.Write(right)
        time.sleep(STATE_PERIOD_S)
    stop.set()
    watcher.join(timeout=1.0)

    checks = {
        "arm_sdk_commands": received["arm_sdk"],
        "left_hand_commands": received["left_cmd"],
        "right_hand_commands": received["right_cmd"],
    }
    with lock:
        arm = last.get("arm_sdk")
        left_cmd = last.get("left_cmd")
        right_cmd = last.get("right_cmd")
    if arm is not None:
        checks["arm_first_motor_q"] = round(float(arm.motor_cmd[15].q), 4)
        checks["arm_last_motor_q"] = round(float(arm.motor_cmd[28].q), 4)
        checks["arm_enable_slot_29_q"] = round(float(arm.motor_cmd[29].q), 4)
    if left_cmd is not None:
        checks["left_hand_first_motor_q"] = round(float(left_cmd.motor_cmd[0].q), 4)
        checks["left_hand_motor_count"] = len(left_cmd.motor_cmd)
    if right_cmd is not None:
        checks["right_hand_first_motor_q"] = round(float(right_cmd.motor_cmd[0].q), 4)
        checks["right_hand_motor_count"] = len(right_cmd.motor_cmd)
    return _summary("sdk", {"published": int(args.seconds / STATE_PERIOD_S)}, received, errors, checks)


def run_ros(args):
    """Subscribe to state topics and publish command topics (container side)."""
    import rclpy
    from rclpy.node import Node
    from unitree_hg.msg import HandCmd, HandState, LowCmd, LowState, MotorCmd

    received = {"lowstate": 0, "left_state": 0, "right_state": 0}
    errors = []
    last_state = {}

    class Probe(Node):
        def __init__(self):
            super().__init__("flux_dds_probe")
            self.create_subscription(LowState, "/lowstate", self._low, 10)
            self.create_subscription(HandState, "/dex3/left/state", self._left, 10)
            self.create_subscription(HandState, "/dex3/right/state", self._right, 10)
            self.arm_pub = self.create_publisher(LowCmd, "/arm_sdk", 10)
            self.left_pub = self.create_publisher(HandCmd, "/dex3/left/cmd", 10)
            self.right_pub = self.create_publisher(HandCmd, "/dex3/right/cmd", 10)

        def _low(self, msg):
            received["lowstate"] += 1
            last_state["arm"] = [round(float(msg.motor_state[index].q), 4) for index in range(15, 29)]

        def _left(self, msg):
            received["left_state"] += 1
            last_state["left"] = [round(float(motor.q), 4) for motor in msg.motor_state]

        def _right(self, msg):
            received["right_state"] += 1
            last_state["right"] = [round(float(motor.q), 4) for motor in msg.motor_state]

    rclpy.init()
    node = Probe()
    executor_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    executor_thread.start()
    deadline = time.monotonic() + args.seconds
    published = {"arm": 0, "left": 0, "right": 0}
    try:
        while time.monotonic() < deadline:
            phase = (time.monotonic() * 10.0) % 1.0
            arm = LowCmd()
            if len(arm.motor_cmd) != 35:
                errors.append("LowCmd has %d motor slots" % len(arm.motor_cmd))
                break
            arm.motor_cmd[29].q = 1.0
            for offset, index in enumerate(range(15, 29)):
                arm.motor_cmd[index].q = 3.0 + 0.01 * offset + phase
            left = HandCmd()
            left.motor_cmd = [MotorCmd() for _ in range(7)]
            right = HandCmd()
            right.motor_cmd = [MotorCmd() for _ in range(7)]
            for index in range(7):
                left.motor_cmd[index].q = 4.0 + 0.01 * index + phase
                right.motor_cmd[index].q = 5.0 + 0.01 * index + phase
            node.arm_pub.publish(arm)
            node.left_pub.publish(left)
            node.right_pub.publish(right)
            published["arm"] += 1
            published["left"] += 1
            published["right"] += 1
            time.sleep(STATE_PERIOD_S)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        executor_thread.join(timeout=2.0)

    checks = {
        "lowstate_frames": received["lowstate"],
        "left_state_frames": received["left_state"],
        "right_state_frames": received["right_state"],
    }
    if "arm" in last_state:
        checks["arm_first_motor_q"] = last_state["arm"][0]
        checks["arm_last_motor_q"] = last_state["arm"][-1]
    if "left" in last_state:
        checks["left_hand_first_motor_q"] = last_state["left"][0]
        checks["left_hand_motors"] = len(last_state["left"])
    if "right" in last_state:
        checks["right_hand_first_motor_q"] = last_state["right"][0]
        checks["right_hand_motors"] = len(last_state["right"])
    return _summary("ros", received, published, errors, checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("sdk", "ros"), required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--domain-id", type=int, default=42)
    parser.add_argument("--interface", default="lo")
    args = parser.parse_args(argv)

    summary = run_sdk(args) if args.role == "sdk" else run_ros(args)
    print(json.dumps(summary, sort_keys=True), flush=True)

    checks = summary["checks"]
    failures = []
    if summary["errors"]:
        failures.append("errors reported")
    if args.role == "sdk":
        if checks["arm_sdk_commands"] == 0:
            failures.append("no /arm_sdk command received")
        if checks["left_hand_commands"] == 0 or checks["right_hand_commands"] == 0:
            failures.append("no hand command received")
        if checks.get("arm_last_motor_q") is None or not math.isfinite(checks["arm_last_motor_q"]):
            failures.append("arm command value not finite")
    else:
        if checks["lowstate_frames"] == 0 or checks["left_state_frames"] == 0 or checks["right_state_frames"] == 0:
            failures.append("no robot state received")
        if checks.get("arm_last_motor_q") is None:
            failures.append("arm state slice missing")
    if failures:
        print("PROBE FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("PROBE PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
