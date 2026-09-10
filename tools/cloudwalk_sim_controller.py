"""Live simulator-side transport for an explicitly started CloudWalk controller."""
from __future__ import annotations

import base64
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

from cloudwalk_closed_loop import BodyCommand, SonicState, native_decoder_state
from sonic_isaac_inspire_adapter import INSPIRE_HAND_JOINTS, InspireFTPGripMapper, V4Action, unpack_protocol_v4

CONTROL_ENDPOINT = "tcp://127.0.0.1:56115"
OBSERVATION_ENDPOINT = "tcp://127.0.0.1:56116"
BODY_STALE_SECONDS = 0.50
STARTUP_GRACE_SECONDS = 2.0


@dataclass
class ControllerStatus:
    active: bool = False
    state: str = "passive"
    sequence: int = 0
    body_frames: int = 0
    action_frames: int = 0
    last_body_monotonic: float | None = None
    started_monotonic: float | None = None


class SimulatorControllerBridge:
    def __init__(
        self,
        *,
        state_endpoint: str,
        body_endpoint: str,
        action_endpoint: str,
        control_endpoint: str = CONTROL_ENDPOINT,
        observation_endpoint: str = OBSERVATION_ENDPOINT,
    ) -> None:
        try:
            import zmq
        except ModuleNotFoundError:
            sys.path.insert(0, "/opt/venvs/sonic-sim/lib/python3.11/site-packages")
            import zmq

        self.zmq = zmq
        self.context = zmq.Context()
        self.state_pub = self.context.socket(zmq.PUB); self.state_pub.setsockopt(zmq.LINGER, 0); self.state_pub.bind(state_endpoint)
        self.observation_pub = self.context.socket(zmq.PUB); self.observation_pub.setsockopt(zmq.LINGER, 0); self.observation_pub.bind(observation_endpoint)
        self.body_sub = self.context.socket(zmq.SUB); self.body_sub.setsockopt(zmq.LINGER, 0); self.body_sub.setsockopt(zmq.SUBSCRIBE, b""); self.body_sub.connect(body_endpoint)
        self.action_sub = self.context.socket(zmq.SUB); self.action_sub.setsockopt(zmq.LINGER, 0); self.action_sub.setsockopt(zmq.SUBSCRIBE, b""); self.action_sub.connect(action_endpoint)
        self.control = self.context.socket(zmq.REP); self.control.setsockopt(zmq.LINGER, 0); self.control.bind(control_endpoint)
        self.status = ControllerStatus()
        self.last_body = None
        self.last_raw_action = (0.0,) * 29
        self.last_hand_action = V4Action((0.0,) * 64, (0.0,) * 7, (0.0,) * 7)
        self.last_body_sequence = -1
        self.last_action_sequence = -1

    def poll_control(self, actuation: Any, robot: Any, standing_targets: Any, timeline_playing: bool) -> dict[str, Any] | None:
        try:
            request = self.control.recv_json(flags=self.zmq.NOBLOCK)
        except self.zmq.Again:
            return None
        op = request.get("op")
        if op == "start":
            actuation.set_controlled()
            robot.set_joint_position_target(standing_targets)
            now = time.monotonic()
            self.status = ControllerStatus(active=True, state="holding", started_monotonic=now)
            self.last_body = None
            self.last_body_sequence = -1
            self.last_action_sequence = -1
            response = {"ok": True, "state": self.status.state, "timeline_playing": timeline_playing}
        elif op == "run":
            if not self.status.active:
                response = {"ok": False, "error": "controller must be started before policy activation"}
            else:
                self.status.state = "controlled"
                self.status.started_monotonic = time.monotonic()
                self.status.last_body_monotonic = None
                response = {"ok": True, "state": self.status.state, "timeline_playing": timeline_playing}
        elif op == "stop":
            self._set_passive(actuation)
            response = {"ok": True, "state": self.status.state, "timeline_playing": timeline_playing}
        elif op == "status":
            response = {
                "ok": True,
                **self.snapshot(timeline_playing),
                "actuation_mode": actuation.mode.value,
                "root_pos_w": [float(value) for value in robot.data.root_pos_w[0].tolist()],
                "root_lin_vel_w": [float(value) for value in robot.data.root_lin_vel_w[0].tolist()],
            }
        else:
            response = {"ok": False, "error": f"unsupported controller operation: {op}"}
        self.control.send_json(response)
        return response

    def snapshot(self, timeline_playing: bool) -> dict[str, Any]:
        return {
            "state": self.status.state,
            "active": self.status.active,
            "sequence": self.status.sequence,
            "body_frames": self.status.body_frames,
            "action_frames": self.status.action_frames,
            "timeline_playing": timeline_playing,
        }

    def _set_passive(self, actuation: Any) -> None:
        actuation.set_passive()
        self.status.active = False
        self.status.state = "passive"

    def _body_command_stale(self, now: float) -> bool:
        if self.status.state != "controlled":
            return False
        started = self.status.started_monotonic or now
        last_body = self.status.last_body_monotonic
        return now - started > STARTUP_GRACE_SECONDS and (last_body is None or now - last_body > BODY_STALE_SECONDS)

    def _drain_actions(self) -> None:
        while True:
            try:
                packet = self.action_sub.recv(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                return
            action, sequence = unpack_protocol_v4(packet)
            if sequence > self.last_action_sequence:
                self.last_action_sequence = sequence
                self.last_hand_action = action
                self.status.action_frames += 1

    def _drain_body(self) -> None:
        while True:
            try:
                packet = self.body_sub.recv(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                return
            command = BodyCommand.unpack(packet)
            if command.sequence > self.last_body_sequence:
                self.last_body_sequence = command.sequence
                self.last_body = command.positions
                self.last_raw_action = command.raw_actions
                self.status.body_frames += 1
                self.status.last_body_monotonic = time.monotonic()

    def publish_policy_observation(
        self,
        *,
        robot: Any,
        camera: Any,
        body_ids: Any,
        prompt: str,
        validate_observation: Any,
    ) -> None:
        body_q_mujoco = tuple(float(value) for value in robot.data.joint_pos[0, body_ids].tolist())
        rgb = camera.data.output["rgb"][0].cpu().numpy()
        policy_state = (
            list(body_q_mujoco[:22])
            + list(self.last_hand_action.left_hand)
            + list(body_q_mujoco[22:])
            + list(self.last_hand_action.right_hand)
        )
        validate_observation(rgb, policy_state, prompt)
        self.observation_pub.send_json({
            "sequence": self.status.sequence,
            "timestamp": time.time(),
            "rgb": base64.b64encode(rgb.tobytes()).decode("ascii"),
            "state": policy_state,
            "base_quat": [float(value) for value in robot.data.root_quat_w[0].tolist()],
        })

    def control_tick(
        self,
        *,
        robot: Any,
        camera: Any,
        body_ids: Any,
        inspire_ids: Any,
        standing_targets: Any,
        mapper: InspireFTPGripMapper,
        hand_limits: Any,
        hand_open_positions: Any,
        prompt: str,
        validate_observation: Any,
        actuation: Any,
    ) -> str | None:
        if not self.status.active:
            return None
        now = time.monotonic()
        self._drain_actions()
        self._drain_body()
        if self._body_command_stale(now):
            self._set_passive(actuation)
            return "controller_stale"

        body_q_mujoco = tuple(float(value) for value in robot.data.joint_pos[0, body_ids].tolist())
        body_qd_mujoco = tuple(float(value) for value in robot.data.joint_vel[0, body_ids].tolist())
        body_q, body_qd = native_decoder_state(body_q_mujoco, body_qd_mujoco)
        angular = tuple(float(value) for value in robot.data.root_ang_vel_b[0].tolist())
        gravity = tuple(float(value) for value in robot.data.projected_gravity_b[0].tolist())
        sequence = self.status.sequence
        self.state_pub.send(SonicState(sequence, time.monotonic_ns(), angular, body_q, body_qd, self.last_raw_action, gravity).pack())

        if sequence % 20 == 0:
            self.publish_policy_observation(
                robot=robot,
                camera=camera,
                body_ids=body_ids,
                prompt=prompt,
                validate_observation=validate_observation,
            )

        if self.status.state == "controlled" and self.last_body is not None:
            targets = standing_targets.clone()
            targets[:, body_ids] = __import__("torch").tensor(self.last_body, device=targets.device).unsqueeze(0)
            targets[:, inspire_ids] = __import__("torch").tensor(
                mapper.targets(self.last_hand_action, hand_limits, hand_open_positions), device=targets.device
            ).unsqueeze(0)
            robot.set_joint_position_target(targets)
        self.status.sequence += 1
        return None

    def close(self) -> None:
        self.state_pub.close()
        self.observation_pub.close()
        self.body_sub.close()
        self.action_sub.close()
        self.control.close()
        self.context.term()
