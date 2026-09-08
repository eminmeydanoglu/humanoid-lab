#!/usr/bin/env python3
"""Run the deterministic G1 + Inspire FTP CloudWalk Isaac scene on Raider.

This is an Isaac articulation boundary only.  It does not replace upstream
GR00T inference or the SONIC C++ controller, and never publishes Dex3 commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cloudwalk_adapter import CHECKPOINT_REVISION, EMBODIMENT, G1_BODY_JOINTS, G1_INSPIRE_FTP_JOINTS, PROMPT, validate_observation  # noqa: E402
from cloudwalk_closed_loop import DEFAULT_ANGLES  # noqa: E402
from cloudwalk_scene import configure_rtx, decorate_scene, load_scene_config, make_scene_cfg  # noqa: E402
from sonic_isaac_inspire_adapter import ContractError, INSPIRE_HAND_JOINTS, InspireFTPGripMapper, Lifecycle, LifecycleGuard, SafeReference, V4Action  # noqa: E402

DEFAULT_DATASET = Path("/data/datasets/groot/gr00t-g1-grab-bottle-right-hand-v10")
DEFAULT_CHECKPOINT = Path("/data/models/cloudwalk-gr00t-n17-g1-grab-bottle-rh-371ep-v10-finetune/checkpoint-30000")
EXPECTED_JOINT_COUNT = 53
EXPECTED_BODY_COUNT = 54
LEFT_INSPIRE_JOINTS = (
    "L_index_proximal_joint", "L_middle_proximal_joint", "L_pinky_proximal_joint",
    "L_ring_proximal_joint", "L_thumb_proximal_yaw_joint", "L_index_intermediate_joint",
    "L_middle_intermediate_joint", "L_pinky_intermediate_joint", "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint", "L_thumb_intermediate_joint", "L_thumb_distal_joint",
)
RIGHT_INSPIRE_JOINTS = tuple(name.replace("L_", "R_", 1) for name in LEFT_INSPIRE_JOINTS)


def _fail(message: str) -> None:
    raise ContractError(message)


# Process exit code read by the finally-block hard exit below; set in __main__.
_exit_code = [2]


def _validate_cloudwalk_contract(dataset: Path, checkpoint: Path) -> None:
    info = dataset / "meta" / "info.json"
    processor = checkpoint / "processor_config.json"
    embodiment = checkpoint / "embodiment_id.json"
    if not info.is_file() or not processor.is_file() or not embodiment.is_file():
        _fail("CloudWalk dataset or checkpoint metadata is incomplete")
    names = tuple(json.loads(info.read_text())["features"]["observation.state"]["names"])
    if names != G1_INSPIRE_FTP_JOINTS:
        _fail("dataset does not have the exact CloudWalk 43-value joint ordering")
    if "unitree_g1_sonic" not in processor.read_text().lower() or "unitree_g1_sonic" not in embodiment.read_text().lower():
        _fail("checkpoint does not expose UNITREE_G1_SONIC")
    expected = {"config.json": "7d06d0da7baf0873016f2172fd2df5478f52a92e1c90e94ce2585e10f170e249", "processor_config.json": "c0ea0f1e830dfd25238dff8baaad80d1551bf324dc1e53076b3349fabe3d6540"}
    for name, checksum in expected.items():
        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != checksum:
            _fail(f"checkpoint metadata checksum mismatch: {name}")


def _enable_required_camera(args: argparse.Namespace) -> None:
    args.enable_cameras = True


def _safe_reference() -> SafeReference:
    # SONIC's upstream checkpoint-specific initial standing token, not zeros(64).
    token = (
        -0.0625, 0.0, -0.0625, -0.125, -0.1875, -0.0625, 0.1875, 0.25,
        0.1875, -0.125, 0.0625, -0.0625, -0.25, -0.25, -0.3125, -0.0625,
        0.0, -0.0625, -0.125, -0.1875, 0.0, -0.25, 0.0, -0.25, -0.0625,
        0.0625, 0.125, -0.125, 0.25, 0.1875, 0.25, -0.125, 0.125, 0.1875,
        -0.0625, 0.0, -0.1875, -0.1875, 0.25, 0.0, 0.0, -0.125, 0.0625,
        0.0, -0.0625, -0.0625, 0.1875, -0.0625, 0.0, 0.0625, 0.125, 0.0625,
        0.125, 0.0625, 0.125, 0.0, 0.125, 0.1875, 0.0, 0.0, 0.0625,
        0.0625, 0.1875, 0.0625,
    )
    return SafeReference(V4Action(token, (0.0,) * 7, (0.0,) * 7), "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509", "recorded-upstream-sonic-initial-poses.py@a0732b642c0333077e127a2f56ab0014c196bca4")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--capture-path", type=Path, required=True)
    parser.add_argument("--rollout-metrics-path", type=Path)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--scene-config", type=Path, default=ROOT / "configs" / "cloudwalk_scene.json")
    parser.add_argument("--table-position", type=float, nargs=3, default=None)
    parser.add_argument("--table-size", type=float, nargs=3, default=None)
    parser.add_argument("--bottle-position", type=float, nargs=3, default=None)
    parser.add_argument("--interactive", action="store_true", help="Run a paused GUI scene; Timeline Play starts free-base physics.")
    parser.add_argument("--hand-sign-probe", action="store_true")
    parser.add_argument("--checkpoint-revision", default=CHECKPOINT_REVISION)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--vla-endpoint", default="tcp://127.0.0.1:56110")
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:56112")
    parser.add_argument("--body-endpoint", default="tcp://127.0.0.1:56113")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:56114")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    _enable_required_camera(args)
    if args.checkpoint_revision != CHECKPOINT_REVISION:
        _fail("checkpoint revision is not the verified CloudWalk checkpoint revision")
    if args.interactive and args.closed_loop:
        _fail("interactive GUI mode does not start GR00T or SONIC; use the headless closed-loop launcher")
    _validate_cloudwalk_contract(args.dataset, args.checkpoint)
    scene_config = load_scene_config(args.scene_config)
    if args.table_position is not None:
        scene_config["table"]["position_m"] = list(args.table_position)
    if args.table_size is not None:
        scene_config["table"]["size_m"] = list(args.table_size)
    if args.bottle_position is not None:
        scene_config["bottle"]["position_m"] = list(args.bottle_position)
    if args.validate_only:
        print(json.dumps({"embodiment": EMBODIMENT, "prompt": PROMPT, "asset_cfg": "CLOUDWALK_G1_INSPIRE_FTP_FREE_BASE_CFG", "scene_config": str(args.scene_config)}, sort_keys=True))
        return 0

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import imageio.v3 as iio
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

        # The upstream asset defaults to a fixed, gravity-disabled manipulation setup.
        # CloudWalk runs on the same USD with a free base and ordinary PhysX gravity.
        robot_cfg = G1_INSPIRE_FTP_CFG.copy()
        robot_cfg.spawn.rigid_props.disable_gravity = False
        robot_cfg.spawn.articulation_props.fix_root_link = False
        if robot_cfg.spawn.rigid_props.disable_gravity or robot_cfg.spawn.articulation_props.fix_root_link:
            _fail("CloudWalk physics requires a free-base gravity-enabled G1")
        configure_rtx(scene_config)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
        scene = InteractiveScene(make_scene_cfg(scene_config, robot_cfg)(num_envs=1, env_spacing=1.0))
        # All visual child prims are authored before reset creates the PhysX tensor view.
        decorate_scene(scene_config)
        sim.reset()
        robot, camera = scene["robot"], scene["head_camera"]
        if robot.num_joints != EXPECTED_JOINT_COUNT or robot.num_bodies != EXPECTED_BODY_COUNT:
            _fail(f"G1 Inspire runtime contract is {robot.num_joints} joints/{robot.num_bodies} bodies, expected 53/54")
        required = set(LEFT_INSPIRE_JOINTS + RIGHT_INSPIRE_JOINTS)
        if not required.issubset(robot.joint_names):
            _fail("G1_INSPIRE_FTP_CFG did not expose its exact 24 Inspire joint names")
        body_joint_names = tuple(name for name in robot.joint_names if name not in required)
        if len(body_joint_names) != 29:
            _fail(f"G1 Inspire runtime contract has {len(body_joint_names)} body joints, expected 29")
        if set(body_joint_names) != set(G1_BODY_JOINTS):
            _fail(f"Isaac body DOFs differ from the pinned SONIC names: {body_joint_names}")
        default_pos, default_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        # This ordering is shared by the upstream SONIC decoder and the free-base initial pose.
        body_ids = torch.tensor([robot.joint_names.index(name) for name in G1_BODY_JOINTS], device=default_pos.device)
        inspire_ids = torch.tensor([robot.joint_names.index(name) for name in INSPIRE_HAND_JOINTS], device=default_pos.device)
        hand_limits = tuple(tuple(float(value) for value in robot.data.soft_joint_pos_limits[0, joint_id].tolist()) for joint_id in inspire_ids.tolist())
        hand_open_positions = tuple(float(value) for value in default_pos[0, inspire_ids].tolist())
        mapper = InspireFTPGripMapper(dataset_channel_order_verified=True, right_thumb_yaw_closes_at_upper=True)
        standing_targets = default_pos.clone()
        standing_targets[:, body_ids] = torch.tensor(DEFAULT_ANGLES, device=default_pos.device).unsqueeze(0)
        standing_targets[:, inspire_ids] = torch.tensor(mapper.targets(V4Action((0.0,) * 64, (0.0,) * 7, (0.0,) * 7), hand_limits, hand_open_positions), device=default_pos.device).unsqueeze(0)
        # A free root is initialized in the standing posture before the timeline advances.
        robot.write_joint_state_to_sim(standing_targets, default_vel)
        robot.set_joint_position_target(standing_targets)
        standing_reference = _safe_reference()
        lifecycle = LifecycleGuard(watchdog_seconds=0.10)
        lifecycle.initialize(standing_reference)
        lifecycle.start(time.monotonic())
        bottle = scene["bottle"]
        right_hand_body_ids = [index for index, name in enumerate(robot.body_names) if name.startswith("R_") and ("hand" in name.lower() or "finger" in name.lower() or "thumb" in name.lower())]
        if not right_hand_body_ids:
            right_hand_body_ids = [index for index, name in enumerate(robot.body_names) if name.startswith("R_")]
        rollout_samples: list[dict[str, object]] = []
        video_frames: list[object] = []
        initial_bottle_z = float(bottle.data.root_pos_w[0, 2])
        stable_grasp_frames = 0
        events: list[str] = ["initial_pose", "gravity_settle", "hand_open"]
        if args.hand_sign_probe:
            closed = mapper.targets(V4Action((0.0,) * 64, (1.0,) * 7, (1.0,) * 7), hand_limits, hand_open_positions)
            print(json.dumps({"event": "right_thumb_sign_probe", "asset": "g1_29dof_inspire_hand.usd", "joint": "R_thumb_proximal_yaw_joint", "limits": dict(zip(INSPIRE_HAND_JOINTS, hand_limits)), "open_target": float(standing_targets[0, robot.joint_names.index("R_thumb_proximal_yaw_joint")]), "closed_target": closed[INSPIRE_HAND_JOINTS.index("R_thumb_proximal_yaw_joint")], "closure_direction": "lower_to_upper"}, sort_keys=True), flush=True)
            _exit_code[0] = 0
            return 0
        if args.closed_loop:
            import base64
            import importlib.util
            try:
                import zmq
            except ModuleNotFoundError:
                sys.path.insert(0, "/opt/venvs/sonic-sim/lib/python3.11/site-packages")
                import zmq
            from cloudwalk_closed_loop import BodyCommand, SonicState, native_decoder_state
            from sonic_isaac_inspire_adapter import ACTION_RATE_HZ, INFERENCE_RATE_HZ, split_groot_action_chunk

            module_spec = importlib.util.spec_from_file_location("cloudwalk_upstream_packer", ROOT / "scripts" / "sonic-isolated-vla-producer.py")
            packer_module = importlib.util.module_from_spec(module_spec)
            assert module_spec.loader is not None
            module_spec.loader.exec_module(packer_module)
            pack = packer_module.load_upstream_packer(Path(os.environ.get("SONIC_ROOT", "/opt/src/sonic")))
            context = zmq.Context()
            state_pub = context.socket(zmq.PUB); state_pub.setsockopt(zmq.LINGER, 0); state_pub.bind(args.state_endpoint)
            action_pub = context.socket(zmq.PUB); action_pub.setsockopt(zmq.LINGER, 0); action_pub.bind(args.action_endpoint)
            body_sub = context.socket(zmq.SUB); body_sub.setsockopt(zmq.LINGER, 0); body_sub.setsockopt(zmq.SUBSCRIBE, b""); body_sub.connect(args.body_endpoint)
            inference_requests: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=1)
            inference_results: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
            inference_stop = threading.Event()

            def inference_worker() -> None:
                worker_context = zmq.Context()
                worker_socket = worker_context.socket(zmq.REQ)
                worker_socket.setsockopt(zmq.LINGER, 0); worker_socket.setsockopt(zmq.RCVTIMEO, 30000); worker_socket.setsockopt(zmq.SNDTIMEO, 30000)
                worker_socket.connect(args.vla_endpoint)
                try:
                    while not inference_stop.is_set():
                        item = inference_requests.get()
                        if item is None:
                            return
                        worker_socket.send_json(item)
                        reply = worker_socket.recv_json()
                        reply["inference_started_monotonic"] = item["sent_monotonic"]
                        try:
                            inference_results.put_nowait(reply)
                        except queue.Full:
                            inference_results.get_nowait(); inference_results.put_nowait(reply)
                finally:
                    worker_socket.close(); worker_context.term()

            worker = threading.Thread(target=inference_worker, name="cloudwalk-groot", daemon=True)
            worker.start()
            time.sleep(0.25)
            chunk = None; chunk_index = 0; sequence = 0; last_command_sequence = -1; body_frames = 0; inference_frames = 0; warmup_frames = 10
            last_body = tuple(float(value) for value in standing_targets[0, body_ids].tolist())
            last_raw_action = (0.0,) * 29
            last_hand_action = standing_reference.action
            inference_pending = False; last_inference_started = float("-inf"); last_inference_tick = -10**9
            control_period = 1.0 / ACTION_RATE_HZ
            physics_steps_per_control = round(control_period / sim.get_physics_dt())
            if physics_steps_per_control < 1 or abs(physics_steps_per_control * sim.get_physics_dt() - control_period) > 1e-9:
                _fail("Isaac physics dt must divide the upstream SONIC 50 Hz control period")
            started_at = time.monotonic(); control_tick = 0; step = 0
            while step < args.steps:
                tick_started = time.monotonic()
                if not worker.is_alive() and inference_pending:
                    _fail("GR00T inference worker exited before returning an action")
                if chunk is None and tick_started - started_at > 60.0:
                    _fail("GR00T cold-start inference exceeded 60 seconds")
                body_q_mujoco = tuple(float(value) for value in robot.data.joint_pos[0, body_ids].tolist())
                body_qd_mujoco = tuple(float(value) for value in robot.data.joint_vel[0, body_ids].tolist())
                body_q, body_qd = native_decoder_state(body_q_mujoco, body_qd_mujoco)
                angular = tuple(float(value) for value in robot.data.root_ang_vel_b[0].tolist())
                gravity = tuple(float(value) for value in robot.data.projected_gravity_b[0].tolist())
                state_timestamp = time.monotonic_ns()
                state_pub.send(SonicState(sequence, state_timestamp, angular, body_q, body_qd, last_raw_action, gravity).pack())
                print(json.dumps({"event": "isaac_state_publish", "sequence": sequence, "monotonic_ns": state_timestamp, "warmup_remaining": warmup_frames}, sort_keys=True), flush=True)
                while True:
                    try:
                        received = body_sub.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    command = BodyCommand.unpack(received)
                    if command.sequence > last_command_sequence:
                        last_command_sequence = command.sequence; last_body = command.positions; last_raw_action = command.raw_actions; body_frames += 1
                        print(json.dumps({"event": "isaac_body_receive", "sequence": command.sequence, "monotonic_ns": command.monotonic_ns, "count": body_frames, "raw_action_linf": max(abs(value) for value in last_raw_action)}, sort_keys=True), flush=True)
                try:
                    reply = inference_results.get_nowait()
                except queue.Empty:
                    reply = None
                if reply is not None:
                    inference_pending = False
                    if not reply.get("ok"):
                        _fail(f"GR00T worker rejected live observation: {reply.get('error', 'unknown error')}")
                    chunk = split_groot_action_chunk(reply["actions"])
                    delay = tick_started - float(reply["inference_started_monotonic"])
                    chunk_index = min(round(delay * ACTION_RATE_HZ), len(chunk) - 1)
                    inference_frames += 1
                    print(json.dumps({"event": "isaac_groot_reply", "sequence": reply.get("sequence"), "latency_seconds": reply.get("latency_seconds"), "latency_compensated_chunk_index": chunk_index, "inference_frames": inference_frames}, sort_keys=True), flush=True)
                if warmup_frames:
                    warmup_frames -= 1
                elif not inference_pending and (chunk is None or control_tick - last_inference_tick >= round(ACTION_RATE_HZ / INFERENCE_RATE_HZ)):
                    rgb = camera.data.output["rgb"][0].cpu().numpy()
                    policy_state = list(body_q_mujoco[:22]) + list(last_hand_action.left_hand) + list(body_q_mujoco[22:]) + list(last_hand_action.right_hand)
                    validate_observation(rgb, policy_state, PROMPT)
                    inference_requests.put_nowait({"sequence": sequence, "sent_monotonic": tick_started, "timestamp": time.time(), "rgb": base64.b64encode(rgb.tobytes()).decode("ascii"), "state": policy_state, "base_quat": [float(value) for value in robot.data.root_quat_w[0].tolist()]})
                    inference_pending = True; last_inference_started = tick_started; last_inference_tick = control_tick
                    print(json.dumps({"event": "isaac_groot_request", "sequence": sequence, "monotonic_ns": time.monotonic_ns()}, sort_keys=True), flush=True)
                action = standing_reference.action if chunk is None else chunk[min(chunk_index, len(chunk) - 1)]
                last_hand_action = action
                action_timestamp = time.monotonic_ns()
                action_pub.send(pack(action.motion_token, __import__("numpy").array([sequence], dtype=__import__("numpy").int64), action.left_hand, action.right_hand))
                print(json.dumps({"event": "isaac_action_publish", "sequence": sequence, "monotonic_ns": action_timestamp, "chunk_index": (None if chunk is None else chunk_index), "source": ("standing_reference" if chunk is None else "groot")}, sort_keys=True), flush=True)
                if chunk is not None:
                    chunk_index = min(chunk_index + 1, len(chunk) - 1)
                sequence += 1
                targets = standing_targets.clone()
                targets[:, body_ids] = torch.tensor(last_body, device=targets.device).unsqueeze(0)
                targets[:, inspire_ids] = torch.tensor(mapper.targets(last_hand_action, hand_limits, hand_open_positions), device=targets.device).unsqueeze(0)
                robot.set_joint_position_target(targets)
                for _ in range(physics_steps_per_control):
                    scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
                bottle_pos = bottle.data.root_pos_w[0]
                hand_pos = robot.data.body_pos_w[0, right_hand_body_ids]
                distances = torch.linalg.vector_norm(hand_pos - bottle_pos, dim=1)
                closest_hand_index = int(torch.argmin(distances))
                distance = float(distances[closest_hand_index])
                closure = float(sum(last_hand_action.right_hand[:6]) / 6.0)
                contact_proxy = distance < 0.12
                stable_grasp_frames = stable_grasp_frames + 1 if contact_proxy and closure >= 0.6 else 0
                if chunk is not None:
                    rollout_samples.append({"step": step, "simulated_seconds": step * control_period, "wall_seconds": tick_started - started_at, "bottle_pos_w": [float(value) for value in bottle_pos.tolist()], "closest_hand_body": robot.body_names[right_hand_body_ids[closest_hand_index]], "closest_hand_pos_w": [float(value) for value in hand_pos[closest_hand_index].tolist()], "hand_object_distance_m": distance, "right_hand_closure": closure, "contact_proxy": contact_proxy, "stable_grasp_frames": stable_grasp_frames, "lift_m": float(bottle_pos[2]) - initial_bottle_z, "inference_frames": inference_frames, "native_body_frames": body_frames, "action_sequence": sequence, "chunk_index": chunk_index})
                    if args.video_path is not None and step % 5 == 0:
                        video_frames.append(camera.data.output["rgb"][0].cpu().numpy())
                    step += 1
                control_tick += 1
            inference_stop.set(); inference_requests.put(None); worker.join(timeout=35.0)
            if worker.is_alive():
                _fail("GR00T inference worker did not stop")
            stop_context = zmq.Context(); stop_socket = stop_context.socket(zmq.REQ)
            stop_socket.setsockopt(zmq.LINGER, 0); stop_socket.setsockopt(zmq.RCVTIMEO, 5000); stop_socket.connect(args.vla_endpoint)
            stop_socket.send_json({"op": "stop"}); stop_socket.recv_json(); stop_socket.close(); stop_context.term()
            state_pub.close(); action_pub.close(); body_sub.close(); context.term()
            if not inference_frames or not body_frames:
                _fail("closed loop did not receive both upstream GR00T and native SONIC body frames")
            events.extend(("live_rgb_state", "upstream_groot_policy", "upstream_v4_serializer", "native_sonic_29_body", "hands_24_joint_applied"))
            print(json.dumps({"event": "closed_loop", "inference_frames": inference_frames, "native_body_frames": body_frames, "hand_application": "verified_24_joint_normalized_mapper", "body_dofs": list(G1_BODY_JOINTS), "prompt": PROMPT, "embodiment": EMBODIMENT}, sort_keys=True), flush=True)
        elif args.interactive:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            timeline.stop()
            print(json.dumps({"event": "interactive_ready", "physics": "free_base_gravity_enabled", "instruction": "Press Timeline Play to begin PhysX stepping; do not alter stage topology after Play."}, sort_keys=True), flush=True)
            while simulation_app.is_running():
                if timeline.is_playing():
                    robot.set_joint_position_target(standing_targets)
                    scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
                else:
                    sim.render()
            events.append("interactive_scene")
        else:
            for _ in range(args.steps):
                robot.set_joint_position_target(standing_targets)
                scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
            events.append("scene_only")
        lifecycle.pause()
        lifecycle.stop()
        lifecycle.reset()
        scene.reset()
        robot.write_joint_state_to_sim(standing_targets, default_vel)
        robot.set_joint_position_target(standing_targets)
        events.append("reset")
        if not args.interactive:
            for _ in range(5):
                scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
        rgb = camera.data.output["rgb"][0].cpu().numpy()
        args.capture_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.capture_path, rgb)
        rollout_metrics = None
        if args.rollout_metrics_path is not None:
            args.rollout_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            max_stable = max((int(sample["stable_grasp_frames"]) for sample in rollout_samples), default=0)
            max_lift = max((float(sample["lift_m"]) for sample in rollout_samples), default=0.0)
            max_grasp_lift = max((float(sample["lift_m"]) for sample in rollout_samples if int(sample["stable_grasp_frames"]) >= 10), default=0.0)
            bottle_start = scene_config["bottle"]["position_m"]
            table_top = scene_config["table"]["position_m"][2] + scene_config["table"]["size_m"][2] / 2.0
            max_horizontal_displacement = max((math.hypot(float(sample["bottle_pos_w"][0]) - bottle_start[0], float(sample["bottle_pos_w"][1]) - bottle_start[1]) for sample in rollout_samples), default=0.0)
            bottle_fell = any(float(sample["bottle_pos_w"][2]) < table_top - 0.05 for sample in rollout_samples)
            rollout_metrics = {"initial_bottle_z_m": initial_bottle_z, "samples": rollout_samples, "simulated_duration_seconds": len(rollout_samples) / ACTION_RATE_HZ, "wall_duration_seconds": (float(rollout_samples[-1]["wall_seconds"]) if rollout_samples else 0.0), "inference_frames": inference_frames, "native_body_frames": body_frames, "approach": any(float(sample["hand_object_distance_m"]) < 0.20 for sample in rollout_samples), "contact_proxy": any(bool(sample["contact_proxy"]) for sample in rollout_samples), "hand_close": any(float(sample["right_hand_closure"]) >= 0.6 for sample in rollout_samples), "stable_grasp": max_stable >= 10, "stable_grasp_frames": max_stable, "lift": max_grasp_lift >= 0.05, "max_grasp_associated_lift_m": max_grasp_lift, "max_transient_lift_m": max_lift, "max_horizontal_displacement_m": max_horizontal_displacement, "bottle_fell": bottle_fell, "video": str(args.video_path) if args.video_path else None}
            args.rollout_metrics_path.write_text(json.dumps(rollout_metrics, sort_keys=True) + "\n")
        if args.video_path is not None and video_frames:
            args.video_path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(args.video_path, video_frames, fps=10, codec="libx264")
        result = {"asset_cfg": "CLOUDWALK_G1_INSPIRE_FTP_FREE_BASE_CFG", "usd": "Robots/Unitree/G1/g1_29dof_inspire_hand.usd", "joints": robot.num_joints, "bodies": robot.num_bodies, "camera_shape": list(rgb.shape), "capture": str(args.capture_path), "events": events, "prompt": PROMPT, "embodiment": EMBODIMENT, "scene_config": str(args.scene_config), "scene_parameters": {"table_height_m": scene_config["table"]["size_m"][2], "table_position": scene_config["table"]["position_m"], "bottle_position": scene_config["bottle"]["position_m"], "bottle_height_m": scene_config["bottle"]["height_m"], "camera_focal_length_mm": scene_config["camera"]["focal_length_mm"]}, "physics": {"root_fixed": False, "gravity_enabled": True, "stage_topology": "immutable_after_reset", "mode": "interactive" if args.interactive else "batch"}, "isaac_target_dofs": {"body": list(body_joint_names), "inspire": list(INSPIRE_HAND_JOINTS)}, "vla_connection": "upstream_policy_native_sonic_connected" if args.closed_loop else "not_connected", "hand_application": "verified_24_joint_normalized_mapper" if args.closed_loop else "scene_only", "rollout_metrics": rollout_metrics}
        # Flush before close(): Kit's teardown can discard buffered stdout.
        print(json.dumps(result, sort_keys=True), flush=True)
        _exit_code[0] = 0
    except ContractError as error:
        # __main__'s handler runs after close(), which can hard-exit the process first.
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        _exit_code[0] = 2
        raise
    except BaseException:
        # Same for a traceback: print it before Kit's teardown.
        import traceback
        traceback.print_exc()
        _exit_code[0] = 1
        raise
    finally:
        # Kit teardown can hang after a headless camera run; process exit releases it.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_exit_code[0])
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        _exit_code[0] = exit_code
    except ContractError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        _exit_code[0] = 2
    except BaseException:
        import traceback
        traceback.print_exc()
        _exit_code[0] = 1
    raise SystemExit(_exit_code[0])
