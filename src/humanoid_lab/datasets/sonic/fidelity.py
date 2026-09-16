"""Reference (A), SONIC joint command (B), and measured robot (C) fidelity.

The publisher and simulator share a wall clock.  Each simulator sample is paired
with the latest token sent at that instant.  This is a transport-time estimate,
not proof of which token the C++ policy consumed on that exact control tick.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from humanoid_lab.controllers.sonic import BODY_JOINT_ORDER, SONIC_REFERENCE_JOINT_ORDER

from .joints import reorder


# Gross-fidelity gates, deliberately independent of the existing B-to-C PD gate.
# These are provisional pilot limits; preserve failing measurements while
# evaluating whether the policy's q setpoint is an appropriate fidelity target.
FIDELITY_LIMITS = {
    "reference_to_command.body_mae_rad": 0.35,
    "reference_to_command.body_worst_joint_mae_rad": 0.50,
    "reference_to_command.left_arm_mae_rad": 0.45,
    "reference_to_command.left_arm_worst_joint_mae_rad": 0.45,
    "reference_to_response.body_mae_rad": 0.45,
    "reference_to_response.body_worst_joint_mae_rad": 0.60,
    "reference_to_response.left_arm_mae_rad": 0.55,
    "reference_to_response.left_arm_worst_joint_mae_rad": 0.55,
    "command_to_response.body_mae_rad": 0.35,
}
LEFT_ARM_NAMES = tuple(name for name in BODY_JOINT_ORDER if name.startswith(("left_shoulder", "left_elbow", "left_wrist")))
DEFAULT_MODEL_XML = Path("/opt/src/sonic/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml")


def simulation_frame_due(current_sim_s: float, start_sim_s: float, frame_index: int) -> bool:
    """Handle 20 ms clock quantization without skipping a frame on float error."""
    return current_sim_s + 1e-6 >= start_sim_s + frame_index / 50.0


def left_wrist_positions(body_positions: np.ndarray, model_xml: Path) -> np.ndarray:
    """Pelvis-local left wrist position from the pinned G1 MuJoCo geometry."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0  # neutral free-root quaternion
    addresses = []
    for name in BODY_JOINT_ORDER:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"MuJoCo model has no joint {name}")
        addresses.append(model.jnt_qposadr[joint])
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    result = np.empty((len(body_positions), 3), dtype=np.float64)
    for row, values in enumerate(body_positions):
        data.qpos[addresses] = values
        mujoco.mj_forward(model, data)
        result[row] = data.xpos[wrist] - data.xpos[pelvis]
    return result


def root_local_to_world(local: np.ndarray, root_pos: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    """Apply a wxyz root pose to local wrist points, row by row."""
    quaternion = np.asarray(root_quat_wxyz, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    vector = quaternion[:, 1:]
    point = np.asarray(local, dtype=np.float64)
    crossed = np.cross(vector, point)
    rotated = point + 2.0 * (quaternion[:, :1] * crossed + np.cross(vector, crossed))
    return np.asarray(root_pos, dtype=np.float64) + rotated


def _errors(first: np.ndarray, second: np.ndarray, names: tuple[str, ...]) -> dict:
    difference = np.abs(first - second)
    per_joint = {
        name: {"mae_rad": float(difference[:, j].mean()), "p95_rad": float(np.percentile(difference[:, j], 95))}
        for j, name in enumerate(names)
    }
    result = {
        "body_mae_rad": float(difference.mean()),
        "body_p95_rad": float(np.percentile(difference, 95)),
        "body_worst_joint_mae_rad": float(difference.mean(axis=0).max()),
        "per_joint": per_joint,
    }
    if names == BODY_JOINT_ORDER:
        indices = [names.index(name) for name in LEFT_ARM_NAMES]
        result["left_arm_mae_rad"] = float(difference[:, indices].mean())
        result["left_arm_p95_rad"] = float(np.percentile(difference[:, indices], 95))
        result["left_arm_worst_joint_mae_rad"] = float(difference[:, indices].mean(axis=0).max())
    return result


def compare_reference_command_response(
    reference: dict, tracking: dict, timeline: dict, *, model_xml: Path | None = None
) -> dict:
    """Score only samples inside the actual token stream and after support release.

    Missing timing or incomplete replay is an explicit failure, never an
    apparently good error computed from a short prefix.
    """
    sent = np.asarray(timeline["sent_wall_time_ns"], dtype=np.int64)
    frames = np.asarray(timeline["frame_index"], dtype=np.int64)
    if len(sent) < 2 or len(sent) != len(frames) or np.any(np.diff(sent) <= 0):
        raise ValueError("publisher timeline must contain increasing send times for every frame")
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("publisher frame indices must cover the complete reference")
    body = reorder(np.asarray(reference["joint_pos"]), SONIC_REFERENCE_JOINT_ORDER, BODY_JOINT_ORDER)
    if len(body) != len(frames):
        raise ValueError("reference and published token frame counts differ")
    sent_sim = np.asarray(timeline.get("sent_sim_s", []), dtype=np.float64)
    use_sim_clock = len(sent_sim) == len(sent) and np.isfinite(sent_sim).all()
    if use_sim_clock:
        if np.any(np.diff(sent_sim) <= 0):
            raise ValueError("published simulation times must increase")
        trace_time = np.asarray(tracking["sim_s"], dtype=np.float64)
        send_time = sent_sim
        tail_tolerance = 0.1
    else:
        wall = np.asarray(tracking.get("wall_time_ns", []), dtype=np.int64)
        if len(wall) != len(tracking["sim_s"]):
            raise ValueError("tracking trace lacks wall_time_ns; rerun with the timed recorder")
        if np.any(np.diff(wall) <= 0):
            raise ValueError("tracking wall times must increase")
        trace_time = wall
        send_time = sent
        tail_tolerance = 100_000_000
    index = np.searchsorted(send_time, trace_time, side="right") - 1
    in_stream = (index >= 0) & (trace_time <= send_time[-1] + tail_tolerance)
    free = ~np.asarray(tracking["support_active"], dtype=bool)
    mask = in_stream & free
    if not np.any(mask):
        raise ValueError("no free robot samples overlap the token stream")
    indices = index[mask]
    seen = np.unique(indices)
    coverage = {
        "expected_frames": int(len(frames)),
        "first_observed_frame": int(seen[0]),
        "last_observed_frame": int(seen[-1]),
        "observed_unique_frames": int(len(seen)),
        "observed_fraction": float(len(seen) / len(frames)),
        "trace_extends_past_final_send": bool(trace_time[-1] >= send_time[-1]),
        "clip_start_sim_s": float(np.asarray(tracking["sim_s"])[mask][0]),
        "clip_end_sim_s": float(np.asarray(tracking["sim_s"])[mask][-1]),
        "full_motion": bool(seen[0] <= 1 and seen[-1] >= len(frames) - 2
                            and len(seen) / len(frames) >= 0.95 and trace_time[-1] >= send_time[-1]),
        "alignment": "latest publisher send by Isaac simulation clock" if use_sim_clock else
                     "latest publisher send by shared wall clock; transport latency is not measured",
    }
    command = np.asarray(tracking["body_target"])[mask]
    measured = np.asarray(tracking["body_measured"])[mask]
    desired = body[indices]
    pairs = {
        "reference_to_command": _errors(desired, command, BODY_JOINT_ORDER),
        "reference_to_response": _errors(desired, measured, BODY_JOINT_ORDER),
        "command_to_response": _errors(command, measured, BODY_JOINT_ORDER),
    }
    hands = {}
    for side in ("left", "right"):
        desired_hand = np.asarray(reference[f"{side}_hand_joints"])[indices]
        command_hand = np.asarray(tracking[f"{side}_hand_target"])[mask]
        measured_hand = np.asarray(tracking[f"{side}_hand_measured"])[mask]
        names = tuple(f"{side}_hand_{j}" for j in range(7))
        hands[side] = {
            "reference_to_command": _errors(desired_hand, command_hand, names),
            "reference_to_response": _errors(desired_hand, measured_hand, names),
            "command_to_response": _errors(command_hand, measured_hand, names),
        }
    decisions = []
    for pair in ("reference_to_command", "reference_to_response"):
        value = hands["left"][pair]["body_mae_rad"]
        decisions.append({"metric": f"{pair}.left_hand_mae_rad", "value": value, "limit": 0.35,
                          "result": "PASS" if value <= 0.35 else "FAIL"})
    for metric, limit in FIDELITY_LIMITS.items():
        pair, field = metric.split(".")
        value = pairs[pair][field]
        decisions.append({"metric": metric, "value": value, "limit": limit, "result": "PASS" if value <= limit else "FAIL"})
    decisions.insert(0, {"metric": "coverage.full_motion", "value": float(coverage["full_motion"]), "limit": 1.0,
                         "result": "PASS" if coverage["full_motion"] else "FAIL"})
    wrist = None
    world_wrist = None
    if model_xml is not None:
        paths = {"reference": left_wrist_positions(desired, model_xml),
                 "command": left_wrist_positions(command, model_xml),
                 "response": left_wrist_positions(measured, model_xml)}
        wrist = {"frame": "pelvis-local", "link": "left_wrist_yaw_link", "model_xml": str(model_xml)}
        for pair, key in (("reference_to_command", "command"), ("reference_to_response", "response")):
            displacement = paths[key] - paths["reference"]
            value = float(np.linalg.norm(displacement, axis=1).mean())
            limit = 0.15 if key == "command" else 0.18
            wrist[pair] = {
                "path_mae_m": value,
                "reference_delta_y_m": float(paths["reference"][-1, 1] - paths["reference"][0, 1]),
                "other_delta_y_m": float(paths[key][-1, 1] - paths[key][0, 1]),
            }
            decisions.append({"metric": f"{pair}.left_wrist_path_mae_m", "value": value, "limit": limit,
                              "result": "PASS" if value <= limit else "FAIL"})
        # A direction gate is meaningful only for a substantial net lateral move.
        reference_y = wrist["reference_to_command"]["reference_delta_y_m"]
        if abs(reference_y) >= 0.08 and coverage["full_motion"]:
            for pair in ("reference_to_command", "reference_to_response"):
                other_y = wrist[pair]["other_delta_y_m"]
                same_direction = reference_y * other_y > 0
                decisions.append({"metric": f"{pair}.left_wrist_lateral_direction", "value": float(same_direction),
                                  "limit": 1.0, "result": "PASS" if same_direction else "FAIL"})
        reference_world = root_local_to_world(
            paths["reference"], np.asarray(reference["body_pos"])[indices],
            np.asarray(reference["body_quat_wxyz"])[indices],
        )
        actual_root = np.asarray(tracking["root_position"])[mask]
        actual_quat = np.asarray(tracking["root_quaternion_wxyz"])[mask]
        world_paths = {"reference": reference_world,
                       "command": root_local_to_world(paths["command"], actual_root, actual_quat),
                       "response": root_local_to_world(paths["response"], actual_root, actual_quat)}
        world_wrist = {"frame": "world", "root_reference": "canonical episode root",
                       "root_command_response": "measured Isaac robot root"}
        for pair, key in (("reference_to_command", "command"), ("reference_to_response", "response")):
            value = float(np.linalg.norm(world_paths[key] - reference_world, axis=1).mean())
            delta_a = float(reference_world[-1, 1] - reference_world[0, 1])
            delta_other = float(world_paths[key][-1, 1] - world_paths[key][0, 1])
            world_wrist[pair] = {"path_mae_m": value, "reference_delta_y_m": delta_a,
                                "other_delta_y_m": delta_other}
            limit = 0.20 if key == "command" else 0.22
            decisions.append({"metric": f"{pair}.world_left_wrist_path_mae_m", "value": value, "limit": limit,
                              "result": "PASS" if value <= limit else "FAIL"})
            if abs(delta_a) >= 0.08 and coverage["full_motion"]:
                same_direction = delta_a * delta_other > 0
                decisions.append({"metric": f"{pair}.world_left_wrist_lateral_direction", "value": float(same_direction),
                                  "limit": 1.0, "result": "PASS" if same_direction else "FAIL"})
    action_components = None
    required = ("body_velocity_target", "body_feedforward_torque", "body_kp", "body_kd",
                "body_applied_torque", "body_measured_velocity")
    if all(name in tracking for name in required):
        arm_indices = [BODY_JOINT_ORDER.index(name) for name in LEFT_ARM_NAMES]
        position_term = np.asarray(tracking["body_kp"])[mask] * (command - measured)
        velocity_term = np.asarray(tracking["body_kd"])[mask] * (
            np.asarray(tracking["body_velocity_target"])[mask] - np.asarray(tracking["body_measured_velocity"])[mask]
        )
        feedforward = np.asarray(tracking["body_feedforward_torque"])[mask]
        applied = np.asarray(tracking["body_applied_torque"])[mask]
        action_components = {
            "left_arm_mean_abs_position_term_nm": float(np.abs(position_term[:, arm_indices]).mean()),
            "left_arm_mean_abs_velocity_term_nm": float(np.abs(velocity_term[:, arm_indices]).mean()),
            "left_arm_mean_abs_feedforward_nm": float(np.abs(feedforward[:, arm_indices]).mean()),
            "left_arm_mean_abs_applied_nm": float(np.abs(applied[:, arm_indices]).mean()),
            "left_arm_mean_abs_torque_reconstruction_error_nm": float(
                np.abs(position_term[:, arm_indices] + velocity_term[:, arm_indices]
                       + feedforward[:, arm_indices] - applied[:, arm_indices]).mean()
            ),
        }
    return {
        "coverage": coverage,
        "sample_count": int(mask.sum()),
        "pairs": pairs,
        "hands": hands,
        "left_wrist": wrist,
        "world_left_wrist": world_wrist,
        "sonic_action_components": action_components,
        "decisions": decisions,
        "result": "PASS" if all(item["result"] == "PASS" for item in decisions) else "FAIL",
        "plot_data": {
            "elapsed_s": (trace_time[mask] - send_time[0]).astype(np.float64) * (1.0 if use_sim_clock else 1e-9),
            "reference": desired,
            "command": command,
            "response": measured,
            "reference_left_hand": np.asarray(reference["left_hand_joints"])[indices],
            "command_left_hand": np.asarray(tracking["left_hand_target"])[mask],
            "response_left_hand": np.asarray(tracking["left_hand_measured"])[mask],
            **({"reference_wrist": paths["reference"], "command_wrist": paths["command"],
                "response_wrist": paths["response"]} if wrist is not None else {}),
            **({"reference_world_wrist": world_paths["reference"], "command_world_wrist": world_paths["command"],
                "response_world_wrist": world_paths["response"]} if world_wrist is not None else {}),
        },
    }


def load_timeline(path: Path) -> dict:
    with np.load(path) as payload:
        return {name: payload[name] for name in payload.files}
