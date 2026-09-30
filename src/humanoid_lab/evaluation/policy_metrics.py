"""Policy comparisons in physical units and checkpoint-normalized space."""
import numpy as np


def action_metrics(predicted, reference, normalized_predicted, normalized_reference):
    predicted = np.asarray(predicted, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    count = min(len(predicted), len(reference))
    if not count:
        return {"reference_steps": 0}
    delta = predicted[:count] - reference[:count]
    normalized_delta = np.asarray(normalized_predicted)[:count] - np.asarray(normalized_reference)[:count]
    if not np.isfinite(delta).all() or not np.isfinite(normalized_delta).all():
        raise ValueError("Action metric contains nonfinite values")
    return {"reference_steps": count, "mae": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.square(delta).mean())),
            "normalized_mae": float(np.abs(normalized_delta).mean()),
            "normalized_rmse": float(np.sqrt(np.square(normalized_delta).mean())),
            "per_joint_mae": np.abs(delta).mean(axis=0).tolist(),
            "per_joint_rmse": np.sqrt(np.square(delta).mean(axis=0)).tolist()}


def video_metrics(predicted, reference):
    count = min(len(predicted), len(reference))
    if not count:
        return {"reference_frames": 0}
    delta = np.asarray(predicted[:count], dtype=np.float32) - np.asarray(reference[:count], dtype=np.float32)
    mse = float(np.square(delta).mean())
    return {"reference_frames": count, "mse": mse, "mae": float(np.abs(delta).mean()),
            "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
            "metric_domain": "uint8 RGB before MP4 compression"}


def feedback_state(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (32, 28) or not np.isfinite(actions).all():
        raise ValueError("Dream feedback requires 32×28 finite physical-unit actions")
    return actions[-1].copy()
