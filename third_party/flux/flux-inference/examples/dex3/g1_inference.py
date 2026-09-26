"""G1/Dex3 inference core. A transport callback calls predict with a synchronized observation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from flux_action.data.lerobot.dex3_view import CAMERA


class G1Inference:
    def __init__(self, policy, pre, post, *, device: str = "cuda"):
        cfg = policy.config
        if (
            cfg.camera_order != [CAMERA]
            or cfg.action_dim != 28
            or cfg.chunk_size != 32
            or cfg.n_obs_steps != 1
            or tuple(cfg.canvas_hw) != (192, 256)
            or cfg.action_representation != "absolute"
        ):
            raise ValueError("checkpoint is not the G1/Dex3 single-camera 28D policy")
        self.policy, self.pre, self.post = policy, pre, post
        self.device = torch.device(device)
        self.policy.eval()
        self.reset()

    @classmethod
    def load(cls, checkpoint: str | Path, *, device: str = "cuda") -> G1Inference:
        """Load the raw LoRA adapter and its checkpoint-owned normalization once."""
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.flux3.modeling_flux3 import Flux3Policy
        from peft import PeftConfig, PeftModel

        checkpoint = Path(checkpoint).resolve()
        adapter = PeftConfig.from_pretrained(str(checkpoint))
        if not adapter.base_model_name_or_path:
            raise ValueError("adapter has no base policy")
        base = Flux3Policy.from_pretrained(adapter.base_model_name_or_path, strict=True)
        pre, post = make_pre_post_processors(base.config, pretrained_path=checkpoint)
        policy = PeftModel.from_pretrained(base, str(checkpoint), config=adapter, is_trainable=False)
        policy.to(device)
        return cls(policy, pre, post, device=device)

    def reset(self) -> None:
        """Call on an episode boundary, before accepting its first observation."""
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def predict(self, image: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        """RGB HWC uint8 + 28 measured joints -> 32x28 absolute commands."""
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.shape not in ((480, 640, 3), (192, 256, 3)):
            raise ValueError("image must be RGB uint8 HWC at 480x640 or 192x256")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (28,) or not np.isfinite(state).all():
            raise ValueError("state must be 28 finite values in Dex3 joint order")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty instruction")
        if image.shape[:2] == (480, 640):
            import av

            image = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24").to_ndarray(
                format="rgb24", width=256, height=192
            )
        observation = {
            CAMERA: torch.from_numpy(np.array(image, copy=True, order="C")).permute(2, 0, 1).float() / 255,
            "observation.state": torch.from_numpy(state.copy()),
            "task": task,
        }
        with torch.inference_mode():
            batch = self.pre(observation)
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            commands = self.post(self.policy.predict_action_chunk(batch))
        commands = commands.detach().cpu().numpy().astype(np.float32)
        if commands.shape != (1, 32, 28) or not np.isfinite(commands).all():
            raise ValueError("policy returned an invalid action chunk")
        return commands[0]
