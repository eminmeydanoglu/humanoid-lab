"""G1/Dex3 inference core. A transport callback calls predict with a synchronized observation."""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np
import torch

from flux_action.data.lerobot.dex3_view import CAMERA

# Simulation and robot node accept the same task vocabulary. Cache its text contexts
# before readiness so CPU text encoding never delays a supported task's first chunk.
_TASKS = (
    "", "stack three block", "camera packaging", "object placement", "pour water", "toasted bread",
    "Put the apple into the plate.", "Put the bottle into the plate.",
    "Put the charger into the plate.", "Put the doll into the plate.",
    "Put the gum into the plate.", "Put the snack into the plate.",
    "Put the tissue paper into the plate.",
)


def _keep_text_encoder_on_cpu(base):
    """Skip Qwen when the PEFT policy moves its frozen components to the GPU."""
    frozen = base.frozen
    frozen.text_encoder.to("cpu")

    def apply_video_only(self, fn):
        video = getattr(self.video_vae, "module", self.video_vae)
        if isinstance(video, torch.nn.Module):
            video._apply(fn)

    frozen._apply = MethodType(apply_video_only, frozen)


def _offload_text_encoder(base, device):
    """Cache BF16 text contexts on the DiT device; Qwen stays on CPU."""
    from lerobot.policies.flux3.f3 import VEC_DIM, packing, text_context

    encoder = base.frozen.text_encoder
    encoder.to("cpu")

    def context(self, caption, target):
        hit = self._ctx_cache.get(caption)
        if hit is None or hit[0].device != target:
            if len(self._ctx_cache) > 256:
                self._ctx_cache.clear()
            encoded = text_context(encoder, caption, "cpu", fixed_length=self.config.text_fixed_length)
            encoded = encoded.to(device=target, dtype=self.dtype_)
            self._ctx_cache[caption] = (encoded, packing.pack_text(encoded, VEC_DIM)["ctx_ids"])
        return self._ctx_cache[caption]

    base._context = MethodType(context, base)
    with torch.inference_mode():
        for task in _TASKS:
            base._context(task, torch.device(device))
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


class G1Inference:
    def __init__(self, policy, pre, post, *, device: str = "cuda"):
        cfg = policy.config
        if (
            cfg.camera_order != [CAMERA]
            or cfg.action_dim != 28
            or cfg.chunk_size != 32
            or cfg.n_obs_steps != 1
            or cfg.action_representation != "absolute"
        ):
            raise ValueError("checkpoint is not the G1/Dex3 single-camera 28D policy")
        canvas = cfg.canvas_hw
        if (not isinstance(canvas, (list, tuple)) or len(canvas) != 2
                or any(type(size) is not int or size <= 0 for size in canvas)):
            raise ValueError("checkpoint canvas_hw must contain two positive integer dimensions")
        self.image_hw = tuple(canvas)
        self.policy, self.pre, self.post = policy, pre, post
        self.device = torch.device(device)
        self.policy.eval()
        self.reset()

    @classmethod
    def load(cls, checkpoint: str | Path, *, device: str = "cuda", merge_adapter: bool = False,
             offload_text_encoder: bool = True) -> G1Inference:
        """Load the raw LoRA adapter and its checkpoint-owned normalization once.

        ``merge_adapter`` bakes the LoRA delta into the base weights in memory (the checkpoint on
        disk stays untouched). Identical math up to bf16 rounding; removes the per-linear adapter
        matmuls from sampling (~22% faster at the 4-step, two-pass-CFG default).
        """
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
        if offload_text_encoder:
            _keep_text_encoder_on_cpu(base)
        policy.to(device)
        if merge_adapter:
            policy.merge_adapter()
        if offload_text_encoder:
            _offload_text_encoder(base, device)
        return cls(policy, pre, post, device=device)

    def reset(self) -> None:
        """Call on an episode boundary, before accepting its first observation."""
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def predict(self, image: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        """RGB HWC uint8 + 28 measured joints -> 32x28 absolute commands."""
        image = np.asarray(image)
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
                or any(size <= 0 for size in image.shape[:2])):
            raise ValueError("image must be nonempty RGB uint8 HWC")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (28,) or not np.isfinite(state).all():
            raise ValueError("state must be 28 finite values in Dex3 joint order")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty instruction")
        if image.shape[:2] != self.image_hw:
            import av

            image = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24").to_ndarray(
                format="rgb24", width=self.image_hw[1], height=self.image_hw[0]
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
