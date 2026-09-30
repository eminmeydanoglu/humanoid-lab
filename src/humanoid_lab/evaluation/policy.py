"""G1 deploy inference with the jointly sampled video retained for evaluation.

Use the Flux model service interpreter. Calls are serialized by the GPU worker;
this adapter does not own a thread pool or change upstream modules.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
from pathlib import Path
import sys
import time
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE_POLICY = Path("/home/aksoy-lab/code/flux-training/flux-action/outputs/dex3/g1-base-policy")
FT_CHECKPOINT = ROOT / "data/models/flux-dex3/checkpoint-2500"
DEPLOY = {
    "conditioning": "history", "n_obs_steps": 1, "history_snapshots": 1,
    "canvas_hw": (192, 256), "dtype": "bfloat16", "action_dim": 28,
    "chunk_size": 32, "action_representation": "absolute", "action_scale": 2.0,
    "sampler": "cosmos_unipc", "num_inference_steps": 4, "guidance_scale": 4.0,
    "guidance_scale_action": 1.0, "sampler_shift": 5.0, "inference_seed": 0,
    "fps": 30.0, "video_position_fps": 24.0, "attn_mode": "torch",
    "compile_model": False, "condition_on_past_actions": False,
}


def _validate_config(cfg):
    for key, expected in DEPLOY.items():
        actual = getattr(cfg, key)
        if key == "canvas_hw":
            actual = tuple(actual)
        if actual != expected:
            raise ValueError(f"G1 deploy config {key}: {actual!r} != {expected!r}")
    if cfg.gripper_flip_dims:
        raise ValueError("G1 deploy must not flip joint channels")


def _array(tensor):
    return tensor.detach().float().cpu().numpy().copy()


def _unpack_video(tokens, frames, hw):
    b, _, c = tokens.shape
    return tokens.reshape(b, frames, *hw, c).permute(0, 4, 1, 2, 3).contiguous()


def _decode(vae, observed, future):
    import torch

    latents = torch.cat([observed, future], dim=2)
    if latents.shape[2] != 9:
        raise ValueError("G1 decode requires one snapshot and eight future latents")
    pixels = vae.decode(latents.to(torch.bfloat16))
    if tuple(pixels.shape) != (1, 3, 33, 192, 256) or not torch.isfinite(pixels).all():
        raise ValueError(f"invalid 33-frame VAE decode: {tuple(pixels.shape)}")
    clipping = {"rgb_out_of_range": int(((pixels < -1) | (pixels > 1)).sum().item())}
    rgb = pixels[0, :, 1:].permute(1, 2, 3, 0).float()
    rgb = rgb.add(1).mul(127.5).clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    return rgb, latents, clipping


class FluxJointPolicy:
    """One raw PEFT policy serves FT and original-base comparisons serially."""

    def __init__(self, runner, checkpoint=FT_CHECKPOINT, base_path=BASE_POLICY, log=None):
        self.runner = runner
        self.base = runner.policy.get_base_model()
        _validate_config(self.base.config)
        self.checkpoint, self.base_path = Path(checkpoint), Path(base_path)
        self.log = log or (lambda stage, message: None)
        self.cache_misses = []
        self._install_cache_logging()

    @classmethod
    def load(cls, checkpoint=FT_CHECKPOINT, *, device="cuda", base_path=BASE_POLICY, log=None):
        checkpoint = Path(checkpoint).resolve()
        adapter = json.loads((checkpoint / "adapter_config.json").read_text())
        if Path(adapter["base_model_name_or_path"]).resolve() != Path(base_path).resolve():
            raise ValueError("raw adapter does not reference the requested original base policy")
        saved = json.loads((Path(base_path) / "config.json").read_text())
        for key, expected in DEPLOY.items():
            actual = saved["output_features"]["action"]["shape"][0] if key == "action_dim" else saved[key]
            if key == "canvas_hw":
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(f"saved G1 deploy config mismatch: {key}")
        for path in (ROOT / "third_party/flux/flux-inference", ROOT / "third_party/flux/flux-training/src"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from examples.dex3.g1_inference import G1Inference

        runner = G1Inference.load(checkpoint, device=device, merge_adapter=False, offload_text_encoder=True)
        instance = cls(runner, checkpoint, base_path, log)
        instance.assert_head_restoration()
        instance.log("model", f"raw LoRA loaded: {checkpoint}; text encoder=CPU; deploy={DEPLOY}")
        return instance

    def _install_cache_logging(self):
        original = self.base._context
        owner = self

        def context(base, caption, device):
            hit = base._ctx_cache.get(caption)
            if hit is None or hit[0].device != device:
                owner.cache_misses.append(caption)
                owner.log("text", f"CPU text cache miss: {caption!r}")
            result = original(caption, device)
            while len(base._ctx_cache) > 32:
                oldest = next(key for key in base._ctx_cache if key not in ("", caption))
                base._ctx_cache.pop(oldest)
            return result

        self.base._context = MethodType(context, self.base)

    def _heads(self):
        from peft.utils.other import ModulesToSaveWrapper

        heads = {name: module for name, module in self.base.named_modules()
                 if isinstance(module, ModulesToSaveWrapper)}
        expected = set(self.runner.policy.peft_config["default"].modules_to_save)
        if set(heads) != expected or len(heads) != 4:
            raise AssertionError(f"G1 saved-head wrappers mismatch: {set(heads)} != {expected}")
        return heads

    @contextmanager
    def _variant(self, variant):
        if variant not in ("base", "ft"):
            raise ValueError("variant must be 'base' or 'ft'")
        heads = self._heads()
        if any(h.disable_adapters or h.active_adapters != ["default"] for h in heads.values()):
            raise AssertionError("FT heads are not active before prediction")
        if variant == "ft":
            yield
            return
        try:
            with self.runner.policy.disable_adapter():
                if not all(h.disable_adapters for h in heads.values()):
                    raise AssertionError("disable_adapter did not select original saved heads")
                yield
        finally:
            if any(h.disable_adapters or h.active_adapters != ["default"] for h in heads.values()):
                raise AssertionError("FT saved heads were not restored")

    def assert_head_restoration(self):
        """Verify original and FT heads against their actual checkpoint tensors."""
        import torch
        from safetensors import safe_open

        heads = self._heads()
        checked = 0
        with safe_open(str(self.base_path / "model.safetensors"), framework="pt", device="cpu") as base_file, \
             safe_open(str(self.checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu") as ft_file:
            for name, wrapper in heads.items():
                for key, tensor in wrapper.original_module.state_dict().items():
                    expected = base_file.get_tensor(f"{name}.{key}").to(tensor.dtype)
                    torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)
                    ft = wrapper.modules_to_save["default"].state_dict()[key]
                    expected = ft_file.get_tensor(f"base_model.model.{name}.{key}").to(ft.dtype)
                    torch.testing.assert_close(ft.cpu(), expected, rtol=0, atol=0)
                    checked += 1
        with self._variant("base"):
            pass
        return {"head_wrappers": len(heads), "head_tensors_verified": checked, "ft_restored": True}

    def reset(self):
        if self.runner is None:
            raise RuntimeError("policy is unloaded")
        self.runner.reset()

    def _batch(self, image, state, task):
        import torch

        image, state = np.asarray(image), np.asarray(state, dtype=np.float32)
        if image.dtype != np.uint8 or image.shape not in ((480, 640, 3), (192, 256, 3)):
            raise ValueError("image must be RGB uint8 HWC at 480x640 or 192x256")
        if state.shape != (28,) or not np.isfinite(state).all():
            raise ValueError("state must contain 28 finite Dex3 joints")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty instruction")
        if image.shape[:2] == (480, 640):
            import av
            image = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24").to_ndarray(
                format="rgb24", width=256, height=192)
        camera = self.base.config.camera_order[0]
        observation = {camera: torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255,
                       "observation.state": torch.from_numpy(state.copy()), "task": task}
        batch = self.runner.pre(observation)
        return {k: v.to(self.runner.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _sync(self):
        if self.runner.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.runner.device)

    def predict(self, image, state, task, *, variant="ft", seed=0):
        """Return CPU arrays; actions are absolute, not clipped to motor limits."""
        import torch
        from lerobot.policies.flux3.f3 import packing

        if self.runner is None:
            raise RuntimeError("policy is unloaded")
        if variant not in ("base", "ft"):
            raise ValueError("variant must be 'base' or 'ft'")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        timings = {}
        self._sync()
        started = tick = time.perf_counter()
        misses = len(self.cache_misses)
        with torch.inference_mode(), self._variant(variant):
            batch = self._batch(image, state, task)
            normalized_state, past = self.base._conditioning_inputs(batch)
            normalized_state = normalized_state.clone()
            self._sync()
            timings["preprocess_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            cfg = self.base.config
            cams = self.base._cameras(batch)
            canvas = packing.materialize_video(cams[0], None, normalized_state.device,
                                              layout=cfg.camera_layout, canvas_hw=cfg.canvas_hw)
            cond = self.base.packer.pack_video(cfg, self.base.frozen.video_vae, canvas[None], targets=False)
            cond.update(self.base.packer.pack_actions(cfg, normalized_state, past, targets=False))
            self._sync()
            timings["encode_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            caption = self.base._captions(batch, normalized_state.shape[0])[0]
            final = _sample_joint(self.base, cond, caption, seed)
            normalized = self.base._flip(final[f"x_{self.base.modality}"] / cfg.action_scale).float()
            commands = self.runner.post(normalized.clone())
            self._sync()
            timings["sample_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            observed = _unpack_video(cond["x_video_cond"], 1, cfg.latent_hw)
            future = _unpack_video(final["x_video"], 8, cfg.latent_hw)
            rgb, latents, clipping = _decode(self.base.frozen.video_vae, observed, future)
            self._sync()
            timings["decode_seconds"] = time.perf_counter() - tick
            actions = _array(commands)[0]
            if actions.shape != (32, 28) or not np.isfinite(actions).all():
                raise ValueError("invalid 32x28 postprocessed action chunk")
            limit = cfg.normalization_clip
            clipping.update(normalized_state_at_clip=int((normalized_state.abs() >= limit).sum().item()),
                            normalized_actions_outside_clip=int((normalized.abs() > limit).sum().item()),
                            normalization_clip=limit, actions_clipped=False)
            result = {"frames": rgb, "actions": actions, "normalized_actions": _array(normalized)[0],
                      "state": np.asarray(state, dtype=np.float32).copy(),
                      "normalized_state": _array(normalized_state)[0], "latents": _array(latents),
                      "metadata": {**DEPLOY, "variant": variant, "seed": seed, "task": task,
                                   "checkpoint": str(self.checkpoint), "base_policy": str(self.base_path),
                                   "latent_shape": list(latents.shape), "future_latent_shape": list(future.shape),
                                   "conditioning_token_shapes": {k: list(v.shape) for k, v in cond.items() if torch.is_tensor(v)},
                                   "position_ids": {k: _array(v).tolist() for k, v in cond.items() if k.endswith("_ids")},
                                   "vae_implementation": type(self.base.frozen.video_vae).__module__ + "." + type(self.base.frozen.video_vae).__name__,
                                   "vae_weights": str(self.base.config.video_vae_id), "decoder_max_t": 8,
                                   "decoded_frames": 33, "dropped_frames": 1, "text_cache_misses": self.cache_misses[misses:],
                                   "latent_normalization": "KinoVAE normalized", "adapter_merged": False},
                      "timings": timings, "clipping": clipping}
        timings["total_seconds"] = time.perf_counter() - started
        return result

    def unload(self):
        """Release the policy before handing the GPU worker to a standalone VAE."""
        import torch

        device = self.runner.device if self.runner is not None else None
        if self.base is not None:
            self.base._ctx_cache.clear()
            self.base._context = None
        self.runner = self.base = None
        gc.collect()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def gpu_parity_probe(self, image, state, task, *, seed=0, atol=1e-5):
        """Explicitly invoked only after the manager grants exclusive GPU ownership."""
        import torch

        if self.runner.device.type != "cuda":
            raise ValueError("parity probe requires GPU ownership and a CUDA-loaded policy")
        report = self.assert_head_restoration()
        old_seed = self.base.config.inference_seed
        variants = {}
        try:
            self.base.config.inference_seed = seed
            for variant in ("ft", "base"):
                self.reset()
                with torch.inference_mode(), self._variant(variant):
                    batch = self._batch(image, state, task)
                    direct = self.runner.policy.predict_action_chunk(batch)
                    direct_actions = _array(self.runner.post(direct.clone()))[0]
                    direct = _array(direct)[0]
                self.reset()
                joint = self.predict(image, state, task, variant=variant, seed=seed)
                np.testing.assert_allclose(joint["normalized_actions"], direct, rtol=0, atol=atol)
                np.testing.assert_allclose(joint["actions"], direct_actions, rtol=0, atol=atol)
                assert joint["frames"].shape == (32, 192, 256, 3)
                assert joint["frames"].dtype == np.uint8
                variants[variant] = {"normalized_max_abs_error": float(np.max(np.abs(direct - joint["normalized_actions"]))),
                                     "actions_max_abs_error": float(np.max(np.abs(direct_actions - joint["actions"]))),
                                     "frame_shape": list(joint["frames"].shape), "timings": joint["timings"]}
        finally:
            self.base.config.inference_seed = old_seed
            self.reset()
        return {**report, "variants": variants, "seed": seed, "atol": atol}


def unload_policy(policy):
    """Manager hook for switching the shared GPU worker to VAE reconstruction."""
    if policy is not None:
        policy.unload()


# Sampling adapted from LeRobot Flux3 (Black Forest Labs, Apache-2.0).
def _sample_joint(self, cond, caption, seed):
    """Deploy solver, retaining both final streams instead of discarding video."""
    import torch
    from lerobot.policies.flux3.f3 import (
        VEC_DIM, packing, sampling, batched_prc_vid, batched_prc_audio, times_to_ids,
    )
    cfg, m, mdt = self.config, self.modality, self.dtype_
    ak, ck = f"x_{m}", f"x_{m}_cond"
    device = cond["x_video_cond"].device
    n_pred = self.packer.predicted_latent_frames(cfg)
    rng = torch.Generator().manual_seed(seed)
    video_noise = torch.randn(1, packing.LATENT_CHANNELS, n_pred, *cfg.latent_hw, generator=rng)
    x_video, x_video_ids = batched_prc_vid(
        video_noise,
        self.packer.predicted_video_times(cfg, 1),
    )
    times = self.packer.action_times(cfg, 1)
    action_noise = torch.randn(1, cfg.action_dim, cfg.chunk_size, generator=rng)
    x_action, x_action_ids = batched_prc_audio(action_noise, times_to_ids(times))
    # The solver state stays fp32 (scaled joint targets would lose ~0.01 rad per bf16 round trip);
    # inputs are cast at the model boundary.
    flow = {"x_video": x_video.to(device), ak: x_action.to(device)}
    fixed = {
        "x_video_ids": x_video_ids.to(device),
        f"{ak}_ids": x_action_ids.to(device),
        "x_video_cond": cond["x_video_cond"].to(device, mdt),
        "x_video_cond_ids": cond["x_video_cond_ids"].to(device),
        "x_video_cond_timesteps": torch.zeros(1, cond["x_video_cond"].shape[1], device=device),
        ck: cond[ck].to(device, mdt),
        f"{ck}_ids": cond[f"{ck}_ids"].to(device),
        f"{ck}_timesteps": torch.zeros(1, cond[ck].shape[1], device=device),
        "vector": torch.zeros(1, VEC_DIM, device=device, dtype=mdt),
    }
    guidance = {
        "x_video": cfg.guidance_scale,
        ak: cfg.guidance_scale if cfg.guidance_scale_action is None else cfg.guidance_scale_action,
    }
    ctx_c = self._context(caption, device)
    ctx_uc = self._context("", device) if any(g != 1.0 for g in guidance.values()) else None
    dit = self._inference_dit()

    def predict(samples: dict[str, torch.Tensor], t) -> dict[str, torch.Tensor]:
        t = float(t) / 1000.0 if isinstance(t, torch.Tensor) and not t.is_floating_point() else float(t)
        timesteps = {
            "x_video_timesteps": torch.full((1, samples["x_video"].shape[1]), t, device=device),
            f"{ak}_timesteps": torch.full((1, cfg.chunk_size), t, device=device),
        }
        model_in = {k: v.to(mdt) for k, v in samples.items()}
        if ctx_uc is None:  # guidance 1.0 on every stream: a single conditional pass
            ctx, ctx_ids = ctx_c
            pred = dit(
                **model_in,
                **fixed,
                **timesteps,
                ctx=ctx,
                ctx_ids=ctx_ids,
                timesteps_ctx=torch.zeros(ctx.shape[:2], device=device),
            )
            pred = {k: pred[k] for k in model_in}
        else:
            pred = sampling.cfg_two_pass(dit, model_in, fixed, timesteps, ctx_uc, ctx_c, guidance)
        return {k: v.float() for k, v in pred.items()}

    return sampling.cosmos_unipc_order2(
        flow, predict, n_steps=cfg.num_inference_steps, shift=cfg.sampler_shift
    )
