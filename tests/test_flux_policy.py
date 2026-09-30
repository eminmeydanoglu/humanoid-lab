"""CPU boundaries for joint G1 inference; no model weights or CUDA initialization."""
from contextlib import contextmanager
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_lab.evaluation import policy as module


@pytest.fixture()
def adapter(monkeypatch):
    cfg = SimpleNamespace(**module.DEPLOY, gripper_flip_dims=[], latent_hw=(1, 1),
                          camera_order=["camera"], camera_layout="single", normalization_clip=6.0,
                          video_vae_id="fixture-vae.safetensors")
    calls = []
    f3 = ModuleType("lerobot.policies.flux3.f3")

    def pack_video(cfg, vae, canvas, targets):
        assert not targets
        return {"x_video_cond": torch.full((1, 1, 2), 7.0),
                "x_video_cond_ids": torch.zeros(1, 1, 4, dtype=torch.long)}

    def pack_actions(cfg, state, past, targets):
        assert not targets and past is None
        return {"x_action_cond": state * 2,
                "x_action_cond_ids": torch.zeros(1, 1, 4, dtype=torch.long)}

    packer = SimpleNamespace(pack_video=pack_video, pack_actions=pack_actions,
                             predicted_latent_frames=lambda cfg: 8,
                             predicted_video_times=lambda cfg, b: torch.arange(8)[None],
                             action_times=lambda cfg, b: torch.arange(32)[None].float() / 30)
    f3.VEC_DIM = 3
    f3.packing = SimpleNamespace(LATENT_CHANNELS=2,
                                materialize_video=lambda cams, *a, **kw: cams[0].permute(1, 0, 2, 3))
    f3.batched_prc_vid = lambda x, t: (x.permute(0, 2, 3, 4, 1).reshape(1, -1, 2), torch.zeros(1, 8, 4))
    f3.batched_prc_audio = lambda x, t: (x.transpose(1, 2), torch.zeros(1, 32, 4))
    f3.times_to_ids = lambda times: (times * 100).long()

    def cfg_two_pass(dit, samples, fixed, ticks, uc, c, guidance):
        calls.append((guidance.copy(), samples["x_action"].dtype, ticks["x_action_timesteps"].dtype))
        assert fixed["x_action_cond"].dtype == torch.bfloat16
        return {k: v * 0.1 for k, v in samples.items()}

    def solve(flow, predict, n_steps, shift):
        assert n_steps == 4 and shift == 5
        for t in (1., .8, .6, .2):
            pred = predict(flow, t)
            flow = {k: v - pred[k] for k, v in flow.items()}
        return flow

    f3.sampling = SimpleNamespace(cfg_two_pass=cfg_two_pass, cosmos_unipc_order2=solve)
    monkeypatch.setitem(sys.modules, "lerobot.policies.flux3.f3", f3)

    class VAE:
        def decode(self, latents):
            assert latents.shape == (1, 2, 9, 1, 1)
            assert latents.dtype == torch.bfloat16
            assert (latents[:, :, 0] == 7).all()
            out = torch.zeros(1, 3, 33, 192, 256)
            out[:, :, 0] = -1
            out[:, :, 1] = 2
            return out

    base = SimpleNamespace(config=cfg, dtype_=torch.bfloat16, modality="action", packer=packer,
                           frozen=SimpleNamespace(video_vae=VAE()), _ctx_cache={})

    def context(caption, device):
        if caption not in base._ctx_cache:
            base._ctx_cache[caption] = (torch.ones(1, 2, 3, device=device), torch.zeros(1, 2, 4))
        return base._ctx_cache[caption]

    base._context = context
    base._conditioning_inputs = lambda batch: (batch["observation.state"], None)
    base._captions = lambda batch, size: [batch["task"]]
    base._cameras = lambda batch: batch["camera"][:, None]
    base._flip = lambda x: x
    base._inference_dit = lambda: None
    reset_calls = []

    def pre(obs):
        return {"observation.state": obs["observation.state"][None, None].clamp(-6, 6),
                "camera": obs["camera"][None, None], "task": obs["task"]}

    runner = SimpleNamespace(policy=SimpleNamespace(get_base_model=lambda: base), pre=pre,
                             post=lambda x: x * 3 + 1, device=torch.device("cpu"),
                             reset=lambda: reset_calls.append(True))
    policy = module.FluxJointPolicy(runner)

    @contextmanager
    def variant(name):
        if name not in ("base", "ft"):
            raise ValueError("variant")
        yield

    monkeypatch.setattr(policy, "_variant", variant)
    return policy, calls, reset_calls


def test_joint_pipeline_seed_snapshot_decode_processors_and_cache(adapter):
    policy, calls, resets = adapter
    image = np.zeros((192, 256, 3), dtype=np.uint8)
    state = np.full(28, 10., dtype=np.float32)
    first = policy.predict(image, state, "custom exact instruction", seed=3)
    second = policy.predict(image, state, "custom exact instruction", seed=3, variant="base")
    assert first["frames"].shape == (32, 192, 256, 3)
    assert first["frames"].dtype == np.uint8
    assert (first["frames"][0] == 255).all()
    assert (first["frames"][1] == 128).all()
    assert first["actions"].shape == first["normalized_actions"].shape == (32, 28)
    np.testing.assert_allclose(first["actions"], first["normalized_actions"] * 3 + 1)
    np.testing.assert_array_equal(first["actions"], second["actions"])
    np.testing.assert_array_equal(first["state"], state)
    assert first["normalized_state"].shape == (1, 28)
    assert first["latents"].shape == (1, 2, 9, 1, 1)
    assert first["metadata"]["text_cache_misses"] == ["custom exact instruction", ""]
    assert second["metadata"]["text_cache_misses"] == []
    assert first["clipping"]["normalized_state_at_clip"] == 28
    assert first["clipping"]["rgb_out_of_range"] == 3 * 192 * 256
    assert first["clipping"]["actions_clipped"] is False
    assert all(c[0] == {"x_video": 4., "x_action": 1.} for c in calls)
    assert all(c[1] == torch.bfloat16 and c[2] == torch.float32 for c in calls)
    assert set(first["timings"]) == {"preprocess_seconds", "encode_seconds", "sample_seconds", "decode_seconds", "total_seconds"}
    assert all(value >= 0 for value in first["timings"].values())
    policy.reset()
    assert resets == [True]
    module.unload_policy(policy)
    assert policy.runner is policy.base is None
    with pytest.raises(RuntimeError, match="unloaded"):
        policy.predict(image, state, "instruction")


@pytest.mark.parametrize("field,value", [("sampler", "euler"), ("num_inference_steps", 5),
                                        ("video_position_fps", 30), ("compile_model", True)])
def test_deploy_settings_reject_drift(field, value):
    cfg = SimpleNamespace(**module.DEPLOY, gripper_flip_dims=[])
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        module._validate_config(cfg)


@pytest.mark.parametrize("image,state,task", [
    (np.zeros((192, 256, 3), dtype=np.float32), np.zeros(28), "task"),
    (np.zeros((192, 256, 3), dtype=np.uint8), np.zeros(27), "task"),
    (np.zeros((192, 256, 3), dtype=np.uint8), np.full(28, np.nan), "task"),
    (np.zeros((192, 256, 3), dtype=np.uint8), np.zeros(28), " "),
])
def test_observation_validation(adapter, image, state, task):
    with pytest.raises(ValueError):
        adapter[0]._batch(image, state, task)


def test_base_heads_are_original_and_ft_restores_even_on_error(tmp_path):
    from peft import LoraConfig, get_peft_model
    from peft.utils.other import ModulesToSaveWrapper

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Linear(2, 2)
            self.dit = torch.nn.Module()
            self.dit.emb_in = torch.nn.ModuleDict({"action": torch.nn.Linear(2, 2), "action_cond": torch.nn.Linear(2, 2)})
            self.dit.final_layer = torch.nn.ModuleDict({"action": torch.nn.Linear(2, 2), "action_cond": torch.nn.Linear(2, 2)})

    names = [f"dit.{group}.{stream}" for group in ("emb_in", "final_layer") for stream in ("action", "action_cond")]
    peft = get_peft_model(Tiny(), LoraConfig(r=1, target_modules=["trunk"], modules_to_save=names))
    policy = object.__new__(module.FluxJointPolicy)
    policy.runner = SimpleNamespace(policy=peft)
    policy.base = peft.get_base_model()
    heads = policy._heads()
    x = torch.ones(1, 2)
    for head in heads.values():
        assert isinstance(head, ModulesToSaveWrapper)
        with torch.no_grad():
            head.modules_to_save["default"].weight.add_(2)
        ft = head(x).detach().clone()
        with policy._variant("base"):
            torch.testing.assert_close(head(x), head.original_module(x), rtol=0, atol=0)
        torch.testing.assert_close(head(x), ft, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="test failure"):
        with policy._variant("base"):
            raise RuntimeError("test failure")
    assert all(not h.disable_adapters for h in heads.values())
    from safetensors.torch import save_file

    policy.base_path = tmp_path / "base"
    policy.checkpoint = tmp_path / "ft"
    policy.base_path.mkdir()
    policy.checkpoint.mkdir()
    original = {f"{name}.{key}": tensor.clone() for name, head in heads.items()
                for key, tensor in head.original_module.state_dict().items()}
    finetuned = {f"base_model.model.{name}.{key}": tensor.clone() for name, head in heads.items()
                 for key, tensor in head.modules_to_save["default"].state_dict().items()}
    save_file(original, str(policy.base_path / "model.safetensors"))
    save_file(finetuned, str(policy.checkpoint / "adapter_model.safetensors"))
    assert policy.assert_head_restoration() == {
        "head_wrappers": 4, "head_tensors_verified": 8, "ft_restored": True}
    with torch.no_grad():
        next(iter(heads.values())).original_module.weight.add_(1)
    with pytest.raises(AssertionError):
        policy.assert_head_restoration()


def test_loader_reuses_raw_deploy_loader(tmp_path, monkeypatch):
    base = tmp_path / "base"
    checkpoint = tmp_path / "ft"
    base.mkdir()
    checkpoint.mkdir()
    saved = {key: value for key, value in module.DEPLOY.items() if key != "action_dim"}
    saved["output_features"] = {"action": {"type": "ACTION", "shape": [28]}}
    (base / "config.json").write_text(json.dumps(saved))
    (checkpoint / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    deploy = ModuleType("examples.dex3.g1_inference")
    calls = []
    deploy.G1Inference = SimpleNamespace(load=lambda *a, **kw: calls.append((a, kw)) or "runner")
    monkeypatch.setitem(sys.modules, deploy.__name__, deploy)
    monkeypatch.setattr(module.FluxJointPolicy, "__init__", lambda self, *a: setattr(self, "log", lambda *a: None))
    monkeypatch.setattr(module.FluxJointPolicy, "assert_head_restoration", lambda self: {})
    module.FluxJointPolicy.load(checkpoint, device="cpu", base_path=base)
    assert calls == [((checkpoint.resolve(),), {"device": "cpu", "merge_adapter": False, "offload_text_encoder": True})]


def test_gpu_probe_is_explicit_and_cpu_rejected(adapter):
    with pytest.raises(ValueError, match="GPU ownership"):
        adapter[0].gpu_parity_probe(None, None, "task")


@pytest.mark.skipif(os.environ.get("FLUX_GPU_PARITY") != "1", reason="requires explicit exclusive GPU ownership")
def test_actual_training_deploy_conditioning_and_joint_parity():
    from humanoid_lab.evaluation.datasets import Dex3Dataset
    from lerobot.policies.flux3.f3 import packing
    from flux_action.processing import history

    dataset = Dex3Dataset()
    observation = dataset.observation("893", 150)
    task = dataset.describe("893")["prompt"]
    policy = module.FluxJointPolicy.load()
    try:
        report = policy.gpu_parity_probe(observation["image"], observation["state"], task)
        assert report["ft_restored"]
        cfg = policy.base.config
        vae = policy.base.frozen.video_vae
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            batch = policy._batch(observation["image"], observation["state"], task)
            cameras = policy.base._cameras(batch)
            canvas = packing.materialize_video(cameras[0], None, torch.device("cuda"),
                                              layout=cfg.camera_layout, canvas_hw=cfg.canvas_hw)
            import av
            raw = dataset.read_frames("893", 150, 33, resized=False)
            frames = np.stack([
                av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24").to_ndarray(
                    format="rgb24", width=256, height=192) for frame in raw])
            source = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].cuda().float().div(127.5).sub(1)
            torch.testing.assert_close(canvas[None], source[:, :, :1], rtol=0, atol=2e-7)
            source[:, :, :1].copy_(canvas[None])
            deploy = packing.history_pack_video(cfg, vae, canvas[None], targets=False)
            train = packing.pack_history_video(vae, source, 1, 1, 24., cfg.latent_hw, targets=True)
            for key in ("x_video_cond", "x_video_cond_ids"):
                torch.testing.assert_close(deploy[key], train[key], rtol=0, atol=0)
            del train
            latents = history.encode_windows(vae, source, cfg)
            independent = history.pack_video(latents, cfg)
            for key in ("x_video_cond", "x_video_cond_ids"):
                torch.testing.assert_close(deploy[key], independent[key], rtol=0, atol=0)
            single = latents[:, :, :1].float().flatten()
            del independent, latents
            clip = vae.encode_task(source.to(torch.bfloat16))
            first = clip[:, :, :1, :cfg.latent_hw[0], :cfg.latent_hw[1]].float().flatten()
            l2 = float(torch.linalg.vector_norm(single - first))
            cosine = float(torch.nn.functional.cosine_similarity(single[None], first[None]))
            assert np.isfinite(l2) and np.isfinite(cosine)
            print(json.dumps({"parity": report, "single_vs_clip_l2": l2,
                              "single_vs_clip_cosine": cosine,
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated()}))
    finally:
        policy.unload()
