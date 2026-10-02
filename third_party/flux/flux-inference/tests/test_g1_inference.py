"""G1 observation-to-action boundary without loading the 7B policy."""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import examples.dex3.g1_inference as inference_module
import torch

from examples.dex3.g1_inference import G1Inference, _TASKS, _keep_text_encoder_on_cpu, _offload_text_encoder
from flux_action.data.lerobot.dex3_view import CAMERA


class FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace(
            camera_order=[CAMERA],
            action_dim=28,
            chunk_size=32,
            n_obs_steps=1,
            canvas_hw=[192, 256],
            action_representation="absolute",
        )
        self.resets = 0

    def eval(self):
        pass

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch):
        assert batch[CAMERA].shape == (1, 3, *self.config.canvas_hw)
        assert batch["observation.state"].shape == (1, 28)
        assert batch["task"] == ["stack three block"]
        assert not torch.is_grad_enabled()
        return torch.ones(1, 32, 28)


class FakePre:
    def __init__(self):
        self.resets = 0
        self.last_image = None

    def reset(self):
        self.resets += 1

    def __call__(self, obs):
        self.last_image = obs[CAMERA]
        return {
            CAMERA: obs[CAMERA][None],
            "observation.state": obs["observation.state"][None],
            "task": [obs["task"]],
        }


class FakePost:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, actions):
        return actions + 2


def test_predict_returns_denormalized_chunk_and_resets():
    policy, pre, post = FakePolicy(), FakePre(), FakePost()
    runner = G1Inference(policy, pre, post, device="cpu")
    image = np.full((192, 256, 3), 255, dtype=np.uint8)
    actions = runner.predict(image, np.zeros(28, np.float32), "stack three block")
    assert actions.shape == (32, 28) and actions.dtype == np.float32
    np.testing.assert_array_equal(actions, 3)
    torch.testing.assert_close(pre.last_image, torch.ones(3, 192, 256))
    runner.reset()
    assert (policy.resets, pre.resets, post.resets) == (2, 2, 2)
    image.setflags(write=False)
    assert runner.predict(image, np.zeros(28), "stack three block").shape == (32, 28)


def test_raw_camera_resizes_and_bad_inputs_fail():
    pre = FakePre()
    runner = G1Inference(FakePolicy(), pre, FakePost(), device="cpu")
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    assert runner.predict(raw, np.zeros(28), "stack three block").shape == (32, 28)
    assert pre.last_image.shape == (3, 192, 256)
    for image, state, task in (
        (raw.astype(np.float32), np.zeros(28), "stack three block"),
        (raw, np.zeros(27), "stack three block"),
        (raw, np.full(28, np.nan), "stack three block"),
        (raw, np.zeros(28), ""),
    ):
        with pytest.raises(ValueError):
            runner.predict(image, state, task)


@pytest.mark.parametrize("canvas_hw", [(192, 256), (480, 640), (256, 256)])
@pytest.mark.parametrize("source_hw", [(192, 256), (480, 640), (240, 320)])
def test_image_resolution_follows_checkpoint(canvas_hw, source_hw, monkeypatch):
    policy, pre = FakePolicy(), FakePre()
    policy.config.canvas_hw = list(canvas_hw)
    runner = G1Inference(policy, pre, FakePost(), device="cpu")
    assert runner.image_hw == canvas_hw
    if source_hw == canvas_hw:
        monkeypatch.setitem(sys.modules, "av", None)
    image = np.full((*source_hw, 3), 255, dtype=np.uint8)
    actions = runner.predict(image, np.zeros(28, np.float32), "stack three block")
    assert actions.shape == (32, 28)
    torch.testing.assert_close(pre.last_image, torch.ones(3, *canvas_hw))


@pytest.mark.parametrize("canvas_hw", [[], [192], [192, 256, 3], [192, -1], [192, 256.0], [True, 256], None])
def test_invalid_canvas_dimensions_fail(canvas_hw):
    policy = FakePolicy()
    policy.config.canvas_hw = canvas_hw
    with pytest.raises(ValueError, match="canvas_hw"):
        G1Inference(policy, FakePre(), FakePost(), device="cpu")


def test_invalid_checkpoint_geometry_or_actions_fail():
    policy = FakePolicy()
    policy.config.canvas_hw = (0, 256)
    with pytest.raises(ValueError, match="checkpoint"):
        G1Inference(policy, FakePre(), FakePost(), device="cpu")
    policy.config.canvas_hw = (192, 256)
    policy.predict_action_chunk = lambda batch: torch.full((1, 32, 28), float("nan"))
    runner = G1Inference(policy, FakePre(), FakePost(), device="cpu")
    with pytest.raises(ValueError, match="invalid action chunk"):
        runner.predict(np.zeros((192, 256, 3), np.uint8), np.zeros(28), "stack three block")


def test_saved_g1_processors_accept_live_observation():
    pytest.importorskip("lerobot")
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.flux3.configuration_flux3 import Flux3Config

    base = Path(__file__).resolve().parents[1] / "outputs/dex3/g1-base-policy"
    if not (base / "policy_preprocessor.json").exists():
        pytest.skip("G1 processor artifact unavailable")
    config = Flux3Config.from_pretrained(base)
    pre, post = make_pre_post_processors(config, pretrained_path=base)
    runner = G1Inference(FakePolicy(), pre, post, device="cpu")
    # The saved processor owns state and action quantiles; the policy stub supplies normalized zeros.
    runner.policy.predict_action_chunk = lambda batch: torch.zeros(1, 32, 28)
    actions = runner.predict(np.zeros((192, 256, 3), np.uint8), np.zeros(28), "stack three block")
    assert actions.shape == (32, 28) and np.isfinite(actions).all()


def test_precomputed_tasks_match_ros_node():
    node = Path(__file__).resolve().parents[1] / "ros2/flux_dex3/flux_dex3/node.py"
    tree = ast.parse(node.read_text(encoding="utf-8"))
    prompts = next(statement.value.args[0] for statement in tree.body
                   if isinstance(statement, ast.Assign) and
                   any(isinstance(target, ast.Name) and target.id == "PROMPTS" for target in statement.targets))
    assert set(_TASKS) == {"", *ast.literal_eval(prompts)}


def test_text_encoder_is_skipped_when_frozen_components_move():
    video = torch.nn.Linear(2, 2)
    encoder = torch.nn.Linear(2, 2)
    frozen = SimpleNamespace(video_vae=SimpleNamespace(module=video), text_encoder=encoder)
    _keep_text_encoder_on_cpu(SimpleNamespace(frozen=frozen))
    frozen._apply(lambda tensor: tensor.to(torch.float64))
    assert video.weight.dtype == torch.float64
    assert encoder.weight.dtype == torch.float32
    assert encoder.weight.device.type == "cpu"


def test_offloaded_text_contexts_are_cached_and_encode_new_tasks_on_cpu(monkeypatch):
    calls = []

    class Encoder:
        def to(self, device):
            calls.append(("move", device))
            return self

    def encode(encoder, caption, device, *, fixed_length):
        calls.append(("encode", caption, device, fixed_length))
        return torch.ones(1, 2, 3)

    module = types.ModuleType("lerobot.policies.flux3.f3")
    module.VEC_DIM = 3
    module.text_context = encode
    module.packing = SimpleNamespace(pack_text=lambda ctx, width: {"ctx_ids": torch.zeros(1, 2)})
    monkeypatch.setitem(sys.modules, "lerobot.policies.flux3.f3", module)
    base = SimpleNamespace(frozen=SimpleNamespace(text_encoder=Encoder()), _ctx_cache={},
                           config=SimpleNamespace(text_fixed_length=320), dtype_=torch.bfloat16)
    _offload_text_encoder(base, "cpu")
    assert calls[0] == ("move", "cpu")
    assert [c[1] for c in calls if c[0] == "encode"] == list(_TASKS)
    assert base._context("new instruction", torch.device("cpu"))[0].dtype == torch.bfloat16
    assert calls[-1] == ("encode", "new instruction", "cpu", 320)
    count = len(calls)
    base._context("new instruction", torch.device("cpu"))
    assert len(calls) == count


def test_load_merges_adapter_only_when_requested(monkeypatch, tmp_path):
    """``merge_adapter`` bakes the LoRA delta in memory on request, after the device move."""
    calls = []

    class LoadedPolicy:
        config = SimpleNamespace(
            camera_order=[CAMERA],
            action_dim=28,
            chunk_size=32,
            n_obs_steps=1,
            canvas_hw=[192, 256],
            action_representation="absolute",
        )

        def eval(self):
            pass

        def reset(self):
            pass

    class StubPolicy(LoadedPolicy):
        @classmethod
        def from_pretrained(cls, name, strict=True):
            calls.append("base")
            return cls()

    class StubPeftConfig:
        base_model_name_or_path = "base-policy"

        @classmethod
        def from_pretrained(cls, path):
            calls.append("config")
            return cls()

    class StubPeftModel:
        config = LoadedPolicy.config  # PeftModel forwards the base policy's config

        @classmethod
        def from_pretrained(cls, base, path, config=None, is_trainable=False):
            calls.append("adapter")
            return cls()

        def to(self, device):
            calls.append(f"to:{device}")
            return self

        def eval(self):
            return self

        def reset(self):
            pass

        def merge_adapter(self):
            calls.append("merge")

    modules = {
        "lerobot": types.ModuleType("lerobot"),
        "lerobot.policies": types.ModuleType("lerobot.policies"),
        "lerobot.policies.factory": types.ModuleType("lerobot.policies.factory"),
        "lerobot.policies.flux3": types.ModuleType("lerobot.policies.flux3"),
        "lerobot.policies.flux3.modeling_flux3": types.ModuleType("lerobot.policies.flux3.modeling_flux3"),
        "peft": types.ModuleType("peft"),
    }
    modules["lerobot.policies.factory"].make_pre_post_processors = lambda config, pretrained_path: (
        FakePre(),
        FakePost(),
    )
    modules["lerobot.policies.flux3.modeling_flux3"].Flux3Policy = StubPolicy
    modules["peft"].PeftConfig = StubPeftConfig
    modules["peft"].PeftModel = StubPeftModel
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert isinstance(G1Inference.load(tmp_path, device="cpu", merge_adapter=True, offload_text_encoder=False).policy, StubPeftModel)
    assert calls.index("merge") > calls.index("to:cpu")
    calls.clear()
    G1Inference.load(tmp_path, device="cpu", offload_text_encoder=False)
    assert "merge" not in calls
    calls.clear()
    monkeypatch.setattr(inference_module, "_keep_text_encoder_on_cpu", lambda base: calls.append("keep-on-cpu"))
    monkeypatch.setattr(inference_module, "_offload_text_encoder", lambda base, device: calls.append("cache-text"))
    G1Inference.load(tmp_path, device="cpu")
    assert calls.index("keep-on-cpu") < calls.index("to:cpu") < calls.index("cache-text")
