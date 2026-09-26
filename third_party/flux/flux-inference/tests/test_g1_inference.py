"""G1 observation-to-action boundary without loading the 7B policy."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.dex3.g1_inference import G1Inference
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
        assert batch[CAMERA].shape == (1, 3, 192, 256)
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


def test_invalid_checkpoint_geometry_or_actions_fail():
    policy = FakePolicy()
    policy.config.canvas_hw = (256, 256)
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
