"""Right-hand joint-order contract for the Psi0 v1.1-init entrypoint.

The v1.1 init was post-trained with the right hand mirroring the left
(thumb, middle, index); the Unitree collection and the v1 pack store the SONIC
hardware order (thumb, index, middle).  These tests pin the two orders, the
explicit mapping decision, the exact Psi0 slice keys that implement it, and the
fact that the historical v1.0 wrapper argv is unchanged.  The psi0-environment
tests at the bottom drive Psi0's real repack/statistics transforms; they are
skipped outside ``./dev.sh psi0-tests``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from humanoid_lab.datasets.psi0 import hand_order  # noqa: E402

WRAPPER_V10 = REPO_ROOT / "scripts/psi0-unitree-dex3-sonic-v1.sh"
WRAPPER_V11 = REPO_ROOT / "scripts/psi0-unitree-dex3-sonic-v1.1.sh"
VERIFIER = REPO_ROOT / "scripts/verify-psi0-unitree-dex3-sonic-v1.py"

LEFT = hand_order.LEFT_HAND_NAMES


def load_verifier():
    name = "verify_psi0_unitree_dex3_sonic_v1_hand_order"
    spec = importlib.util.spec_from_file_location(name, VERIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so dataclasses can resolve cls.__module__.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def wrapper_argv(wrapper: Path, *overrides: str) -> list[str]:
    env = dict(os.environ, **dict(pair.split("=", 1) for pair in overrides))
    resolved = subprocess.run(
        ["bash", str(wrapper), "--print-args"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    return [line for line in resolved.stdout.splitlines() if line.strip()]


def flag_value(tokens: list[str], name: str) -> str | None:
    prefix = f"--{name}="
    for token in tokens:
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def flag_values(tokens: list[str], name: str) -> list[str]:
    name = f"--{name}"
    for index, token in enumerate(tokens):
        if token == name:
            values: list[str] = []
            for follow in tokens[index + 1:]:
                if follow.startswith("--"):
                    break
                values.append(follow)
            return values
    return []


def right_names(order) -> list[str]:
    return [f"right_hand_{name}_joint" for name in order]


def write_info(
    repo: Path,
    *,
    state_right=hand_order.HARDWARE_RIGHT_HAND_ORDER,
    action_right=hand_order.HARDWARE_RIGHT_HAND_ORDER,
    state_names=None,
    action_names=None,
    nested: bool = False,
) -> None:
    state = (
        state_names
        if state_names is not None
        else [f"body_{index:02d}_joint" for index in range(29)] + [*LEFT, *right_names(state_right)]
    )
    action = action_names if action_names is not None else [*LEFT, *right_names(action_right)]
    if nested:
        state, action = [state], [action]
    (repo / "meta").mkdir(parents=True, exist_ok=True)
    (repo / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [43], "names": state},
                    "action": {"dtype": "float32", "shape": [14], "names": action},
                },
            }
        ),
        encoding="utf-8",
    )


def fake_cfg(mapping: str):
    return SimpleNamespace(
        data=SimpleNamespace(
            transform=SimpleNamespace(
                repack=SimpleNamespace(
                    action_keys=hand_order.action_keys(mapping),
                    state_keys=hand_order.state_keys(mapping),
                ),
                field=SimpleNamespace(
                    stat_action_keys=hand_order.action_keys(mapping),
                    stat_state_keys=hand_order.state_keys(mapping),
                ),
            )
        )
    )


def fake_context(verify, pack: Path, mapping: str):
    return verify.Context(
        cfg=fake_cfg(mapping),
        tokens=["finetune_sonic_psi0_config"],
        dataset_root=pack,
        init_dir=pack,
        stats_path=pack / "train/meta/stats_psi0.json",
        mask_key="action.mask",
        instruction_key="task_description",
        right_hand_map=mapping,
    )


class OrderDefinitionTest(unittest.TestCase):
    def test_checkpoint_order_mirrors_the_left_hand(self) -> None:
        self.assertEqual(hand_order.CHECKPOINT_RIGHT_HAND_ORDER, hand_order.LEFT_HAND_ORDER)

    def test_hardware_order_puts_index_before_middle(self) -> None:
        self.assertEqual(
            hand_order.HARDWARE_RIGHT_HAND_ORDER,
            ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"),
        )

    def test_classify_accepts_only_the_two_known_orders(self) -> None:
        self.assertEqual(hand_order.classify_hand_order(hand_order.CHECKPOINT_HAND_NAMES), hand_order.CHECKPOINT)
        self.assertEqual(hand_order.classify_hand_order(hand_order.HARDWARE_HAND_NAMES), hand_order.HARDWARE)
        self.assertIsNone(hand_order.classify_hand_order(tuple(reversed(hand_order.HARDWARE_HAND_NAMES))))
        self.assertIsNone(hand_order.classify_hand_order(hand_order.HARDWARE_HAND_NAMES[:-1]))


class PermutationTest(unittest.TestCase):
    def test_hand14_moves_the_right_index_and_middle_blocks(self) -> None:
        source = np.arange(14, dtype=np.float32)
        mapped = hand_order.apply_hand14(source, hand_order.MAP_HARDWARE_TO_CHECKPOINT)
        self.assertEqual(mapped.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 10, 11])

    def test_state43_keeps_the_left_body_and_swaps_the_right_hand(self) -> None:
        source = np.arange(43, dtype=np.float32)
        mapped = hand_order.apply_state43(source, hand_order.MAP_HARDWARE_TO_CHECKPOINT)
        self.assertEqual(mapped[:36].tolist(), list(range(36)))
        self.assertEqual(mapped[36:].tolist(), [36, 37, 38, 41, 42, 39, 40])

    def test_both_permutations_are_their_own_inverse(self) -> None:
        for source, apply in (
            (np.arange(14), hand_order.apply_hand14),
            (np.arange(43), hand_order.apply_state43),
        ):
            twice = apply(apply(source, hand_order.MAP_HARDWARE_TO_CHECKPOINT), hand_order.MAP_HARDWARE_TO_CHECKPOINT)
            self.assertEqual(twice.tolist(), source.tolist())

    def test_identity_mapping_returns_the_input(self) -> None:
        source = np.arange(14)
        self.assertIs(hand_order.apply_hand14(source, hand_order.MAP_NONE), source)
        self.assertIs(hand_order.apply_state43(source, hand_order.MAP_NONE), source)

    def test_key_lists_are_the_exact_slice_expressions(self) -> None:
        self.assertEqual(hand_order.action_keys(hand_order.MAP_NONE), ["action.body_token_v1_1", "action"])
        self.assertEqual(hand_order.state_keys(hand_order.MAP_NONE), ["observation.state"])
        self.assertEqual(
            hand_order.action_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT),
            ["action.body_token_v1_1", "action[0:7]", "action[7:10]", "action[12:14]", "action[10:12]"],
        )
        self.assertEqual(
            hand_order.state_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT),
            ["observation.state[0:36]", "observation.state[36:39]", "observation.state[41:43]", "observation.state[39:41]"],
        )


class ReadSplitHandOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_hardware_order_pack_reads_as_hardware(self) -> None:
        write_info(self.root)
        reading = hand_order.read_split_hand_order(self.root)
        self.assertEqual((reading.state, reading.action), (hand_order.HARDWARE, hand_order.HARDWARE))

    def test_checkpoint_order_pack_reads_as_checkpoint(self) -> None:
        write_info(
            self.root,
            state_right=hand_order.CHECKPOINT_RIGHT_HAND_ORDER,
            action_right=hand_order.CHECKPOINT_RIGHT_HAND_ORDER,
        )
        reading = hand_order.read_split_hand_order(self.root)
        self.assertEqual((reading.state, reading.action), (hand_order.CHECKPOINT, hand_order.CHECKPOINT))

    def test_nested_lerobot_names_are_accepted(self) -> None:
        write_info(self.root, nested=True)
        reading = hand_order.read_split_hand_order(self.root)
        self.assertEqual(reading.state, hand_order.HARDWARE)

    def test_unknown_right_order_is_refused(self) -> None:
        write_info(self.root, state_right=tuple(reversed(hand_order.HARDWARE_RIGHT_HAND_ORDER)))
        with self.assertRaisesRegex(hand_order.HandOrderError, "only.*understands"):
            hand_order.read_split_hand_order(self.root)

    def test_missing_names_are_refused(self) -> None:
        write_info(self.root)
        info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        del info["features"]["action"]["names"]
        (self.root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaisesRegex(hand_order.HandOrderError, "refuses to guess"):
            hand_order.read_split_hand_order(self.root)

    def test_reordered_left_hand_is_refused(self) -> None:
        left = list(LEFT)
        left[3], left[5] = left[5], left[3]
        write_info(
            self.root,
            state_names=[f"body_{index:02d}_joint" for index in range(29)]
            + left
            + right_names(hand_order.HARDWARE_RIGHT_HAND_ORDER),
        )
        with self.assertRaisesRegex(hand_order.HandOrderError, "left hand"):
            hand_order.read_split_hand_order(self.root)

    def test_missing_info_is_refused(self) -> None:
        with self.assertRaisesRegex(hand_order.HandOrderError, "is missing"):
            hand_order.read_split_hand_order(self.root / "absent")


class PlanMappingTest(unittest.TestCase):
    def test_none_accepts_a_checkpoint_order_pack(self) -> None:
        self.assertEqual(
            hand_order.plan_mapping(hand_order.CHECKPOINT, hand_order.CHECKPOINT, hand_order.MAP_NONE),
            hand_order.MAP_NONE,
        )

    def test_none_refuses_a_hardware_order_pack_with_the_exact_knob(self) -> None:
        with self.assertRaisesRegex(
            hand_order.HandOrderError, "RIGHT_HAND_MAP=hardware2checkpoint"
        ):
            hand_order.plan_mapping(hand_order.HARDWARE, hand_order.HARDWARE, hand_order.MAP_NONE)

    def test_mapping_accepts_a_hardware_order_pack(self) -> None:
        self.assertEqual(
            hand_order.plan_mapping(hand_order.HARDWARE, hand_order.HARDWARE, hand_order.MAP_HARDWARE_TO_CHECKPOINT),
            hand_order.MAP_HARDWARE_TO_CHECKPOINT,
        )

    def test_mapping_refuses_a_double_swap(self) -> None:
        with self.assertRaisesRegex(hand_order.HandOrderError, "double-swap"):
            hand_order.plan_mapping(
                hand_order.CHECKPOINT, hand_order.CHECKPOINT, hand_order.MAP_HARDWARE_TO_CHECKPOINT
            )

    def test_state_and_action_disagreement_is_refused(self) -> None:
        with self.assertRaisesRegex(hand_order.HandOrderError, "disagrees with itself"):
            hand_order.plan_mapping(hand_order.HARDWARE, hand_order.CHECKPOINT, hand_order.MAP_NONE)

    def test_unknown_map_is_refused(self) -> None:
        with self.assertRaisesRegex(hand_order.HandOrderError, "unknown right-hand map"):
            hand_order.plan_mapping(hand_order.HARDWARE, hand_order.HARDWARE, "swap-everything")


class VerifierHandOrderCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.verify = load_verifier()
        self.tmp = tempfile.TemporaryDirectory()
        self.pack = Path(self.tmp.name) / "psi0-unitree-dex3-sonic-v1"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_pack(self, *, state_right, action_right, mapping):
        for split in ("train", "val"):
            write_info(
                self.pack / split,
                state_right=state_right,
                action_right=action_right,
            )
        return fake_context(self.verify, self.pack, mapping)

    def test_hardware_pack_without_mapping_fails_with_the_knob(self) -> None:
        ctx = self._write_pack(
            state_right=hand_order.HARDWARE_RIGHT_HAND_ORDER,
            action_right=hand_order.HARDWARE_RIGHT_HAND_ORDER,
            mapping=hand_order.MAP_NONE,
        )
        with self.assertRaisesRegex(self.verify.CheckError, "RIGHT_HAND_MAP=hardware2checkpoint"):
            self.verify.check_hand_order(ctx)

    def test_hardware_pack_with_mapping_and_matching_keys_passes(self) -> None:
        ctx = self._write_pack(
            state_right=hand_order.HARDWARE_RIGHT_HAND_ORDER,
            action_right=hand_order.HARDWARE_RIGHT_HAND_ORDER,
            mapping=hand_order.MAP_HARDWARE_TO_CHECKPOINT,
        )
        notes = self.verify.check_hand_order(ctx)
        self.assertTrue(any("mapping = hardware2checkpoint" in note for note in notes), notes)

    def test_checkpoint_pack_without_mapping_passes(self) -> None:
        ctx = self._write_pack(
            state_right=hand_order.CHECKPOINT_RIGHT_HAND_ORDER,
            action_right=hand_order.CHECKPOINT_RIGHT_HAND_ORDER,
            mapping=hand_order.MAP_NONE,
        )
        notes = self.verify.check_hand_order(ctx)
        self.assertTrue(any("mapping = none" in note for note in notes), notes)

    def test_checkpoint_pack_with_mapping_fails(self) -> None:
        ctx = self._write_pack(
            state_right=hand_order.CHECKPOINT_RIGHT_HAND_ORDER,
            action_right=hand_order.CHECKPOINT_RIGHT_HAND_ORDER,
            mapping=hand_order.MAP_HARDWARE_TO_CHECKPOINT,
        )
        with self.assertRaisesRegex(self.verify.CheckError, "double-swap"):
            self.verify.check_hand_order(ctx)

    def test_train_val_disagreement_fails(self) -> None:
        write_info(self.pack / "train")
        write_info(
            self.pack / "val",
            state_right=hand_order.CHECKPOINT_RIGHT_HAND_ORDER,
            action_right=hand_order.CHECKPOINT_RIGHT_HAND_ORDER,
        )
        ctx = fake_context(self.verify, self.pack, hand_order.MAP_NONE)
        with self.assertRaisesRegex(self.verify.CheckError, "train and val"):
            self.verify.check_hand_order(ctx)

    def test_wrong_key_list_for_the_mapping_fails(self) -> None:
        ctx = self._write_pack(
            state_right=hand_order.HARDWARE_RIGHT_HAND_ORDER,
            action_right=hand_order.HARDWARE_RIGHT_HAND_ORDER,
            mapping=hand_order.MAP_HARDWARE_TO_CHECKPOINT,
        )
        ctx.cfg.data.transform.repack.action_keys = ["action.body_token_v1_1", "action"]
        with self.assertRaisesRegex(self.verify.CheckError, "repack.action_keys"):
            self.verify.check_hand_order(ctx)

    def test_missing_pack_fails_before_reading(self) -> None:
        ctx = fake_context(self.verify, self.pack, hand_order.MAP_NONE)
        with self.assertRaisesRegex(self.verify.CheckError, "does not exist yet"):
            self.verify.check_hand_order(ctx)


class WrapperArgvTest(unittest.TestCase):
    def test_v11_default_is_the_v11_init_with_b2_accum16(self) -> None:
        tokens = wrapper_argv(WRAPPER_V11)
        init = flag_value(tokens, "model.model_name_or_path")
        self.assertTrue(init.endswith("postpre.sonic1.1.unifolm.2609181726.40k"), init)
        self.assertEqual(flag_value(tokens, "model.pretrained-action-header-path"), init)
        self.assertEqual(flag_value(tokens, "train.train_batch_size"), "2")
        self.assertEqual(flag_value(tokens, "train.val_batch_size"), "2")
        self.assertEqual(flag_value(tokens, "train.gradient_accumulation_steps"), "16")
        self.assertEqual(flag_value(tokens, "train.max_training_steps"), "40000")
        self.assertEqual(flag_value(tokens, "train.learning_rate"), "2.5e-5")
        self.assertEqual(flag_values(tokens, "data.transform.model.resize.size"), ["240", "320"])
        self.assertEqual(flag_values(tokens, "data.transform.repack.action-keys"), ["action.body_token_v1_1", "action"])
        self.assertEqual(flag_values(tokens, "data.transform.repack.state-keys"), ["observation.state"])
        self.assertEqual(flag_value(tokens, "model.tune_vlm"), None)
        self.assertEqual(flag_value(tokens, "train.resume_from_checkpoint"), None)

    def test_v11_mapping_uses_the_exact_permutation_keys(self) -> None:
        tokens = wrapper_argv(WRAPPER_V11, "RIGHT_HAND_MAP=hardware2checkpoint")
        expected_action = hand_order.action_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT)
        expected_state = hand_order.state_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT)
        self.assertEqual(flag_values(tokens, "data.transform.repack.action-keys"), expected_action)
        self.assertEqual(flag_values(tokens, "data.transform.field.stat-action-keys"), expected_action)
        self.assertEqual(flag_values(tokens, "data.transform.repack.state-keys"), expected_state)
        self.assertEqual(flag_values(tokens, "data.transform.field.stat-state-keys"), expected_state)
        # No Psi0 argv flag carries the mapping; it is a wrapper/verifier knob.
        self.assertEqual(flag_value(tokens, "right-hand-map"), None)

    def test_v11_rejects_an_unknown_mapping(self) -> None:
        env = dict(os.environ, RIGHT_HAND_MAP="middle2index")
        resolved = subprocess.run(
            ["bash", str(WRAPPER_V11), "--print-args"], cwd=REPO_ROOT, env=env,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(resolved.returncode, 2)
        self.assertIn("hardware2checkpoint", resolved.stderr)

    def test_v10_recipe_is_unchanged(self) -> None:
        tokens = wrapper_argv(WRAPPER_V10)
        init = flag_value(tokens, "model.model_name_or_path")
        self.assertTrue(init.endswith("postpre.sonic1.0.unifolm.2609092156.40k"), init)
        self.assertEqual(flag_value(tokens, "train.train_batch_size"), "1")
        self.assertEqual(flag_value(tokens, "train.gradient_accumulation_steps"), "32")
        self.assertEqual(flag_values(tokens, "data.transform.repack.action-keys"), ["action.body_token_v1_1", "action"])
        self.assertEqual(flag_values(tokens, "data.transform.repack.state-keys"), [])
        self.assertEqual(flag_value(tokens, "train.checkpointing_steps"), "1000")


def has_psi() -> bool:
    return importlib.util.find_spec("psi") is not None


def has_lerobot() -> bool:
    return importlib.util.find_spec("lerobot") is not None


@unittest.skipUnless(has_psi(), "needs the psi0 environment (./dev.sh psi0-tests)")
class V11CheckpointContractTest(unittest.TestCase):
    def test_v11_header_matches_the_action_contract(self) -> None:
        verify = load_verifier()
        ctx = verify.build_context(wrapper_argv(WRAPPER_V11))
        if not (ctx.init_dir / "action_header.safetensors").is_file():
            self.skipTest(f"v1.1 checkpoint is not fetched: {ctx.init_dir}")
        notes = verify.check_ckpt(ctx)
        self.assertTrue(any("action_proj_out" in note for note in notes), notes)


@unittest.skipUnless(has_psi(), "needs the psi0 environment (./dev.sh psi0-tests)")
class SlicedStateDeltaFallbackTest(unittest.TestCase):
    """The pinned transform hands sliced state keys to LeRobot as column names."""

    def test_slice_keys_collapse_to_the_base_field(self) -> None:
        from psi.config import transform_psi0_sonic as ps

        from humanoid_lab.psi0_compat import install_sliced_state_delta_fallback

        install_sliced_state_delta_fallback()
        repack = ps.SonicRepackTransform(
            state_keys=hand_order.state_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT),
            num_past_frames=0,
            action_chunk_size=30,
            state_temporal_jitter=10,
        )
        delta = repack.delta_timestamps(30)
        for key in hand_order.state_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT):
            self.assertNotIn(key, delta)
        self.assertIn("observation.state", delta)
        self.assertEqual(len(delta["observation.state"]), 21)

    def test_plain_state_key_is_untouched(self) -> None:
        from psi.config import transform_psi0_sonic as ps

        from humanoid_lab.psi0_compat import install_sliced_state_delta_fallback

        install_sliced_state_delta_fallback()
        repack = ps.SonicRepackTransform(
            state_keys=["observation.state"], num_past_frames=0, action_chunk_size=30, state_temporal_jitter=0
        )
        self.assertEqual(repack.delta_timestamps(30)["observation.state"], [0.0])


@unittest.skipUnless(has_psi(), "needs the psi0 environment (./dev.sh psi0-tests)")
class VerifierTransformWithMappingTest(unittest.TestCase):
    def test_aug1_transform_check_runs_with_the_mapping_and_jitter_window(self) -> None:
        verify = load_verifier()
        verify._select_available_attention_backend()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tokens = wrapper_argv(
            WRAPPER_V11,
            f"DATASET_ROOT={Path(self.tmp.name) / 'psi0-unitree-dex3-sonic-v1'}",
            "AUG=1",
            "RIGHT_HAND_MAP=hardware2checkpoint",
        )
        ctx = verify.build_context(tokens)
        ctx.right_hand_map = hand_order.MAP_HARDWARE_TO_CHECKPOINT
        notes = verify.check_transform(ctx)
        self.assertTrue(any("invalid cells closed" in note for note in notes), notes)
        self.assertTrue(any("normalised range" in note for note in notes), notes)

    def test_versioned_pack_root_needs_the_explicit_pin(self) -> None:
        verify = load_verifier()
        tokens = wrapper_argv(WRAPPER_V11, "DATASET_DIR=psi0-unitree-dex3-sonic-v1.1")
        ctx = verify.build_context(tokens)
        # Without the wrapper pin the historical strict name check refuses it.
        with self.assertRaisesRegex(verify.CheckError, "must be an absolute"):
            verify.check_config(ctx)
        # The wrapper passes its DATASET_ROOT basename, which binds that exact root.
        ctx.dataset_dir_name = "psi0-unitree-dex3-sonic-v1.1"
        notes = verify.check_config(ctx)
        self.assertTrue(any("pinned by the wrapper" in note for note in notes), notes)


@unittest.skipUnless(has_psi() and has_lerobot(), "needs the psi0 environment (./dev.sh psi0-tests)")
class MappedLoaderChainTest(unittest.TestCase):
    """The mapped keys must open the real LeRobot chain and deliver the permutation."""

    @classmethod
    def setUpClass(cls) -> None:
        from humanoid_lab.datasets.psi0 import contract

        cls.verify = load_verifier()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pack = Path(cls.tmp.name) / contract.DATASET_DIR
        # The pack (and its stats_psi0.json) must exist before the config is
        # resolved: the field transform loads the statistics at construction.
        cls._build_pack(contract)
        tokens = wrapper_argv(
            WRAPPER_V11,
            f"DATASET_ROOT={cls.pack}",
            "AUG=1",
            "RIGHT_HAND_MAP=hardware2checkpoint",
        )
        cls.ctx = cls.verify.build_context(tokens)
        cls.ctx.right_hand_map = hand_order.MAP_HARDWARE_TO_CHECKPOINT
        cls._checkpoint_present = (cls.ctx.init_dir / "config.json").is_file()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    @classmethod
    def _build_pack(cls, contract) -> None:
        import numpy as np
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            contract.IMAGE_KEY: {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
            contract.STATE_KEY: {"dtype": "float32", "shape": (43,), "names": [f"s{i}" for i in range(43)]},
            contract.BODY_TOKEN_KEY: {"dtype": "float32", "shape": (64,), "names": [f"t{i}" for i in range(64)]},
            contract.HAND_KEY: {"dtype": "float32", "shape": (14,), "names": [f"h{i}" for i in range(14)]},
            contract.MASK_KEY: {"dtype": "float32", "shape": (80,), "names": [f"m{i}" for i in range(80)]},
            contract.ANCHOR_MASK_KEY: {"dtype": "bool", "shape": (1,), "names": None},
            contract.INSTRUCTION_KEY: {"dtype": "string", "shape": (1,), "names": None},
        }
        rng = np.random.default_rng(20260929)
        for split in (contract.TRAIN_REPO_ID, contract.VAL_REPO_ID):
            dataset = LeRobotDataset.create(
                repo_id=split, root=cls.pack / split, fps=30, features=features,
                use_videos=True, robot_type="Unitree_G1",
            )
            for index in range(40):
                frame_valid = index < 35
                mask = np.ones(contract.ACTION_MODEL_DIM, dtype=np.float32)
                if not frame_valid:
                    mask[:] = 0.0
                mask[contract.ACTION_DIM:] = 0.0
                dataset.add_frame(
                    {
                        contract.IMAGE_KEY: (rng.random((480, 640, 3)) * 255).astype(np.uint8),
                        contract.STATE_KEY: rng.normal(size=43).astype(np.float32),
                        contract.BODY_TOKEN_KEY: rng.normal(size=64).astype(np.float32),
                        contract.HAND_KEY: rng.normal(size=14).astype(np.float32),
                        contract.MASK_KEY: mask,
                        contract.ANCHOR_MASK_KEY: np.array(
                            [bool(frame_valid and index + contract.ACTION_CHUNK <= 40)], dtype=bool
                        ),
                        contract.INSTRUCTION_KEY: "pick the doll up",
                    },
                    task="pick the doll up",
                )
            dataset.save_episode()
            rows = [
                json.loads(line)
                for line in (cls.pack / split / "meta/episodes_stats.jsonl").read_text(encoding="utf-8").strip().splitlines()
            ]
            stats: dict[str, dict[str, list[float]]] = {}
            for row in rows:
                for feature, block in row["stats"].items():
                    if "min" not in block:
                        continue
                    entry = stats.setdefault(feature, {"min": list(block["min"]), "max": list(block["max"])})
                    entry["min"] = [min(a, b) for a, b in zip(entry["min"], block["min"])]
                    entry["max"] = [max(a, b) for a, b in zip(entry["max"], block["max"])]
            (cls.pack / split / "meta" / contract.STATS_FILENAME).write_text(json.dumps(stats), encoding="utf-8")
            (cls.pack / split / "meta" / "modality.json").write_text("{}", encoding="utf-8")

    def setUp(self) -> None:
        if not self._checkpoint_present:
            self.skipTest("warm-start checkpoint is not fetched; check_loader needs its processor")
        self.verify._select_available_attention_backend()

    def test_mapped_keys_load_and_deliver_the_permutation(self) -> None:
        notes = self.verify.check_loader(self.ctx)
        self.assertTrue(any("closed cells" in note for note in notes), notes)


@unittest.skipUnless(has_psi(), "needs the psi0 environment (./dev.sh psi0-tests)")
class RealTransformMappingTest(unittest.TestCase):
    """The split keys drive Psi0's own transforms; this is the numeric proof."""

    def _frame(self):
        import torch

        rng = np.random.default_rng(20260929)
        return {
            "action.body_token_v1_1": rng.normal(size=(30, 64)).astype(np.float32),
            "action": rng.normal(size=(30, 14)).astype(np.float32),
            "observation.state": rng.normal(size=(1, 43)).astype(np.float32),
            "action.mask": np.pad(np.ones((30, 78), dtype=np.float32), ((0, 0), (0, 2))),
            "task_description": "pick the doll up",
            "observation.images.egocentric": torch.zeros(3, 480, 640, dtype=torch.uint8),
        }

    def test_repack_applies_the_mapping_and_keeps_the_mask(self) -> None:
        from psi.config import transform_psi0_sonic as ps

        repack = ps.SonicRepackTransform(
            action_keys=hand_order.action_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT),
            state_keys=hand_order.state_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT),
            pad_action_dim=80,
            pad_state_dim=45,
            action_mask_key="action.mask",
            num_past_frames=0,
            action_chunk_size=30,
            instruction_key="task_description",
        )
        frame = self._frame()
        out = repack(dict(frame))
        actions = np.asarray(out["actions"])
        states = np.asarray(out["states"])
        self.assertTrue(np.array_equal(actions[:, :64], frame["action.body_token_v1_1"]))
        self.assertTrue(np.array_equal(actions[:, 64:78], hand_order.apply_hand14(frame["action"], hand_order.MAP_HARDWARE_TO_CHECKPOINT)))
        self.assertTrue(np.array_equal(actions[:, 78:], np.zeros((30, 2), dtype=np.float32)))
        self.assertTrue(np.array_equal(states[:, :43], hand_order.apply_state43(frame["observation.state"], hand_order.MAP_HARDWARE_TO_CHECKPOINT)))
        self.assertTrue(np.array_equal(states[:, 43:], np.zeros((1, 2), dtype=np.float32)))
        self.assertTrue(np.array_equal(np.asarray(out["actions_mask"]), frame["action.mask"]))

    def test_statistics_keys_normalise_the_permuted_channels(self) -> None:
        from psi.config import transform_psi0_sonic as ps

        stats = {
            "action.body_token_v1_1": {"min": [-1.0] * 64, "max": [1.0] * 64},
            "action": {"min": [float(i) for i in range(14)], "max": [float(i) + 2 for i in range(14)]},
            "observation.state": {"min": [float(i) for i in range(43)], "max": [float(i) + 2 for i in range(43)]},
        }
        field = ps.SonicActionStateTransform(
            stat_path="/nonexistent/stats_psi0.json",
            stat_action_keys=hand_order.action_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT),
            stat_state_keys=hand_order.state_keys(hand_order.MAP_HARDWARE_TO_CHECKPOINT),
            action_norm_type="bounds",
            normalize_state=True,
            use_norm_mask=False,
            pad_action_dim=80,
            pad_state_dim=45,
        )
        field.populate_stats(stats)
        frame = self._frame()
        actions = np.concatenate(
            [frame["action.body_token_v1_1"], frame["action"], np.zeros((30, 2), dtype=np.float32)], axis=1
        )
        states = np.concatenate([frame["observation.state"], np.zeros((1, 2), dtype=np.float32)], axis=1)
        out = field({"actions": actions, "states": states})

        perm = list(hand_order.HAND14_HARDWARE_TO_CHECKPOINT)
        low = np.asarray(stats["action"]["min"])[perm]
        high = np.asarray(stats["action"]["max"])[perm]
        expected_hand = np.clip((frame["action"][:, perm] - low) / (high - low) * 2 - 1, -1, 1)
        self.assertTrue(np.allclose(out["actions"][:, 64:78], expected_hand, atol=1e-6))
        self.assertTrue(np.allclose(out["actions"][:, 78:], 0))

        perm43 = list(hand_order.STATE43_HARDWARE_TO_CHECKPOINT)
        slow = np.asarray(stats["observation.state"]["min"])[perm43]
        shigh = np.asarray(stats["observation.state"]["max"])[perm43]
        expected_state = np.clip((frame["observation.state"][:, perm43] - slow) / (shigh - slow) * 2 - 1, -1, 1)
        self.assertTrue(np.allclose(out["states"][:, :43], expected_state, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
