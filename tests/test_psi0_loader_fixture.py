"""Fixture checks against Psi0's real config and transform chain.

These need the psi0 interpreter (``./dev.sh psi0-tests``) because they import
``psi``.  They do not need the produced pack: the wrapper's argv is resolved
through Psi0's own ``finetune_sonic_psi0_config`` and the repack/field transforms
run on one synthetic frame, which is enough to pin the 43->45 state padding,
the 64+14->80 action padding and the strict validity mask before any dataset
exists.  ``loader``/``model`` stay in the verifier -- they need the pack and the
warm-start weights.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from humanoid_lab.datasets.psi0 import contract  # noqa: E402

WRAPPER = REPO_ROOT / "scripts/psi0-unitree-dex3-sonic-v1.sh"
VERIFIER = REPO_ROOT / "scripts/verify-psi0-unitree-dex3-sonic-v1.py"


def load_verifier():
    name = "verify_psi0_unitree_dex3_sonic_v1"
    spec = importlib.util.spec_from_file_location(name, VERIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so dataclasses can resolve cls.__module__.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def wrapper_argv(*overrides: str) -> list[str]:
    """Resolve the wrapper's argv; --print-args returns before any preflight."""
    env = dict(os.environ, **dict(pair.split("=", 1) for pair in overrides))
    resolved = subprocess.run(
        ["bash", str(WRAPPER), "--print-args"], cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=True
    )
    return [line for line in resolved.stdout.splitlines() if line.strip()]


def has_psi() -> bool:
    return importlib.util.find_spec("psi") is not None


def has_lerobot() -> bool:
    return importlib.util.find_spec("lerobot") is not None


def build_synthetic_pack(root: Path, *, episodes: int = 1, frames: int = 40, invalid_tail: int = 5) -> None:
    """A tiny pack that satisfies the contract, so the loader check is runnable now.

    The last ``invalid_tail`` frames of every episode are marked invalid, which is
    the layout the plan's ``anchor_valid[i] = all(valid_30[i:i+30])`` rule exists
    for and the only place a mask bug can show up.
    """
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
    rng = np.random.default_rng(20260917)
    for split in (contract.TRAIN_REPO_ID, contract.VAL_REPO_ID):
        dataset = LeRobotDataset.create(
            repo_id=split, root=root / split, fps=30, features=features, use_videos=True, robot_type="Unitree_G1"
        )
        for _ in range(episodes):
            for index in range(frames):
                frame_valid = index < frames - invalid_tail
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
                        # The strict anchor mask: the whole 30-frame window must be valid.
                        # LeRobot's write path wants the declared [1] shape as an array.
                        contract.ANCHOR_MASK_KEY: np.array(
                            [bool(frame_valid and index + contract.ACTION_CHUNK <= frames)], dtype=bool
                        ),
                        contract.INSTRUCTION_KEY: "pick the doll up",
                    },
                    task="pick the doll up",
                )
            dataset.save_episode()
        # The real pack ships global min/max under meta/stats_psi0.json; this
        # LeRobot build writes per-episode stats, so aggregate them the same way.
        rows = [
            json.loads(line)
            for line in (root / split / "meta/episodes_stats.jsonl").read_text(encoding="utf-8").strip().splitlines()
        ]
        stats: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            for feature, block in row["stats"].items():
                if "min" not in block:
                    continue
                entry = stats.setdefault(feature, {"min": list(block["min"]), "max": list(block["max"])})
                entry["min"] = [min(a, b) for a, b in zip(entry["min"], block["min"])]
                entry["max"] = [max(a, b) for a, b in zip(entry["max"], block["max"])]
        (root / split / "meta" / contract.STATS_FILENAME).write_text(json.dumps(stats), encoding="utf-8")
        (root / split / "meta" / "modality.json").write_text("{}", encoding="utf-8")


@unittest.skipUnless(has_psi(), "needs the psi0 environment (./dev.sh psi0-tests)")
class Psi0LoaderFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verify = load_verifier()
        cls.tmp = tempfile.TemporaryDirectory()
        # Placeholder pack root: --print-args never touches the dataset.
        tokens = wrapper_argv(f"DATASET_ROOT={Path(cls.tmp.name) / contract.DATASET_DIR}", "AUG=0")
        cls.args_file = Path(cls.tmp.name) / "args.txt"
        cls.args_file.write_text("\n".join(tokens), encoding="utf-8")
        cls.ctx = cls.verify.build_context(tokens)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_wrapper_argv_is_accepted_by_psi0(self) -> None:
        self.assertEqual(self.ctx.cfg.train.name, "finetune")
        self.assertEqual(self.ctx.cfg.data.transform.repack.action_mask_key, contract.MASK_KEY)

    def test_config_matches_the_contract(self) -> None:
        self.verify.check_config(self.ctx)

    def test_repack_padding_and_validity(self) -> None:
        notes = self.verify.check_transform(self.ctx)
        self.assertTrue(any("invalid cells closed" in note for note in notes), notes)

    def test_aug_flag_switches_every_augmentation(self) -> None:
        tokens = wrapper_argv(f"DATASET_ROOT={Path(self.tmp.name) / contract.DATASET_DIR}", "AUG=1")
        ctx = self.verify.build_context(tokens)
        self.assertTrue(ctx.cfg.data.transform.model.img_aug)
        self.assertTrue(ctx.cfg.data.transform.model.view_aug)
        self.assertAlmostEqual(ctx.cfg.model.state_drop_prob, 0.1)
        self.assertEqual(ctx.cfg.data.transform.repack.state_temporal_jitter, 10)
        self.verify.check_config(ctx)

    def test_warm_start_header_matches_the_action_contract(self) -> None:
        if not (self.ctx.init_dir / "action_header.safetensors").is_file():
            self.skipTest(f"warm-start checkpoint is not fetched: {self.ctx.init_dir}")
        self.verify.check_ckpt(self.ctx)


@unittest.skipUnless(has_psi() and has_lerobot(), "needs the psi0 environment (./dev.sh psi0-tests)")
class SyntheticPackLoaderTest(unittest.TestCase):
    """Gate 2 on a fixture pack, so the loader path is verified before the real one lands."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.verify = load_verifier()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pack = Path(cls.tmp.name) / contract.DATASET_DIR
        if not cls._checkpoint_present():
            cls.ctx = None
            return
        build_synthetic_pack(cls.pack)
        cls.ctx = cls.verify.build_context(wrapper_argv(f"DATASET_ROOT={cls.pack}", "AUG=0"))

    @classmethod
    def _checkpoint_present(cls) -> bool:
        ckpt = Path(
            os.environ.get(
                "INIT_DIR",
                f"{os.environ.get('PSI_HOME', '/hfm')}/cache/checkpoints/psi0/postpre.sonic1.0.unifolm.2609092156.40k",
            )
        )
        # check_loader needs the real VLM processor, which ships with the checkpoint.
        return (ckpt / "config.json").is_file()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        if self.ctx is None:
            self.skipTest("warm-start checkpoint is not fetched; check_loader needs its processor")

    def test_fixture_pack_satisfies_the_contract(self) -> None:
        self.verify.check_contract(self.ctx)

    def test_real_loader_chain_pads_and_closes_invalid_targets(self) -> None:
        notes = self.verify.check_loader(self.ctx)
        self.assertTrue(any("closed cells" in note for note in notes), notes)


if __name__ == "__main__":
    unittest.main()
