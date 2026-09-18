#!/usr/bin/env python3
"""Gate checks for the Psi0 Unitree Dex3 SONIC v1 training line.

This is the verification half of ``scripts/psi0-unitree-dex3-sonic-v1.sh``: the
wrapper owns the Psi0 argv, this script resolves that exact argv through Psi0's
own config class and checks it against the produced pack, the warm-start
checkpoint and the real transform chain.  Nothing here re-implements the loader;
every value it inspects comes out of ``psi.config.train.finetune_sonic_psi0_config``
or out of the transforms that config builds.

Checks (``--check``, repeatable):

  contract   the pack's meta/stats satisfy the loader contract (plan §7/§9)   Kapı 2
  config     the resolved argv matches the contract and the frozen-VLM plan   Kapı 2-4
  ckpt       the warm-start header matches this action/state contract         Kapı 3
  transform  the real repack+field transforms on a synthetic frame            Kapı 2
  model      VLM frozen, action expert trainable, optimizer set               Kapı 4
  forward    one real no_grad BF16 forward on a real batch (loss, VRAM)       Kapı 3
  loader     train and val batches through the real LeRobot chain             Kapı 2

``contract``/``config``/``ckpt``/``transform`` need no dataset: with the pack
absent, ``transform`` synthesises a stats file at the configured widths so the
padding/validity logic is still exercised.  ``loader`` needs the pack and is the
check that must pass before any forward pass is attempted; ``forward`` needs the
pack, the warm-start checkpoint and a GPU, and builds no optimizer.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from humanoid_lab.datasets.psi0 import contract  # noqa: E402


class CheckError(RuntimeError):
    """A gate criterion failed."""


@dataclass
class Context:
    cfg: Any
    tokens: Sequence[str]
    dataset_root: Path
    init_dir: Path
    stats_path: Path
    mask_key: str
    instruction_key: str

    def flag(self, name: str) -> str | None:
        """Value of ``--name=value`` in the resolved argv, or None."""
        prefix = f"--{name}="
        for token in self.tokens:
            if token.startswith(prefix):
                return token[len(prefix):]
        return None

    def flag_list(self, name: str) -> list[str]:
        """Values of ``--name`` (space separated) in the resolved argv."""
        name = f"--{name}"
        for index, token in enumerate(self.tokens):
            if token == name:
                values: list[str] = []
                for follow in self.tokens[index + 1:]:
                    if follow.startswith("--"):
                        break
                    values.append(follow)
                return values
        return []


def read_args_file(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens:
        raise CheckError(f"{path} holds no Psi0 argv; run the wrapper with --print-args")
    return tokens


def parse_config(tokens: Sequence[str]) -> Any:
    """Resolve the wrapper's argv exactly as third_party/Psi0/scripts/train.py does."""
    import tyro

    tokens = list(tokens)
    try:
        module_name, argv = tokens[0], tokens[1:]
    except IndexError:
        raise CheckError("resolved argv is empty")
    try:
        module = importlib.import_module(f"psi.config.train.{module_name}")
    except ImportError as error:
        raise CheckError(f"cannot import psi.config.train.{module_name}: {error}") from error
    try:
        return tyro.cli(module.DynamicLaunchConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=argv)
    except SystemExit as error:
        raise CheckError(f"Psi0 rejected the resolved argv (see the tyro error above)") from error


def build_context(tokens: Sequence[str]) -> Context:
    cfg = parse_config(tokens)
    dataset_root = Path(cfg.data.root_dir)
    init_dir = Path(str(cfg.model.model_name_or_path))
    stats_path = Path(str(cfg.data.transform.field.stat_path))
    return Context(
        cfg=cfg,
        tokens=tokens,
        dataset_root=dataset_root,
        init_dir=init_dir,
        stats_path=stats_path,
        mask_key=str(cfg.data.transform.repack.action_mask_key),
        instruction_key=str(cfg.data.transform.repack.instruction_key),
    )


# --------------------------------------------------------------------------- Kapı 2
def check_contract(ctx: Context) -> list[str]:
    if not ctx.dataset_root.is_dir():
        raise CheckError(
            f"the pack does not exist yet: {ctx.dataset_root}\n"
            "       it is produced by the dataset owner (plan §3) before Kapı 2 can run"
        )
    try:
        train, val = contract.validate_pack(
            ctx.dataset_root, mask_key=ctx.mask_key, instruction_key=ctx.instruction_key
        )
    except contract.DatasetContractError as error:
        raise CheckError(
            f"{error}\n"
            "       the loader contract is declared in src/humanoid_lab/datasets/psi0/contract.py; "
            "fix the pack or re-point MASK_KEY/INSTRUCTION_KEY"
        ) from error
    return [
        f"train  {train.summary()}",
        f"val    {val.summary()}",
    ]


# --------------------------------------------------------------------------- Kapı 2-4
def check_config(ctx: Context) -> list[str]:
    cfg, repack, field, model = ctx.cfg, ctx.cfg.data.transform.repack, ctx.cfg.data.transform.field, ctx.cfg.model
    train_cfg = ctx.cfg.train
    notes: list[str] = []

    def expect(label: str, got: Any, want: Any) -> None:
        if got != want:
            raise CheckError(f"{label}: resolved {got!r}, expected {want!r}")
        notes.append(f"{label} = {got!r}")

    expect("trainer", train_cfg.name, "finetune")
    expect("data_parallel", train_cfg.data_parallel, "ddp")
    expect("mixed_precision", train_cfg.mixed_precision, "bf16")
    if train_cfg.train_batch_size not in (1, 2):
        raise CheckError(
            f"train_batch_size: resolved {train_cfg.train_batch_size!r}, expected the verified single-GPU range 1..2"
        )
    notes.append(f"train_batch_size = {train_cfg.train_batch_size!r}")

    # The one knob that decides VLM frozen vs tuned (plan §13.4).
    expect("model.tune_vlm", model.tune_vlm, False)
    expect("model.gradient_checkpointing", model.gradient_checkpointing, True)

    expect("model.action_dim", model.action_dim, contract.ACTION_MODEL_DIM)
    expect("model.odim", model.odim, contract.STATE_MODEL_DIM)
    expect("model.action_chunk_size", model.action_chunk_size, contract.ACTION_CHUNK)
    expect("model.num_blocks", model.num_blocks, 12)
    expect("model.vlm_layer_indices", list(model.vlm_layer_indices or []), [3, 5, 8, 10, 12, 14, 17, 19, 21, 23, 26, 28])
    expect("model.rtc (training RTC must be off)", model.rtc, False)
    expect("model.max_delay", model.max_delay, 8)

    # 43->45 and 64+14->80, declared on both the repack and the field transform.
    expect("repack.pad_state_dim", repack.pad_state_dim, contract.STATE_MODEL_DIM)
    expect("repack.pad_action_dim", repack.pad_action_dim, contract.ACTION_MODEL_DIM)
    expect("field.pad_state_dim", field.pad_state_dim, contract.STATE_MODEL_DIM)
    expect("field.pad_action_dim", field.pad_action_dim, contract.ACTION_MODEL_DIM)
    expect("repack.action_chunk_size", repack.action_chunk_size, contract.ACTION_CHUNK)

    expect("repack.image_keys", list(repack.image_keys), [contract.IMAGE_KEY])
    expect("repack.state_keys", list(repack.state_keys), [contract.STATE_KEY])
    expect("repack.action_keys", list(repack.action_keys), [contract.BODY_TOKEN_KEY, contract.HAND_KEY])
    if ctx.mask_key not in (contract.MASK_KEY, *contract.MASK_KEY_ALIASES):
        raise CheckError(
            f"action mask key {ctx.mask_key!r} is not one of "
            f"{(contract.MASK_KEY, *contract.MASK_KEY_ALIASES)}"
        )
    if ctx.instruction_key not in (contract.INSTRUCTION_KEY, *contract.INSTRUCTION_KEY_ALIASES):
        raise CheckError(
            f"instruction key {ctx.instruction_key!r} is not one of "
            f"{(contract.INSTRUCTION_KEY, *contract.INSTRUCTION_KEY_ALIASES)}"
        )
    notes.append(f"action mask key = {ctx.mask_key}, instruction key = {ctx.instruction_key}")
    expect("field.stat_action_keys", list(field.stat_action_keys), [contract.BODY_TOKEN_KEY, contract.HAND_KEY])
    expect("field.stat_state_keys", list(field.stat_state_keys), [contract.STATE_KEY])
    expect("field.normalize_state", field.normalize_state, True)
    expect("field.action_norm_type", field.action_norm_type, "bounds")

    if not ctx.dataset_root.is_absolute() or ctx.dataset_root.name not in (
        contract.DATASET_DIR,
        *contract.DATASET_DIR_ALIASES,
    ):
        raise CheckError(
            f"data.root_dir {ctx.dataset_root} must be an absolute .../{contract.DATASET_DIR} "
            f"or .../{contract.GATE_DATASET_DIR}"
        )
    notes.append(f"data.root_dir = {ctx.dataset_root}")
    expect("data.train_repo_ids", list(ctx.cfg.data.train_repo_ids), [contract.TRAIN_REPO_ID])
    expect("data.val_repo_ids", list(ctx.cfg.data.val_repo_ids), [contract.VAL_REPO_ID])

    expect("model.resize", tuple(ctx.cfg.data.transform.model.resize.size), (240, 320))
    expect("model.center_crop", tuple(ctx.cfg.data.transform.model.center_crop.size), (240, 320))
    expect("log.report_to", ctx.cfg.log.report_to, "wandb")

    if ctx.stats_path.name != contract.STATS_FILENAME:
        raise CheckError(f"stat path {ctx.stats_path} is not meta/{contract.STATS_FILENAME}")
    notes.append(f"stats path = {ctx.stats_path}")
    notes.append(
        "state aug: drop={} jitter={}@{} noise={}; img/view aug {} (AUG=0 is the "
        "deterministic smoke)".format(
            model.state_drop_prob,
            repack.state_temporal_jitter,
            repack.state_temporal_jitter_prob,
            field.state_noise_std,
            "on" if ctx.cfg.data.transform.model.img_aug else "off",
        )
    )
    if model.state_drop_prob == 0.0 and ctx.cfg.data.transform.model.img_aug:
        raise CheckError("AUG=0 must disable every augmentation (plan §14)")
    return notes


# --------------------------------------------------------------------------- Kapı 3
def check_ckpt(ctx: Context) -> list[str]:
    for name in ("config.json", "model.safetensors", "action_header.safetensors"):
        if not (ctx.init_dir / name).is_file():
            raise CheckError(f"warm-start checkpoint is missing {ctx.init_dir / name}")
    config = json.loads((ctx.init_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise CheckError(f"{ctx.init_dir}/config.json has architectures {config.get('architectures')!r}")

    from safetensors import safe_open

    model_cfg = ctx.cfg.model
    with safe_open(ctx.init_dir / "model.safetensors", framework="pt") as handle:
        vlm_tensors = len(handle.keys())
    with safe_open(ctx.init_dir / "action_header.safetensors", framework="pt") as handle:
        keys = list(handle.keys())
        # These two tensors are what Psi0 compares before loading the header whole:
        # a mismatch silently falls back to "transformer blocks only" (see
        # FinetuneTrainer.init_models), i.e. a cold action expert.
        dec_pos = tuple(handle.get_slice("action_proj_in.dec_pos").get_shape())
        proj_out = tuple(handle.get_slice("action_proj_out.linear.weight").get_shape())
        blocks = sorted({int(key.split(".")[1]) for key in keys if key.startswith("transformer_blocks.")})

    if dec_pos[0] != model_cfg.action_chunk_size:
        raise CheckError(
            f"warm-start header predicts {dec_pos[0]} steps, config expects {model_cfg.action_chunk_size}: "
            "Psi0 would drop the action projection and warm-start only the transformer blocks"
        )
    if proj_out[0] != model_cfg.action_dim:
        raise CheckError(
            f"warm-start header outputs {proj_out[0]} dims, config expects {model_cfg.action_dim}: "
            "Psi0 would drop the action projection and warm-start only the transformer blocks"
        )
    expected_blocks = list(range(model_cfg.num_blocks))
    if blocks != expected_blocks:
        raise CheckError(f"warm-start header has transformer blocks {blocks}, config expects {expected_blocks}")
    if len(model_cfg.vlm_layer_indices or []) != model_cfg.num_blocks:
        raise CheckError("num_blocks and the vlm_layer_indices pairing disagreed; Psi0 asserts it at build time")
    return [
        f"model.safetensors: {vlm_tensors} tensors",
        f"action_header.safetensors: {len(keys)} tensors, action_proj_in.dec_pos {dec_pos}, "
        f"action_proj_out.linear.weight {proj_out}, {len(blocks)} blocks (loads whole, not partial)",
    ]


# --------------------------------------------------------------------------- Kapı 2
def _synthetic_stats(path: Path) -> None:
    """A stats file at the contract's *raw* widths, for the dataset-free transform check."""
    stats: dict[str, dict[str, list[float]]] = {}
    for key, width in {**contract.STAT_ACTION_WIDTHS, **contract.STAT_STATE_WIDTHS}.items():
        center = [0.01 * (index + 1) for index in range(width)]
        stats[key] = {
            "min": [value - 0.5 for value in center],
            "max": [value + 0.5 for value in center],
        }
    path.write_text(json.dumps(stats), encoding="utf-8")


def _synthetic_frame(ctx: Context) -> tuple[dict[str, Any], dict[str, Any]]:
    """One raw frame shaped like a LeRobot item, plus the expected repacked blocks."""
    import numpy as np
    import torch

    chunk, token_dim, hand_dim = contract.ACTION_CHUNK, contract.BODY_TOKEN_DIM, contract.HAND_DIM
    rng = np.random.default_rng(20260917)
    frame = {
        contract.IMAGE_KEY: torch.zeros(3, 240, 320, dtype=torch.uint8),
        contract.STATE_KEY: rng.normal(size=(1, contract.STATE_DIM)).astype(np.float32),
        contract.BODY_TOKEN_KEY: rng.normal(size=(chunk, token_dim)).astype(np.float32),
        contract.HAND_KEY: rng.normal(size=(chunk, hand_dim)).astype(np.float32),
        ctx.instruction_key: "pick the doll up",
    }
    # Wide validity: the first 20 targets are supervised, the tail is not.
    mask = np.ones((chunk, contract.ACTION_MODEL_DIM), dtype=np.float32)
    mask[20:] = 0.0
    mask[:, contract.ACTION_DIM:] = 0.0
    frame[ctx.mask_key] = mask
    expected = {
        "token": frame[contract.BODY_TOKEN_KEY].copy(),
        "hand": frame[contract.HAND_KEY].copy(),
        "state": frame[contract.STATE_KEY].copy(),
        "mask": mask.copy(),
    }
    return frame, expected


def check_transform(ctx: Context) -> list[str]:
    import numpy as np

    repack = ctx.cfg.data.transform.repack.model_copy()
    field = ctx.cfg.data.transform.field.model_copy()
    notes: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = ctx.stats_path
        if not stats_path.is_file():
            stats_path = Path(tmp) / contract.STATS_FILENAME
            _synthetic_stats(stats_path)
            notes.append(f"{contract.STATS_FILENAME} absent; used a synthetic stats file at the configured widths")
        field.stat_path = str(stats_path)
        field.populate_stats(json.loads(stats_path.read_text(encoding="utf-8")))

        frame, expected = _synthetic_frame(ctx)
        repacked = repack(dict(frame))
        states = np.asarray(repacked["states"])
        actions = np.asarray(repacked["actions"])
        mask = np.asarray(repacked["actions_mask"])

        if states.shape != (1, contract.STATE_MODEL_DIM):
            raise CheckError(f"repacked states are {states.shape}, expected (1, {contract.STATE_MODEL_DIM})")
        if actions.shape != (contract.ACTION_CHUNK, contract.ACTION_MODEL_DIM):
            raise CheckError(
                f"repacked actions are {actions.shape}, expected ({contract.ACTION_CHUNK}, {contract.ACTION_MODEL_DIM})"
            )
        if not np.array_equal(states[..., : contract.STATE_DIM], expected["state"]):
            raise CheckError("state 0:43 is not the measured observation.state")
        if not np.array_equal(states[..., contract.STATE_DIM:], np.zeros_like(states[..., contract.STATE_DIM:])):
            raise CheckError("state 43:45 must be zero padding")
        if not np.array_equal(actions[:, : contract.BODY_TOKEN_DIM], expected["token"]):
            raise CheckError("action 0:64 is not action.body_token_v1_1")
        if not np.array_equal(actions[:, contract.BODY_TOKEN_DIM: contract.ACTION_DIM], expected["hand"]):
            raise CheckError("action 64:78 is not the Dex3 hand target")
        if not np.array_equal(actions[:, contract.ACTION_DIM:], np.zeros_like(actions[:, contract.ACTION_DIM:])):
            raise CheckError("action 78:80 must be zero padding")

        if mask.shape != (contract.ACTION_CHUNK, contract.ACTION_MODEL_DIM):
            raise CheckError(f"actions_mask is {mask.shape}, expected ({contract.ACTION_CHUNK}, {contract.ACTION_MODEL_DIM})")
        if mask[:, contract.ACTION_DIM:].any():
            raise CheckError("the neck padding columns must stay masked out of the loss")
        invalid = ~expected["mask"][:, : contract.ACTION_DIM].astype(bool)
        if mask[:, : contract.ACTION_DIM][invalid].any():
            raise CheckError("a target the pack marked invalid still reaches the loss")
        if not mask[:, : contract.ACTION_DIM][~invalid].all():
            raise CheckError("a target the pack marked valid was masked out of the loss")
        notes.append(
            f"mask: {int(mask[:, : contract.ACTION_DIM].sum())} supervised cells, "
            f"{int(invalid.sum())} invalid cells closed"
        )

        normalized = field(dict(repacked))
        if not np.isfinite(normalized["states"]).all():
            raise CheckError("normalised states contain NaN/Inf")
        if not np.isfinite(normalized["actions"]).all():
            raise CheckError("normalised actions contain NaN/Inf")
        if np.abs(normalized["actions"]).max() > 1.0 + 1e-6:
            raise CheckError("bounds normalisation left the actions outside [-1, 1]")
        if not np.array_equal(normalized["states"][..., contract.STATE_DIM:], np.zeros((1, 2), dtype=normalized["states"].dtype)):
            raise CheckError("the padded state dims must normalise to zero on the read split")
        if normalized["instruction"] != "pick the doll up":
            raise CheckError(f"instruction did not survive the repack: {normalized['instruction']!r}")
        notes.append(
            f"normalised range: actions [{normalized['actions'].min():.3f}, {normalized['actions'].max():.3f}], "
            f"states [{normalized['states'].min():.3f}, {normalized['states'].max():.3f}]"
        )
    return notes


# --------------------------------------------------------------------------- Kapı 4
def check_model(ctx: Context) -> list[str]:
    import torch

    from psi.trainers import Trainer

    trainer = Trainer.instantiate(ctx.cfg, 0)
    trainer.init_models()

    frozen_vlm = [name for name, parameter in trainer.model.vlm_model.named_parameters() if not parameter.requires_grad]
    vlm_params = [name for name, _ in trainer.model.vlm_model.named_parameters()]
    if len(frozen_vlm) != len(vlm_params):
        raise CheckError(
            f"{len(vlm_params) - len(frozen_vlm)} of {len(vlm_params)} VLM tensors are trainable; "
            "the plan freezes the whole VLM (§13.4)"
        )
    header = [(name, parameter) for name, parameter in trainer.model.named_parameters() if name.startswith("action_header.")]
    if not header:
        raise CheckError("the model exposes no action_header parameters")
    trainable_header = [name for name, parameter in header if parameter.requires_grad]
    if len(trainable_header) != len(header):
        raise CheckError(f"{len(header) - len(trainable_header)} action-expert tensors are frozen; they must all train")

    optimizer = trainer.create_optimizers()
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in trainer.model.parameters() if parameter.requires_grad}
    if optimized != trainable:
        raise CheckError(
            f"the optimizer holds {len(optimized)} tensors but the model has {len(trainable)} trainable ones"
        )
    trainable_count = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    notes = [
        f"VLM frozen: {len(vlm_params)} tensors, 0 trainable",
        f"action expert trainable: {len(header)} tensors, {trainable_count:,} trainable parameters total",
        f"optimizer: {len(optimizer.param_groups)} group(s) over exactly the trainable set, lr {optimizer.param_groups[0]['lr']:g}",
    ]
    if torch.cuda.is_available():
        notes.append(
            f"GPU after model init: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak allocated"
        )
    return notes


# --------------------------------------------------------------------------- Kapı 3
def _select_available_attention_backend() -> None:
    """Install the shared fallback; see ``humanoid_lab.psi0_compat``.

    Psi0's trainer hardcodes ``flash_attention_2`` and raises before the
    checkpoint loads, while its own model loader already picks ``sdpa`` when
    flash-attn is absent.  The real training launch installs the same patch
    through ``scripts/psi0_shim``; this call covers a standalone verifier run.
    """
    from humanoid_lab.psi0_compat import install_sdpa_fallback

    install_sdpa_fallback()


def check_forward(ctx: Context) -> list[str]:
    """One real ``no_grad`` forward through the warm-started model on a real batch.

    The plan's Kapı 3 settings: real mini pack, warm-start ``model.safetensors``
    plus ``action_header.safetensors``, batch 1, VLM frozen, BF16, augmentation
    off. No backward and no optimizer are built -- that is Kapı 4 -- so this
    check cannot move a weight even by accident.
    """
    import numpy as np
    import torch
    from accelerate import Accelerator
    from psi.trainers import Trainer
    from psi.utils import batch_str_to_tensor, move_to_device

    if not ctx.dataset_root.is_dir():
        raise CheckError(f"the pack does not exist yet: {ctx.dataset_root}")
    if not torch.cuda.is_available():
        raise CheckError("Kapı 3 needs a CUDA device")
    if ctx.cfg.train.train_batch_size != 1:
        raise CheckError(f"Kapı 3 runs batch 1, got {ctx.cfg.train.train_batch_size}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    trainer = Trainer.instantiate(ctx.cfg, 0)
    try:
        trainer.init_models()
    except torch.cuda.OutOfMemoryError as error:
        raise CheckError(f"Kapı 3 OOM while loading the warm-start checkpoint: {error}") from error

    trainable_vlm = [name for name, parameter in trainer.model.vlm_model.named_parameters() if parameter.requires_grad]
    if trainable_vlm:
        raise CheckError(
            f"{len(trainable_vlm)} VLM tensors are trainable; the plan freezes the whole VLM (Kapı 3)"
        )

    # A header whose action projections were dropped still forwards, so check the
    # loaded module and not only the file on disk.  ``dec_pos`` is an embedding,
    # so its leading dimension is the predicted horizon (see ``check_ckpt``).
    dec_pos = tuple(int(dim) for dim in trainer.model.action_header.action_proj_in.dec_pos.shape)
    proj_out = tuple(int(dim) for dim in trainer.model.action_header.action_proj_out.linear.weight.shape)
    if dec_pos[0] != ctx.cfg.model.action_chunk_size:
        raise CheckError(
            f"the loaded action header predicts {dec_pos[0]} steps, expected "
            f"{ctx.cfg.model.action_chunk_size}: the warm start dropped action_proj_in"
        )
    if proj_out[0] != ctx.cfg.model.action_dim:
        raise CheckError(
            f"the loaded action header outputs {proj_out[0]} dims, expected "
            f"{ctx.cfg.model.action_dim}: the warm start dropped action_proj_out"
        )

    accelerator = Accelerator(mixed_precision=ctx.cfg.train.mixed_precision)
    trainer.accelerator = accelerator
    trainer.model = accelerator.prepare_model(trainer.model)

    train_dataset, val_dataset = trainer.create_datasets()
    _, val_dataloader = trainer.create_dataloaders(train_dataset, val_dataset)
    if val_dataset.transform_kwargs.get("no_aug") is not True:
        raise CheckError(
            f"Kapı 3 must run with augmentation off; val transform_kwargs={val_dataset.transform_kwargs}"
        )

    # The real loop feeds `trainer.step(batch_str_to_tensor(batch), ...)`; without
    # that decode the collator's `instruction_str` never becomes `instruction` and
    # the pooled-projection conditioning silently arrives as None.
    batch = move_to_device(batch_str_to_tensor(next(iter(val_dataloader))), trainer.device)
    batch_size = int(batch["actions"].shape[0])
    if batch_size != 1:
        raise CheckError(f"the validation batch holds {batch_size} frames, expected 1")
    if "instruction" not in batch:
        raise CheckError("the collated batch carries no instruction string to condition on")

    outputs: list[Any] = []
    handle = trainer.model.register_forward_hook(lambda module, args, output: outputs.append(output))
    try:
        with torch.no_grad(), trainer.forward_autocast():
            losses = trainer.forward_and_loss(trainer.model, batch)
    except torch.cuda.OutOfMemoryError as error:
        raise CheckError(f"Kapı 3 OOM during the forward pass: {error}") from error
    finally:
        handle.remove()

    if not outputs:
        raise CheckError("the model forward produced no output to inspect")
    action = getattr(outputs[-1], "action", None)
    if action is None:
        raise CheckError("the model output carries no `action` field")

    loss = float(torch.as_tensor(losses["loss"]).detach().float())
    if not np.isfinite(loss):
        raise CheckError(f"the forward loss is not finite: {loss}")

    action_shape = tuple(int(dim) for dim in action.shape)
    expected = (1, ctx.cfg.model.action_chunk_size, ctx.cfg.model.action_dim)
    if action_shape != expected:
        raise CheckError(f"the predicted action is {action_shape}, expected {expected} (batch, 30, 80)")
    if not bool(torch.isfinite(action.float()).all()):
        raise CheckError("the predicted action contains NaN/Inf")

    return [
        f"loss {loss:.6f} (finite); predicted action {action_shape} = {expected[1]}x{expected[2]}",
        f"batch {batch_size}; VLM frozen ({len(trainable_vlm)} trainable VLM tensors); augmentation off; no_grad",
        f"loaded action header: action_proj_in.dec_pos {dec_pos}, action_proj_out.linear.weight {proj_out}",
        "CUDA peak allocated "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB / reserved "
        f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB; no OOM",
    ]


# --------------------------------------------------------------------------- Kapı 2
def _validity_probe_indices(dataset: Any) -> list[int]:
    """Anchors that straddle the first episode's invalid tail, where the mask must bite.

    The plan's rule is ``anchor_valid[i] = all(valid_30[i:i+30])``; the loader
    applies it per target frame, so the last few anchors before an episode ends
    are the only place the two differ, and the only place a mask bug shows up.
    """
    chunk = contract.ACTION_CHUNK
    try:
        index = dataset.raw_dataset.episode_data_index
        first, last = int(index["from"][0]), int(index["to"][0])
    except Exception:
        return [0, max(0, len(dataset) // 2), max(0, len(dataset) - 1)]
    picks = {first, max(first, last - chunk - 1), max(first, last - chunk // 2), max(first, last - 1)}
    return sorted(pick for pick in picks if 0 <= pick < len(dataset))


def check_loader(ctx: Context) -> list[str]:
    import numpy as np

    if not ctx.dataset_root.is_dir():
        raise CheckError(f"the pack does not exist yet: {ctx.dataset_root}")
    # FinetuneTrainer.create_datasets builds the datasets with exactly this kwarg;
    # only the processor is needed, so the VLM weights stay off the GPU here.
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(ctx.init_dir))
    transform_kwargs = dict(vlm_processor=processor)
    train = ctx.cfg.data(split="train", transform_kwargs=transform_kwargs)
    val = ctx.cfg.data(split="val", transform_kwargs=transform_kwargs)
    if train.transform_kwargs.get("no_aug") is not False:
        raise CheckError(f"the train split must be augmented-enabled, got {train.transform_kwargs}")
    if val.transform_kwargs.get("no_aug") is not True:
        raise CheckError(f"the val split must run with no_aug=True, got {val.transform_kwargs}")

    notes = [f"train frames {len(train)}, val frames {len(val)}, val no_aug={val.transform_kwargs['no_aug']}"]
    for split, dataset in (("train", train), ("val", val)):
        probes = _validity_probe_indices(dataset)
        closed = supervised = anchors_valid = 0
        for position in probes:
            raw = dataset.raw_dataset[position]
            item = dataset[position]
            states = np.asarray(item["states"])
            actions = np.asarray(item["actions"])
            mask = np.asarray(item["actions_mask"])
            if states.shape != (1, contract.STATE_MODEL_DIM):
                raise CheckError(f"[{split}#{position}] states are {states.shape}, expected (1, 45)")
            if actions.shape != (contract.ACTION_CHUNK, contract.ACTION_MODEL_DIM):
                raise CheckError(f"[{split}#{position}] actions are {actions.shape}, expected (30, 80)")
            if mask.shape != actions.shape:
                raise CheckError(f"[{split}#{position}] actions_mask {mask.shape} != actions {actions.shape}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise CheckError(f"[{split}#{position}] normalised tensors contain NaN/Inf")
            if mask[:, contract.ACTION_DIM:].any():
                raise CheckError(f"[{split}#{position}] the neck padding columns reach the loss")

            # The strict validity invariant, in both directions: a target the pack
            # marked invalid must be closed, a target it marked valid must be kept.
            raw_mask = np.asarray(raw[ctx.mask_key]).astype(bool)
            body = mask[:, : contract.ACTION_DIM]
            if body[~raw_mask[:, : contract.ACTION_DIM]].any():
                raise CheckError(f"[{split}#{position}] an invalid target still reaches the loss")
            if not body[raw_mask[:, : contract.ACTION_DIM]].all():
                raise CheckError(f"[{split}#{position}] a valid target was masked out of the loss")
            closed += int((~raw_mask[:, : contract.ACTION_DIM]).sum())
            supervised += int(raw_mask[:, : contract.ACTION_DIM].sum())
            anchor = raw.get(contract.ANCHOR_MASK_KEY) if hasattr(raw, "get") else None
            if anchor is not None:
                anchors_valid += int(bool(np.asarray(anchor).reshape(-1)[0]))

            raw_actions = np.asarray(item["raw_actions"])
            expected = np.concatenate(
                [np.asarray(raw[contract.BODY_TOKEN_KEY]), np.asarray(raw[contract.HAND_KEY])], axis=-1
            )
            if not np.allclose(raw_actions[:, : contract.ACTION_DIM], expected):
                raise CheckError(f"[{split}#{position}] raw_actions do not match the pack's token+hand fields")
            if raw_actions[:, contract.ACTION_DIM:].any():
                raise CheckError(f"[{split}#{position}] raw_actions 78:80 must be zero padding")
            if np.asarray(raw[contract.STATE_KEY]).shape[-1] != contract.STATE_DIM:
                raise CheckError(f"[{split}#{position}] the pack's observation.state is not 43D")
            if not isinstance(item["instruction"], str) or not item["instruction"].islower():
                raise CheckError(f"[{split}#{position}] instruction is not lower-cased: {item['instruction']!r}")
            if "input_ids" not in item or "pixel_values" not in item:
                raise CheckError(f"[{split}#{position}] the VLM processor did not produce input_ids/pixel_values")
        notes.append(
            f"{split}: states (1, 45), actions (30, 80) at anchors {probes}; "
            f"{supervised} supervised / {closed} closed cells, {anchors_valid}/{len(probes)} strict anchors open"
            + ("" if closed else " (no invalid target in these anchors: mask stayed fully open)")
        )
    return notes


CHECKS: dict[str, Callable[[Context], list[str]]] = {
    "contract": check_contract,
    "config": check_config,
    "ckpt": check_ckpt,
    "transform": check_transform,
    "model": check_model,
    "forward": check_forward,
    "loader": check_loader,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--args-file", required=True, type=Path, help="Psi0 argv written by the wrapper")
    parser.add_argument("--check", nargs="+", choices=sorted(CHECKS), default=["contract", "ckpt", "config"])
    options = parser.parse_args(argv)

    try:
        tokens = read_args_file(options.args_file)
        ctx = build_context(tokens)
    except CheckError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    print(f"resolved argv: {tokens[0]} ({len(tokens) - 1} flags)")
    # Any check that builds the model needs the shared fallback; applying it once
    # here keeps `--check model`/`forward` self-sufficient outside the wrapper.
    _select_available_attention_backend()
    for name in options.check:
        try:
            notes = CHECKS[name](ctx)
        except CheckError as error:
            print(f"FAIL [{name}]: {error}", file=sys.stderr)
            return 2
        except contract.DatasetContractError as error:
            print(f"FAIL [{name}]: {error}", file=sys.stderr)
            return 2
        print(f"[ok] {name}")
        for note in notes:
            print(f"       {note}")
    print(f"PASS: {' '.join(options.check)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
