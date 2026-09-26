"""LeRobot FLUX3 PEFT/LoRA smoke path over the validated Dex3 dataset view.

Modes:
  resolve  resolved PEFT config, optimizer/scheduler groups and parameter breakdown; no training
  probe    staged single-step memory probe (load -> PEFT -> optimizer -> batch -> step)
  smoke    ``run.steps`` optimizer steps with finiteness/LR/memory checks, then a checkpoint
  reload   fresh-process checkpoint reload and validation forward/inference

The policy, PEFT wrapping, optimizer, scheduler and saved processors all come from LeRobot; this
driver only replaces the dataset factory (the Dex3 view) and the distributed trainer loop.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from flux_action.data.lerobot.dex3_view import CAMERA, Dex3Flux3View  # noqa: E402

REQUIRED_KEYS = {"base_policy", "dataset", "peft", "optimizer", "scheduler", "run", "device", "dtype"}
VALIDATION_SEED = 4242  # shared by every validation pass so raw/EMA and time points stay comparable
PROCESSOR_FILES = ("policy_preprocessor", "policy_postprocessor")


def load_run_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text())
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"config {path} lacks {sorted(missing)}")
    if not config["device"].startswith("cuda"):
        raise ValueError("this probe measures the CUDA path; device must be cuda")
    return config


def host_rss_mib() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def host_peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def smi_used_mib() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return float(out.strip().splitlines()[0])


def cuda_stats() -> dict[str, float]:
    if not torch.cuda.is_initialized():
        return {}
    return {
        "cuda_allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "cuda_reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "cuda_max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "cuda_max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


class Memory:
    """Stage measurements plus a sampler thread that catches peaks between stages."""

    def __init__(self, interval: float = 0.5):
        self.started = time.perf_counter()
        self.records: list[dict] = []
        self.peak_rss_sampled = 0.0
        self.peak_gpu_sampled = 0.0
        self.peak_gpu_max_allocated = 0.0
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    @staticmethod
    def _gpu_used_mib() -> float:
        if not torch.cuda.is_initialized():
            return 0.0
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2**20

    def _sample(self):
        while not self._stop.wait(self._interval):
            self.peak_rss_sampled = max(self.peak_rss_sampled, host_rss_mib())
            try:
                self.peak_gpu_sampled = max(self.peak_gpu_sampled, self._gpu_used_mib())
                if torch.cuda.is_initialized():
                    self.peak_gpu_max_allocated = max(
                        self.peak_gpu_max_allocated, torch.cuda.max_memory_allocated() / 2**20
                    )
            except RuntimeError:
                pass

    def measure(self, stage: str, **extra) -> dict:
        record = {
            "stage": stage,
            "elapsed_s": round(time.perf_counter() - self.started, 2),
            "host_rss_mib": round(host_rss_mib(), 1),
            "host_peak_rss_mib": round(host_peak_rss_mib(), 1),
            "gpu_smi_used_mib": smi_used_mib(),
            **{k: round(v, 1) for k, v in cuda_stats().items()},
            **extra,
        }
        self.records.append(record)
        print("MEASURE " + json.dumps(record), flush=True)
        return record

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=5)
        return {
            "peak_host_rss_mib": round(host_peak_rss_mib(), 1),
            "peak_host_rss_sampled_mib": round(self.peak_rss_sampled, 1),
            "peak_gpu_smi_used_mib": round(
                max([r["gpu_smi_used_mib"] for r in self.records] + [self.peak_gpu_sampled]), 1
            ),
            "peak_gpu_max_allocated_mib": round(self.peak_gpu_max_allocated, 1),
        }


def build_policy(config: dict, memory: Memory | None = None):
    """Base policy from the exported package, then LoRA/heads setup through LeRobot's PEFT entry."""
    from lerobot.configs.default import PeftConfig
    from lerobot.policies.flux3.modeling_flux3 import Flux3Policy

    base = str(REPO_ROOT / config["base_policy"])
    policy = Flux3Policy.from_pretrained(base, strict=True)
    if policy.config.use_peft:
        raise ValueError("base package already carries an adapter; expected the untrained base policy")
    if policy.config.dtype != config["dtype"]:
        raise ValueError(f"base policy dtype {policy.config.dtype!r} differs from config {config['dtype']!r}")
    if bool(policy.config.gradient_checkpointing) != bool(config.get("gradient_checkpointing", True)):
        raise ValueError("gradient checkpointing differs between base policy and config")
    if policy.config.compile_model:
        raise ValueError("base policy has compile_model enabled; the smoke config requires it off")
    if memory:
        memory.measure("base_policy_loaded", param_numel=sum(p.numel() for p in policy.parameters()))
    peft_config = PeftConfig(**config["peft"])
    policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(peft_config))
    if memory:
        memory.measure("peft_attached")
    return policy, peft_config


def build_optimizer(config: dict, policy, peft_config):
    """AdamW + FLUX3 two-group schedule through LeRobot's optimizer/scheduler factory."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.optim.optimizers import AdamWConfig
    from lerobot.optim.schedulers import FrozenWarmupConstantSchedulerConfig

    # The two-group LR comes from the policy's own preset (lr * optimizer_lr_heads_multiplier);
    # point it at this config's values before the factory asks for the parameter groups.
    policy.config.optimizer_lr = config["optimizer"]["lr"]
    heads_lr = config["optimizer"].get("heads_lr")
    if heads_lr is not None:
        policy.config.optimizer_lr_heads_multiplier = heads_lr / config["optimizer"]["lr"]
    optimizer_cfg = AdamWConfig(
        lr=config["optimizer"]["lr"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
        weight_decay=config["optimizer"]["weight_decay"],
        grad_clip_norm=config["optimizer"]["grad_clip_norm"],
    )
    scheduler_cfg = FrozenWarmupConstantSchedulerConfig(
        **{key: value for key, value in config["scheduler"].items() if key != "type"}
    )
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="local/dex3-flux3-view", root=REPO_ROOT / config["dataset"]["index_dir"]
        ),
        policy=policy.config,
        output_dir=REPO_ROOT / config["run"]["output_dir"],
        steps=config["run"]["steps"],
        batch_size=config["dataset"]["batch_size"],
        num_workers=0,
        save_checkpoint=False,
        seed=config["run"]["seed"],
        use_policy_training_preset=True,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        peft=peft_config,
    )
    optimizer, scheduler = make_optimizer_and_scheduler(train_cfg, policy)
    return optimizer, scheduler, optimizer_cfg, scheduler_cfg


def parameter_report(policy) -> dict:
    named = list(policy.named_parameters())
    total = sum(p.numel() for _, p in named)
    trainable = [(n, p) for n, p in named if p.requires_grad]
    lora = [(n, p) for n, p in trainable if ".lora_A." in n or ".lora_B." in n]
    heads = [(n, p) for n, p in trainable if "modules_to_save" in n]
    other = [(n, p) for n, p in trainable if (n, p) not in lora and (n, p) not in heads]
    trainable_numel = sum(p.numel() for _, p in trainable)
    return {
        "total_parameters": total,
        "frozen_parameters": total - trainable_numel,
        "lora_trainable_parameters": sum(p.numel() for _, p in lora),
        "head_trainable_parameters": sum(p.numel() for _, p in heads),
        "other_trainable_parameters": sum(p.numel() for _, p in other),
        "other_trainable_names": [n for n, _ in other][:20],
        "total_trainable_parameters": trainable_numel,
        "trainable_percent": round(100 * trainable_numel / total, 4),
        "lora_module_count": len(lora),
        "head_module_count": len(heads),
    }


def lora_details(policy) -> dict:
    kind, lora_config = next(iter(policy.peft_config.items()))
    return {
        "adapter_name": kind,
        "peft_type": str(lora_config.peft_type),
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": lora_config.target_modules,
        "modules_to_save": sorted(lora_config.modules_to_save) if lora_config.modules_to_save else [],
        "init_lora_weights": str(lora_config.init_lora_weights),
        "bias": str(lora_config.bias),
    }


def build_views(config: dict):
    dataset_cfg = config["dataset"]
    source_root = dataset_cfg["source_root"]
    index_dir = REPO_ROOT / dataset_cfg["index_dir"]
    train_view = Dex3Flux3View(
        source_root, index_dir, split=dataset_cfg["train_split"], decoder=dataset_cfg["decoder"]
    )
    val_view = Dex3Flux3View(
        source_root, index_dir, split=dataset_cfg["val_split"], decoder=dataset_cfg["decoder"]
    )
    return train_view, val_view


def window_order(count: int, total: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total, generator=generator)[:count].tolist()


def single_loader(view, indices):
    return DataLoader(Subset(view, indices), batch_size=1, shuffle=False, num_workers=0)


def load_processors(config: dict, policy_config):
    from lerobot.policies.factory import make_pre_post_processors

    base = str(REPO_ROOT / config["base_policy"])
    pre, post = make_pre_post_processors(policy_config, pretrained_path=base)
    return pre, post


def prepare_batch(raw: dict, pre, device: str) -> dict:
    batch = dict(raw)
    batch[CAMERA] = batch[CAMERA].float() / 255.0
    batch = pre(batch)
    if "task" not in batch:
        raise ValueError("preprocessor dropped the task text")
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def train_step(policy, optimizer, scheduler, batch, grad_clip_norm: float, memory: Memory, tag: str) -> dict:
    times = {}
    start = time.perf_counter()
    loss, info = policy(batch)
    times["forward_s"] = time.perf_counter() - start
    memory.measure(f"{tag}_forward", loss=float(loss.detach()))
    start = time.perf_counter()
    loss.backward()
    times["backward_s"] = time.perf_counter() - start
    memory.measure(f"{tag}_backward")
    trainable = [p for p in policy.parameters() if p.requires_grad]
    nonfinite_grads = sum(1 for p in trainable if p.grad is not None and not torch.isfinite(p.grad).all())
    if nonfinite_grads:
        raise RuntimeError(f"{nonfinite_grads} trainable tensors have non-finite gradients")
    start = time.perf_counter()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
    times["clip_s"] = time.perf_counter() - start
    memory.measure(f"{tag}_grad_clip", grad_norm=float(grad_norm.detach()))
    start = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad()
    times["optimizer_s"] = time.perf_counter() - start
    memory.measure(f"{tag}_optimizer_step")
    start = time.perf_counter()
    scheduler.step()
    times["scheduler_s"] = time.perf_counter() - start
    memory.measure(f"{tag}_scheduler_step")
    return {
        "loss": float(loss.detach()),
        "action_mse": float(info["action_mse"]),
        "video_mse": float(info["video_mse"]),
        "n_valid_windows": int(info["n_valid_windows"]),
        "nonfinite_grad_tensors": nonfinite_grads,
        "grad_norm": float(grad_norm.detach()),
        "lrs": [group["lr"] for group in optimizer.param_groups],
        "times": {key: round(value, 3) for key, value in times.items()},
        "step_time_s": round(sum(times.values()), 3),
    }


def report_shapes(batch: dict) -> dict:
    return {key: list(value.shape) for key, value in sorted(batch.items()) if torch.is_tensor(value)}


def run_resolve(config: dict) -> dict:
    memory = Memory(interval=1.0)
    policy, peft_config = build_policy(config, memory)
    optimizer, scheduler, optimizer_cfg, scheduler_cfg = build_optimizer(config, policy, peft_config)
    report = parameter_report(policy)
    lora = lora_details(policy)
    flux_config = policy.config
    groups = [
        {"lr": group["lr"], "numel": sum(p.numel() for p in group["params"])}
        for group in optimizer.param_groups
    ]
    factors = {
        "trunk": [round(scheduler.lr_lambdas[0](step), 6) for step in range(config["run"]["steps"] + 1)],
        "heads": [round(scheduler.lr_lambdas[-1](step), 6) for step in range(config["run"]["steps"] + 1)],
    }
    train_view, val_view = build_views(config)
    resolved = {
        "flux3": {
            "dtype": flux_config.dtype,
            "conditioning": flux_config.conditioning,
            "n_obs_steps": flux_config.n_obs_steps,
            "chunk_size": flux_config.chunk_size,
            "action_representation": flux_config.action_representation,
            "camera_order": flux_config.camera_order,
            "canvas_hw": list(flux_config.canvas_hw),
            "action_dim": flux_config.action_dim,
            "fps": flux_config.fps,
            "augment": flux_config.augment,
            "caption_dropout": flux_config.caption_dropout,
            "gradient_checkpointing": flux_config.gradient_checkpointing,
            "compile_model": flux_config.compile_model,
            "use_peft": flux_config.use_peft,
            "action_loss_weight": flux_config.action_loss_weight,
            "video_loss_weight": flux_config.video_loss_weight,
            "normalization_clip": flux_config.normalization_clip,
            "optimizer_lr": flux_config.optimizer_lr,
            "optimizer_lr_heads_multiplier": flux_config.optimizer_lr_heads_multiplier,
        },
        "peft": lora,
        "optimizer": {"type": "adamw", "groups": groups, "config": dataclasses.asdict(optimizer_cfg)},
        "scheduler": {
            "type": "frozen_warmup_constant",
            "config": dataclasses.asdict(scheduler_cfg),
            "lr_factors": factors,
        },
        "parameters": report,
        "run": {
            "microbatch": config["dataset"]["batch_size"],
            "vae_batch_windows": config["dataset"]["vae_batch_windows"],
            "grad_accumulation_steps": config["dataset"]["grad_accumulation_steps"],
            "effective_batch": config["dataset"]["batch_size"] * config["dataset"]["grad_accumulation_steps"],
            "steps": config["run"]["steps"],
            "pilot_steps": config["run"].get("pilot_steps"),
            "checkpoint_every": config["run"].get("checkpoint_every"),
            "validate_every": config["run"].get("validate_every"),
            "output_dir": config["run"]["output_dir"],
        },
        "ema": ema_settings(config),
        "ema_shadow_parameters": sum(p.numel() for p in trainable_parameters(policy))
        if ema_settings(config)
        else 0,
        "dataset": {"train_windows": len(train_view), "val_windows": len(val_view)},
        "policy_presets": {
            "optimizer": dataclasses.asdict(flux_config.get_optimizer_preset()),
            "scheduler": dataclasses.asdict(flux_config.get_scheduler_preset()),
        },
    }
    print("RESOLVED " + json.dumps(resolved, indent=2, default=str), flush=True)
    memory.stop()
    return resolved


def run_probe(config: dict) -> dict:
    memory = Memory()
    memory.measure("process_start")
    policy, peft_config = build_policy(config, memory)
    report = parameter_report(policy)
    optimizer, scheduler, _, _ = build_optimizer(config, policy, peft_config)
    memory.measure("optimizer_created")
    policy.to(config["device"])
    policy.train()
    memory.measure("policy_on_device")
    pre, _ = load_processors(config, policy.config)
    train_view, _ = build_views(config)
    indices = window_order(1, len(train_view), config["run"]["seed"])
    loader = single_loader(train_view, indices)
    raw = next(iter(loader))
    batch = prepare_batch(raw, pre, config["device"])
    memory.measure("batch_ready", shapes=report_shapes(batch))
    metrics = train_step(
        policy, optimizer, scheduler, batch, config["optimizer"]["grad_clip_norm"], memory, "step0"
    )
    memory.measure("single_step_done", **metrics)
    result = {
        "parameters": report,
        "shapes": report_shapes(batch),
        "metrics": metrics,
        "stages": memory.records,
        "peaks": memory.stop(),
    }
    output_dir = REPO_ROOT / config["run"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_report.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("PROBE " + json.dumps(result, indent=2, default=str), flush=True)
    return result


def run_smoke(config: dict) -> dict:
    steps = config["run"]["steps"]
    memory = Memory()
    memory.measure("process_start")
    policy, peft_config = build_policy(config, memory)
    report = parameter_report(policy)
    optimizer, scheduler, _, _ = build_optimizer(config, policy, peft_config)
    memory.measure("optimizer_created")
    policy.to(config["device"])
    policy.train()
    memory.measure("policy_on_device")
    pre, _ = load_processors(config, policy.config)
    train_view, _ = build_views(config)
    indices = window_order(steps, len(train_view), config["run"]["seed"])
    loader = single_loader(train_view, indices)
    history = []
    for step, raw in enumerate(loader):
        if step >= steps:
            break
        batch = prepare_batch(raw, pre, config["device"])
        metrics = train_step(
            policy, optimizer, scheduler, batch, config["optimizer"]["grad_clip_norm"], memory, f"step{step}"
        )
        metrics["step"] = step
        metrics["episode_index"] = int(raw["episode_index"].item())
        metrics["frame_index"] = int(raw["frame_index"].item())
        metrics["host_rss_mib"] = round(host_rss_mib(), 1)
        metrics["cuda_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20, 1)
        history.append(metrics)
        print("STEP " + json.dumps(metrics), flush=True)
    if len(history) != steps:
        raise RuntimeError(f"loaded {len(history)} of {steps} smoke windows")
    for key in ("loss", "action_mse", "video_mse", "grad_norm"):
        if not all(torch.isfinite(torch.tensor(entry[key])) for entry in history):
            raise RuntimeError(f"non-finite {key} during the smoke run")
    lr_flow = sorted({tuple(round(lr, 10) for lr in entry["lrs"]) for entry in history})
    checkpoint = REPO_ROOT / config["run"]["output_dir"] / f"checkpoint-{steps}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint)
    base = REPO_ROOT / config["base_policy"]
    for name in base.glob("*"):
        if name.name.startswith(PROCESSOR_FILES):
            shutil.copy(name, checkpoint / name.name)
    manifest = {
        "base_policy": config["base_policy"],
        "steps": steps,
        "seed": config["run"]["seed"],
        "peft": lora_details(policy),
        "parameters": report,
        "final_metrics": history[-1],
        "lr_values_seen": [list(values) for values in lr_flow],
    }
    (checkpoint / "peft_smoke_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = {
        "parameters": report,
        "steps": history,
        "lr_values_seen": [list(values) for values in lr_flow],
        "rss_first_mib": history[0]["host_rss_mib"],
        "rss_last_mib": history[-1]["host_rss_mib"],
        "reserved_first_mib": history[0]["cuda_reserved_mib"],
        "reserved_last_mib": history[-1]["cuda_reserved_mib"],
        "peaks": memory.stop(),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
    }
    (REPO_ROOT / config["run"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / config["run"]["output_dir"] / "smoke_report.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    print(
        "SMOKE " + json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=2, default=str),
        flush=True,
    )
    return result


def ema_settings(config: dict) -> dict | None:
    settings = config.get("ema")
    if not settings:
        return None
    if isinstance(settings, dict):
        return settings if settings.get("enabled", True) else None
    return {}


def build_ema(config: dict, policy):
    """diffusers EMAModel over the trainable parameters with the constant decay of an EMAConfig.

    The installed diffusers exposes ``step`` (the LeRobot trainer's ``update`` name was removed),
    and ``min_decay == max_decay`` pins the decay to the requested constant.
    """
    settings = ema_settings(config)
    if settings is None:
        return None
    from diffusers.training_utils import EMAModel

    decay = float(settings.get("decay", 0.999))
    return EMAModel(
        trainable_parameters(policy),
        decay=decay,
        min_decay=decay,
        update_after_step=int(settings.get("update_after_step", 0)),
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75,
    )


def trainable_parameters(policy) -> list:
    return [p for p in policy.parameters() if p.requires_grad]


def evaluate(policy, pre, view, indices, device, *, ema=None, tag: str = "val") -> dict:
    """Mean loss / action MSE / video MSE over the given windows, optionally on the EMA weights."""
    params = trainable_parameters(policy)
    if ema is not None:
        ema.store(params)
        ema.copy_to(params)
    policy.eval()
    # Timesteps are sampled from the global RNG; a fixed seed keeps raw/EMA and successive
    # validation points comparable. The training RNG state is restored afterwards, so validation
    # never perturbs the training trajectory.
    rng_state = torch.get_rng_state()
    torch.manual_seed(VALIDATION_SEED)
    totals = {"loss": 0.0, "action_mse": 0.0, "video_mse": 0.0}
    count = 0
    try:
        with torch.no_grad():
            for raw in single_loader(view, indices):
                batch = prepare_batch(raw, pre, device)
                loss, info = policy(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{tag}: non-finite validation loss at window {count}")
                totals["loss"] += float(loss)
                totals["action_mse"] += float(info["action_mse"])
                totals["video_mse"] += float(info["video_mse"])
                count += 1
    finally:
        torch.set_rng_state(rng_state)
        if ema is not None:
            ema.restore(params)
        policy.train()
    if count != len(indices):
        raise RuntimeError(f"{tag}: evaluated {count} of {len(indices)} windows")
    return {key: value / count for key, value in totals.items()}


def save_train_checkpoint(
    config, output_dir, policy, ema, optimizer, scheduler, step, history, validations
) -> Path:
    """Raw adapter (adapter_model.safetensors), EMA adapter (ema/), and the resume state."""
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint)
    base = REPO_ROOT / config["base_policy"]
    for name in base.glob("*"):
        if name.name.startswith(PROCESSOR_FILES):
            shutil.copy(name, checkpoint / name.name)
    if ema is not None:
        params = trainable_parameters(policy)
        ema.store(params)
        ema.copy_to(params)
        try:
            ema_dir = checkpoint / "ema"
            ema_dir.mkdir(exist_ok=True)
            policy.save_pretrained(ema_dir)
        finally:
            ema.restore(params)
        torch.save(ema.state_dict(), checkpoint / "ema_state.pt")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    scheduler_state = {key: value for key, value in scheduler.state_dict().items() if key != "lr_lambdas"}
    torch.save(scheduler_state, checkpoint / "scheduler.pt")
    torch.save(
        {"rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all()},
        checkpoint / "rng_state.pt",
    )
    state = {
        "step": step,
        "base_policy": config["base_policy"],
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
        "microbatch": config["dataset"]["batch_size"],
        "grad_accumulation_steps": config["dataset"]["grad_accumulation_steps"],
        "effective_batch": config["dataset"]["batch_size"] * config["dataset"]["grad_accumulation_steps"],
        "ema": ema_settings(config),
        "last_train": history[-1] if history else None,
        "validation": validations[-1] if validations else None,
    }
    (checkpoint / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
    return checkpoint


def restore_train_state(config: dict, policy, optimizer, scheduler, ema, checkpoint: Path) -> int:
    """Adapter, optimizer, scheduler and EMA shadow of a checkpoint written by ``save_train_checkpoint``."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    device = config["device"]
    set_peft_model_state_dict(policy, load_file(checkpoint / "adapter_model.safetensors"))
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device))
    # The schedule closures are not picklable, so the saved state omits them and the freshly built
    # scheduler keeps its own lambdas (a None entry is left untouched by load_state_dict).
    scheduler_state = torch.load(checkpoint / "scheduler.pt", map_location="cpu")
    scheduler_state["lr_lambdas"] = [None] * len(scheduler.lr_lambdas)
    scheduler.load_state_dict(scheduler_state)
    if ema is not None:
        ema.load_state_dict(torch.load(checkpoint / "ema_state.pt", map_location=device))
    return int(json.loads((checkpoint / "trainer_state.json").read_text())["step"])


def start_tensorboard(config: dict):
    """Scalar writer for ``run.tensorboard_dir`` (None keeps logging off)."""
    relative = config["run"].get("tensorboard_dir")
    if not relative:
        return None
    from torch.utils.tensorboard import SummaryWriter

    path = REPO_ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    print("TENSORBOARD " + json.dumps({"dir": str(path)}), flush=True)
    return SummaryWriter(log_dir=str(path))


def log_step_scalars(writer, step: int, entry: dict) -> None:
    if writer is None:
        return
    writer.add_scalar("train/loss", entry["loss"], step)
    writer.add_scalar("train/action_mse", entry["action_mse"], step)
    writer.add_scalar("train/video_mse", entry["video_mse"], step)
    writer.add_scalar("train/grad_norm", entry["grad_norm"], step)
    writer.add_scalar("train/lr_lora", entry["lrs"][0], step)
    writer.add_scalar("train/lr_head", entry["lrs"][-1], step)
    writer.add_scalar("train/step_time_s", entry["step_time_s"], step)
    writer.add_scalar("train/host_rss_gib", entry["host_rss_mib"] / 1024, step)
    writer.add_scalar("train/vram_reserved_gib", entry["cuda_reserved_mib"] / 1024, step)
    writer.add_scalar("train/vram_max_reserved_gib", entry["cuda_max_reserved_mib"] / 1024, step)


def log_validation_scalars(writer, point: dict) -> None:
    if writer is None:
        return
    for mode in ("raw", "ema"):
        if mode in point:
            writer.add_scalar(f"val/{mode}_loss", point[mode]["loss"], point["step"])
            writer.add_scalar(f"val/{mode}_action_mse", point[mode]["action_mse"], point["step"])
            writer.add_scalar(f"val/{mode}_video_mse", point[mode]["video_mse"], point["step"])


def prune_checkpoints(output_dir: Path, keep: int, newest: Path) -> list[str]:
    """Retention: delete checkpoints older than the newest ``keep`` (0 keeps everything)."""
    if keep <= 0:
        return []
    directories = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    removed = []
    for path in directories[: max(0, len(directories) - keep)]:
        if path == newest:
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


def run_pilot(config: dict, *, resume_from: Path | None = None, max_steps: int | None = None) -> dict:
    """Gradient-accumulated training with EMA, periodic validation and checkpoints; stops at pilot_steps.

    ``resume_from`` restores the adapter, optimizer, scheduler, EMA shadow and the data position of that
    checkpoint; ``max_steps`` bounds the further optimizer steps this session runs (default: pilot_steps
    for a fresh session, or the remaining steps to ``run.steps`` for a resumed one).
    """
    steps = int(config["run"]["steps"])
    accumulation = int(config["dataset"]["grad_accumulation_steps"])
    start_step = 0
    if resume_from is not None:
        start_step = int(json.loads((resume_from / "trainer_state.json").read_text())["step"])
    limit = (
        max_steps
        if max_steps is not None
        else (steps if resume_from is not None else int(config["run"].get("pilot_steps", steps)))
    )
    target = min(steps, start_step + limit)
    checkpoint_every = int(config["run"].get("checkpoint_every", 0) or 0)
    validate_every = int(config["run"].get("validate_every", 0) or 0)
    val_windows = int(config["run"].get("val_windows", 32))
    output_dir = REPO_ROOT / config["run"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_checkpoints = int(config["run"].get("keep_last_checkpoints", 0) or 0)
    writer = start_tensorboard(config)
    memory = Memory()
    memory.measure("process_start")
    policy, peft_config = build_policy(config, memory)
    report = parameter_report(policy)
    optimizer, scheduler, _, _ = build_optimizer(config, policy, peft_config)
    memory.measure("optimizer_created")
    policy.to(config["device"])
    policy.train()
    # The EMA shadow clones the parameters, so it must be built after the device move.
    ema = build_ema(config, policy)
    memory.measure("policy_on_device", ema_enabled=ema is not None)
    if resume_from is not None:
        loaded_step = restore_train_state(config, policy, optimizer, scheduler, ema, resume_from)
        if loaded_step != start_step:
            raise ValueError(f"checkpoint step {loaded_step} differs from {start_step}")
        memory.measure("resume_state_loaded", step=start_step)
    pre, _ = load_processors(config, policy.config)
    train_view, val_view = build_views(config)
    # The window order is a pure function of (seed, configured steps), so a resumed session
    # consumes exactly the windows a single 2500-step run would have.
    train_indices = window_order(steps * accumulation, len(train_view), config["run"]["seed"])
    # Drop the consumed prefix from the index list itself; skipping inside the loader would decode
    # every discarded window.
    stream = iter(single_loader(train_view, train_indices[start_step * accumulation :]))
    val_indices = window_order(val_windows, len(val_view), config["run"]["seed"] + 1)

    validations: list[dict] = []
    if validate_every and start_step == 0:
        point = {
            "step": 0,
            "raw": evaluate(policy, pre, val_view, val_indices, config["device"], tag="val-0"),
        }
        if ema is not None:
            point["ema"] = evaluate(
                policy, pre, val_view, val_indices, config["device"], ema=ema, tag="val-0-ema"
            )
        validations.append(point)
        print("VAL " + json.dumps(point), flush=True)
        log_validation_scalars(writer, point)

    history: list[dict] = []
    checkpoints: list[str] = []
    for step in range(start_step + 1, target + 1):
        start = time.perf_counter()
        micro = {"loss": [], "action_mse": [], "video_mse": []}
        for _ in range(accumulation):
            batch = prepare_batch(next(stream), pre, config["device"])
            loss, info = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at optimizer step {step}")
            (loss / accumulation).backward()
            micro["loss"].append(float(loss.detach()))
            micro["action_mse"].append(float(info["action_mse"]))
            micro["video_mse"].append(float(info["video_mse"]))
        nonfinite = sum(
            1 for p in trainable_parameters(policy) if p.grad is not None and not torch.isfinite(p.grad).all()
        )
        if nonfinite:
            raise RuntimeError(f"{nonfinite} non-finite gradient tensors at optimizer step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters(policy), config["optimizer"]["grad_clip_norm"]
        )
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if ema is not None:
            ema.step(trainable_parameters(policy))
        entry = {
            "step": step,
            "loss": sum(micro["loss"]) / accumulation,
            "action_mse": sum(micro["action_mse"]) / accumulation,
            "video_mse": sum(micro["video_mse"]) / accumulation,
            "grad_norm": float(grad_norm.detach()),
            "lrs": [group["lr"] for group in optimizer.param_groups],
            "step_time_s": round(time.perf_counter() - start, 3),
            "host_rss_mib": round(host_rss_mib(), 1),
            "cuda_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
            "cuda_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
            "cuda_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }
        history.append(entry)
        print("OPTSTEP " + json.dumps(entry), flush=True)
        log_step_scalars(writer, step, entry)
        if validate_every and step % validate_every == 0:
            point = {
                "step": step,
                "raw": evaluate(policy, pre, val_view, val_indices, config["device"], tag=f"val-{step}"),
            }
            if ema is not None:
                point["ema"] = evaluate(
                    policy, pre, val_view, val_indices, config["device"], ema=ema, tag=f"val-{step}-ema"
                )
            validations.append(point)
            print("VAL " + json.dumps(point), flush=True)
            log_validation_scalars(writer, point)
        if checkpoint_every and step % checkpoint_every == 0:
            checkpoint = save_train_checkpoint(
                config, output_dir, policy, ema, optimizer, scheduler, step, history, validations
            )
            checkpoints.append(str(checkpoint.relative_to(REPO_ROOT)))
            removed = prune_checkpoints(output_dir, keep_checkpoints, checkpoint)
            print(
                "CHECKPOINT " + json.dumps({"step": step, "dir": checkpoints[-1], "pruned": removed}),
                flush=True,
            )

    if validate_every and (not validations or validations[-1]["step"] != target):
        point = {
            "step": target,
            "raw": evaluate(policy, pre, val_view, val_indices, config["device"], tag="val-final"),
        }
        if ema is not None:
            point["ema"] = evaluate(
                policy, pre, val_view, val_indices, config["device"], ema=ema, tag="val-final-ema"
            )
        validations.append(point)
        print("VAL " + json.dumps(point), flush=True)
        log_validation_scalars(writer, point)
    if not checkpoints or not checkpoints[-1].endswith(f"checkpoint-{target}"):
        checkpoint = save_train_checkpoint(
            config, output_dir, policy, ema, optimizer, scheduler, target, history, validations
        )
        checkpoints.append(str(checkpoint.relative_to(REPO_ROOT)))
        removed = prune_checkpoints(output_dir, keep_checkpoints, checkpoint)
        print(
            "CHECKPOINT " + json.dumps({"step": target, "dir": checkpoints[-1], "pruned": removed}),
            flush=True,
        )
    if writer is not None:
        writer.close()

    peaks = memory.stop()
    tail = history[-50:]
    result = {
        "parameters": report,
        "configured_steps": steps,
        "executed_steps": target,
        "microbatch": config["dataset"]["batch_size"],
        "grad_accumulation_steps": accumulation,
        "effective_batch": config["dataset"]["batch_size"] * accumulation,
        "ema": ema_settings(config),
        "steps": history,
        "validations": validations,
        "checkpoints": checkpoints,
        "summary": {
            "loss_first10": mean(entry["loss"] for entry in history[:10]),
            "loss_last10": mean(entry["loss"] for entry in history[-10:]),
            "loss_last50": mean(entry["loss"] for entry in tail),
            "action_mse_last10": mean(entry["action_mse"] for entry in history[-10:]),
            "video_mse_last10": mean(entry["video_mse"] for entry in history[-10:]),
            "grad_norm_mean": mean(entry["grad_norm"] for entry in history),
            "grad_norm_max": max(entry["grad_norm"] for entry in history),
            "grad_norm_last50": mean(entry["grad_norm"] for entry in tail),
            "step_time_mean_s": mean(entry["step_time_s"] for entry in history),
            "lrs": sorted({tuple(entry["lrs"]) for entry in history}),
            "rss_first_mib": history[0]["host_rss_mib"],
            "rss_last_mib": history[-1]["host_rss_mib"],
            "reserved_first_mib": history[0]["cuda_reserved_mib"],
            "reserved_last_mib": history[-1]["cuda_reserved_mib"],
            "nonfinite": 0,
        },
        "peaks": peaks,
    }
    (output_dir / "pilot_report.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        "PILOT "
        + json.dumps(
            {k: v for k, v in result.items() if k not in ("steps", "validations")}, indent=2, default=str
        ),
        flush=True,
    )
    return result


def mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else 0.0


def latest_checkpoint(config: dict) -> Path:
    output_dir = REPO_ROOT / config["run"]["output_dir"]
    candidates = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"no checkpoint under {output_dir}")
    return candidates[-1]


def run_reload(config: dict, checkpoint: Path | None = None, use_ema: bool = False) -> dict:
    """Fresh process: reload base + adapter, then run a real validation sample through the model."""
    from lerobot.policies.flux3.modeling_flux3 import Flux3Policy
    from peft import PeftConfig as LibraryPeftConfig
    from peft import PeftModel

    memory = Memory()
    memory.measure("process_start")
    checkpoint = checkpoint or latest_checkpoint(config)
    if use_ema:
        checkpoint = checkpoint / "ema"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    adapter_config = LibraryPeftConfig.from_pretrained(checkpoint)
    base = str(REPO_ROOT / config["base_policy"])
    if adapter_config.base_model_name_or_path:
        base = adapter_config.base_model_name_or_path
    policy = Flux3Policy.from_pretrained(base, strict=True)
    policy = PeftModel.from_pretrained(policy, str(checkpoint), config=adapter_config, is_trainable=False)
    memory.measure("checkpoint_loaded", base=base)
    policy.to(config["device"])
    policy.eval()
    memory.measure("policy_on_device")
    pre, post = load_processors(config, policy.config)
    _, val_view = build_views(config)
    indices = window_order(1, len(val_view), config["run"]["seed"] + 1)
    raw = next(iter(single_loader(val_view, indices)))
    batch = prepare_batch(raw, pre, config["device"])
    with torch.no_grad():
        loss, info = policy(batch)
    memory.measure(
        "val_forward",
        loss=float(loss),
        action_mse=float(info["action_mse"]),
        video_mse=float(info["video_mse"]),
    )
    observation = dict(raw)
    observation[CAMERA] = observation[CAMERA].float() / 255.0
    observation[CAMERA] = observation[CAMERA][:, :1]
    observation["observation.state"] = observation["observation.state"][:, :1]
    observation.pop("action", None)
    observation = pre(observation)
    observation = {
        key: (value.to(config["device"]) if torch.is_tensor(value) else value)
        for key, value in observation.items()
    }
    with torch.no_grad():
        predicted = policy.predict_action_chunk(observation)
        absolute = post(predicted)
    memory.measure("val_inference")
    if predicted.shape != (1, policy.config.chunk_size, policy.config.action_dim):
        raise ValueError(f"unexpected predicted chunk shape {tuple(predicted.shape)}")
    checks = {
        "loss_finite": bool(torch.isfinite(loss)),
        "loss": float(loss),
        "action_mse": float(info["action_mse"]),
        "video_mse": float(info["video_mse"]),
        "predicted_shape": list(predicted.shape),
        "absolute_shape": list(absolute.shape),
        "predicted_finite": bool(torch.isfinite(predicted).all()),
        "absolute_finite": bool(torch.isfinite(absolute).all()),
        "absolute_min": float(absolute.min()),
        "absolute_max": float(absolute.max()),
        "raw_action_min": float(batch["action"].min()),
        "raw_action_max": float(batch["action"].max()),
        "task": batch["task"][0],
        "first_absolute_action": [round(float(v), 6) for v in absolute[0, 0][:6]],
        "first_predicted_action": [round(float(v), 6) for v in predicted[0, 0][:6]],
    }
    result = {"checks": checks, "peaks": memory.stop(), "stages": memory.records}
    print("RELOAD " + json.dumps(result, indent=2, default=str), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/dex3/peft_smoke.json")
    parser.add_argument("--mode", choices=("resolve", "probe", "smoke", "pilot", "reload"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="reload mode: checkpoint directory")
    parser.add_argument("--ema", action="store_true", help="reload mode: use the EMA adapter weights")
    parser.add_argument("--resume-from", type=Path, default=None, help="pilot mode: continue this checkpoint")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="pilot mode: optimizer steps this session"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="override run.output_dir")
    args = parser.parse_args()
    config = load_run_config(args.config)
    if args.output_dir is not None:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
        config = {**config, "run": {**config["run"], "output_dir": str(output_dir)}}
    torch.manual_seed(config["run"]["seed"])
    if args.mode == "resolve":
        run_resolve(config)
    elif args.mode == "probe":
        run_probe(config)
    elif args.mode == "smoke":
        run_smoke(config)
    elif args.mode == "pilot":
        resume_from = args.resume_from
        if resume_from is not None and not resume_from.is_absolute():
            resume_from = REPO_ROOT / resume_from
        if resume_from is not None and not resume_from.is_dir():
            raise FileNotFoundError(f"no checkpoint at {resume_from}")
        run_pilot(config, resume_from=resume_from, max_steps=args.max_steps)
    else:
        run_reload(config, args.checkpoint, args.ema)


if __name__ == "__main__":
    main()
