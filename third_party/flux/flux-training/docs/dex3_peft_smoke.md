# G1 Dex3 LeRobot PEFT/LoRA path

`examples/dex3/peft_smoke.py` drives rank-32 LoRA finetuning of the exported base policy
`outputs/dex3/g1-base-policy` over the `Dex3Flux3View` train/validation views. The policy, the PEFT
wrapping (`wrap_with_peft`), the saved pre/post-processors and the optimizer and scheduler presets all
come from LeRobot; this driver supplies the dataset factory the upstream trainer cannot (the Dex3 view is
not a LeRobot repository) plus a single-device loop.

Two configs share the verified model/data settings (single `cam_left_high` camera at 192x256, 30 Hz, 28D
measured state, 28D absolute chunk-32 actions, the package's q01/q99 processors, augmentation on):

* `configs/dex3/peft_smoke.json` — 20-step feasibility run, accumulation 1, no EMA.
* `configs/dex3/peft_train.json` — production settings: microbatch 1, gradient accumulation 8
  (effective batch 8), LoRA LR 1e-4, G1 head LR 5e-4, weight decay 0, grad clip 1.0, 2500 optimizer
  steps, constant LR, EMA decay 0.999, checkpoint and validation every 125 steps over 32 fixed
  validation windows. `run.pilot_steps` stops a session early (the 200-step pilot used it).

Frozen: the FLUX trunk (6.95B parameters) and both frozen encoders. Trained: LoRA on
`q_proj|k_proj|v_proj|attn_out|mlp_in|mlp_out` (the FLUX3 default targets, 98.99M) and the G1/Dex3 heads
in `modules_to_save` (38.09M) — 137.08M trainable, 1.935% of 7.08B. The trunk is bf16 with activation
checkpointing; `torch.compile` stays off.

Modes: `resolve`, `probe`, `smoke`, `pilot`, `reload`; `pilot` also takes `--resume-from`, `--max-steps`
and `--output-dir`, `reload` takes `--checkpoint` and `--ema`.

```sh
cd /home/aksoy-lab/code/flux-training/flux-action
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /tmp/lerobot-peft-env/bin/python examples/dex3/peft_smoke.py --config configs/dex3/peft_train.json --mode pilot
... --mode pilot --resume-from outputs/dex3/peft-train/checkpoint-200        # continue to run.steps
... --mode reload --checkpoint outputs/dex3/peft-train/checkpoint-200 [--ema]
```

The retained runtime is the LeRobot checkout `/tmp/lerobot-peft` and its environment
`/tmp/lerobot-peft-env` (diffusers is installed for the EMA shadow). Recreate it with
`uv venv --python 3.12 /tmp/lerobot-peft-env` and
`uv pip install --python /tmp/lerobot-peft-env/bin/python --torch-backend cu128 -e '/tmp/lerobot-peft[flux3,dataset,peft]' torch==2.10.0`,
then `natten==0.21.6+torch2100cu128` from `https://whl.natten.org/` and `diffusers`.

Measured on the RTX 5090 (32,607 MiB): the 20-step smoke peaked at 30,061 MiB host RSS (the base-policy
load; ~2.7 GiB steady afterwards) and 27,959 MiB device memory. The 200-step pilot peaked at 30,074 MiB
host RSS and 28,658 MiB device memory, with the allocator reserved bytes flat at 27,622 MiB after the
first ~25 steps; host RSS grew linearly by ~0.5 MiB per optimizer step (PyAV decoding runs in-process).
Pilot trends: training loss 8.15 (steps 1-10) to 2.05 (151-200), action MSE 2.09 to 0.51, video MSE
0.144 to 0.092, unclipped grad norm 34.5 to 18.8 (max 61.3), 3.14 s per optimizer step. Validation fell
8.88 -> 3.31 -> 1.65 at steps 0/125/200; the EMA shadow (decay 0.999) still sat at 8.48 at step 200
because 0.999^200 keeps most of its mass on the initial weights. `expandable_segments` is the only memory
setting; nothing is offloaded and the architecture is unchanged.

Each checkpoint holds the raw adapter (`adapter_model.safetensors`), the EMA adapter under `ema/`, the
processor pipelines with their quantile state, and the resume state `ema_state.pt`, `optimizer.pt`,
`scheduler.pt`, `rng_state.pt`, `trainer_state.json`. `outputs/dex3/peft-train/pilot_report.json` keeps
every optimizer step and validation point.

Notes: the installed diffusers exposes `EMAModel.step` (the LeRobot trainer calls the older `update`), and
the EMA shadow must be built after the device move. Validation runs under a fixed seed and restores the
training RNG, so its points stay comparable. The upstream `lerobot-train` still needs a dataset entry
point for `Dex3Flux3View` before it can drive this path.
