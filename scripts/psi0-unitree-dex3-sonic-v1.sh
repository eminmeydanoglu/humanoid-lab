#!/usr/bin/env bash
#
# Psi0 fine-tuning of the Unitree Dex3 SONIC v1 pack on one RTX 5090.
#
# The gate runs of psi_egitime_hazirlik.md §16 all use this one script; only the
# step/horizon knobs change:
#
#   Kapı 2   ./dev.sh psi0-dex3-check                      loader + checkpoint contract
#   Kapı 3-4 MAX_STEPS=1 ACCUM=1 AUG=0 ./dev.sh psi0-dex3-run    one step: fwd + bwd + optimizer
#   Kapı 5   MAX_STEPS=10 CKPT_STEPS=5 AUG=0 ./dev.sh psi0-dex3-run
#            MAX_STEPS=50 ACCUM=32 AUG=1 ./dev.sh psi0-dex3-run
#   resume   RESUME=latest MAX_STEPS=20 ./dev.sh psi0-dex3-run
#   Kapı 7   MAX_STEPS=40000 ACCUM=32 AUG=1 ./dev.sh psi0-dex3-run
#
# The upstream recipe scripts/train/psi0/finetune-real-sonic-psi0-2.8B-sonic1.1-robust.sh
# is NOT copied: it is 8-GPU, tunes the VLM and points at another pack.  This
# wrapper drives the same Psi0 config (finetune_sonic_psi0_config) with the
# single-GPU/frozen-VLM settings and this repo's dataset paths, and reuses Psi0's
# own train.py unchanged -- third_party/Psi0 stays at the pinned commit.
#
# Everything below is an environment knob so that a gate run is reproducible from
# its command line alone.  --print-args writes the resolved Psi0 argv, which is
# what scripts/verify-psi0-unitree-dex3-sonic-v1.py checks; there is exactly one
# place where the flags are written (here).
#
# Dataset paths (inside the dev container):
#   DATASET_ROOT   $HUMANOID_DATA_ROOT/datasets/psi0-unitree-dex3-sonic-v1
#   STATS          $DATASET_ROOT/train/meta/stats_psi0.json
#   INIT_DIR       $PSI_HOME/cache/checkpoints/psi0/postpre.sonic1.0.unifolm.2609092156.40k
#   OUTPUT_DIR     /outputs/psi0-unitree-dex3-sonic-v1

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

print_help() {
  cat <<'HELP'
Train Psi0 on the Unitree Dex3 SONIC v1 pack (one RTX 5090, VLM frozen).

usage: psi0-unitree-dex3-sonic-v1.sh [--print-args | --check-only | --help]

Gate runs (psi_egitime_hazirlik.md §16), all the same script:
  Kapı 2     ./dev.sh psi0-dex3-check                        loader + checkpoint contract
  Kapı 3-4   MAX_STEPS=1 ACCUM=1 AUG=0 ./dev.sh psi0-dex3-run      one step: forward+backward+optimizer
  Kapı 5     MAX_STEPS=10 CKPT_STEPS=5 AUG=0 ./dev.sh psi0-dex3-run
             MAX_STEPS=50 ACCUM=32 AUG=1 ./dev.sh psi0-dex3-run
  resume     RESUME=latest MAX_STEPS=20 ./dev.sh psi0-dex3-run
  Kapı 7     MAX_STEPS=40000 ACCUM=32 AUG=1 ./dev.sh psi0-dex3-run

Environment knobs: MAX_STEPS BATCH ACCUM LR VAL_STEPS CKPT_STEPS VAL_BATCHES
                   AUG STATE_DROP_PROB STATE_JITTER STATE_JITTER_PROB STATE_NOISE_STD
                   RESUME EXP SEED DATASET_ROOT STATS INIT_DIR OUTPUT_DIR
                   INSTRUCTION_KEY MASK_KEY WANDB_PROJECT CUDA_VISIBLE_DEVICES PSI0_SRC
HELP
}

MODE=run
for arg in "$@"; do
  case "$arg" in
    --print-args) MODE=print ;;
    --check-only) MODE=check ;;
    --help|-h)    print_help; exit 0 ;;
    *) echo "usage: $0 [--print-args | --check-only | --help]" >&2; exit 2 ;;
  esac
done

fail() { echo "FAIL: $*" >&2; exit 2; }

# --- paths -------------------------------------------------------------------
DATAROOT="${HUMANOID_DATA_ROOT:-/data}"
DATASET_DIR="${DATASET_DIR:-psi0-unitree-dex3-sonic-v1}"
DATASET_ROOT="${DATASET_ROOT:-$DATAROOT/datasets/$DATASET_DIR}"
TRAIN_ID="${TRAIN_ID:-train}"
VAL_ID="${VAL_ID:-val}"
STATS="${STATS:-$DATASET_ROOT/$TRAIN_ID/meta/stats_psi0.json}"
PSI_HOME_DIR="${PSI_HOME:-/hfm}"
INIT_DIR="${INIT_DIR:-$PSI_HOME_DIR/cache/checkpoints/psi0/postpre.sonic1.0.unifolm.2609092156.40k}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs/$DATASET_DIR}"
PSI0_SRC="${PSI0_SRC:-$REPO_ROOT/third_party/Psi0}"
EXP="${EXP:-unitree-dex3-sonic-v1}"
SEED="${SEED:-292285}"

# --- single-GPU / smoke knobs ------------------------------------------------
MAX_STEPS="${MAX_STEPS:-40000}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-32}"
LR="${LR:-2.5e-5}"
VAL_BATCHES="${VAL_BATCHES:-10}"
CKPT_STEPS="${CKPT_STEPS:-1000}"
VAL_STEPS="${VAL_STEPS:-500}"
RESUME="${RESUME:-}"
if [ "$RESUME" = "latest" ]; then
  RESUME="$({
    find "$OUTPUT_DIR" -type d -name 'ckpt_*' -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | cut -d' ' -f2- | while IFS= read -r candidate; do
    step="${candidate##*/ckpt_}"
    case "$step" in
      ''|*[!0-9]*) continue ;;
    esac
    printf '%s\n' "$candidate"
    break
  done)"
  [ -n "$RESUME" ] || fail "RESUME=latest requested, but no ckpt_N directory exists under $OUTPUT_DIR"
fi
INSTRUCTION_KEY="${INSTRUCTION_KEY:-task_description}"
# The converter writes the strict anchor mask as `anchor_valid` and the
# loader-facing wide mask as `action.mask`; only the latter is a repack input.
MASK_KEY="${MASK_KEY:-action.mask}"

# Final training defaults to the selected robustness augmentation. Smoke commands
# set AUG=0 explicitly for deterministic forward/backward checks.
AUG="${AUG:-1}"
if [ "$AUG" = "1" ]; then
  STATE_DROP_PROB="${STATE_DROP_PROB:-0.1}"
  STATE_JITTER="${STATE_JITTER:-10}"
  STATE_JITTER_PROB="${STATE_JITTER_PROB:-0.5}"
  STATE_NOISE_STD="${STATE_NOISE_STD:-0.05}"
else
  STATE_DROP_PROB="${STATE_DROP_PROB:-0.0}"
  STATE_JITTER="${STATE_JITTER:-0}"
  STATE_JITTER_PROB="${STATE_JITTER_PROB:-0.5}"
  STATE_NOISE_STD="${STATE_NOISE_STD:-0.0}"
fi

# W&B offline (plan §15): no API key needed, the run stays on this machine.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-psi0-unitree-dex3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
# The shim directory is first so `sitecustomize` loads for every interpreter the
# gate starts, including Psi0's own torchrun'd train.py: it installs the SDPA
# fallback the pinned trainer lacks (see src/humanoid_lab/psi0_compat.py).
export PYTHONPATH="$REPO_ROOT/scripts/psi0_shim:$REPO_ROOT/src:$PSI0_SRC/src${PYTHONPATH:+:$PYTHONPATH}"
ulimit -n 65535 || true

PYTHON_BIN="${PSI0_PYTHON:-python3}"
VERIFY="scripts/verify-psi0-unitree-dex3-sonic-v1.py"

# --- resolved Psi0 argv (single source of truth for the flags) ---------------
args=(
  finetune_sonic_psi0_config
  "--seed=$SEED"
  "--exp=$EXP"
  "--train.name=finetune"
  "--train.data_parallel=ddp"
  "--train.mixed_precision=bf16"
  "--train.output_dir=$OUTPUT_DIR"
  "--train.train_batch_size=$BATCH"
  "--train.val_batch_size=$BATCH"
  "--train.gradient_accumulation_steps=$ACCUM"
  "--train.learning_rate=$LR"
  "--train.max_training_steps=$MAX_STEPS"
  "--train.warmup_ratio=None"
  "--train.warmup_steps=1000"
  "--train.checkpointing_steps=$CKPT_STEPS"
  "--train.max_checkpoints_to_keep=5"
  "--train.validation_steps=$VAL_STEPS"
  "--train.val_num_batches=$VAL_BATCHES"
  "--train.max_grad_norm=1.0"
  "--train.lr_scheduler_type=cosine"
  "--train.lr_scheduler_kwargs.weight_decay=1e-6"
  "--train.lr_scheduler_kwargs.betas" 0.95 0.999
  "--log.report_to=wandb"
  "--data.root_dir=$DATASET_ROOT"
  "--data.train_repo_ids=$TRAIN_ID"
  "--data.val_repo_ids=$VAL_ID"
  "--data.transform.repack.image-keys" observation.images.egocentric
  "--data.transform.repack.instruction-key=$INSTRUCTION_KEY"
  "--data.transform.repack.action-keys" action.body_token_v1_1 action
  "--data.transform.repack.action-mask-key=$MASK_KEY"
  "--data.transform.repack.action-chunk-size=30"
  "--data.transform.repack.pad-action-dim=80"
  "--data.transform.repack.pad-state-dim=45"
  "--data.transform.repack.state-temporal-jitter=$STATE_JITTER"
  "--data.transform.repack.state-temporal-jitter-prob=$STATE_JITTER_PROB"
  "--data.transform.field.stat-path=$STATS"
  "--data.transform.field.stat-action-keys" action.body_token_v1_1 action
  "--data.transform.field.stat-state-keys" observation.state
  "--data.transform.field.action_norm_type=bounds"
  "--data.transform.field.no-use-norm-mask"
  "--data.transform.field.normalize-state"
  "--data.transform.field.pad-action-dim=80"
  "--data.transform.field.pad-state-dim=45"
  "--data.transform.field.state-noise-std=$STATE_NOISE_STD"
  "--data.transform.model.resize.size" 240 320
  "--data.transform.model.center_crop.size" 240 320
  "--data.transform.model.view-aug-min-scale=0.85"
  "--data.transform.model.view-aug-prob=1.0"
  "--model.model_name_or_path=$INIT_DIR"
  "--model.pretrained-action-header-path=$INIT_DIR"
  "--model.noise-scheduler=flow"
  "--model.train-diffusion-steps=1000"
  "--model.n_conditions=0"
  "--model.action-chunk-size=30"
  "--model.action-dim=80"
  "--model.action-exec-horizon=30"
  "--model.observation-horizon=1"
  "--model.odim=45"
  "--model.dropout=0.0"
  "--model.state-feature-dropout=0.0"
  "--model.view_feature_dim=2048"
  "--model.gradient-checkpointing"
  "--model.no-use_film"
  "--model.qk-norm=rms_norm"
  "--model.combined-temb"
  "--model.num-blocks=12"
  "--model.vlm-layer-indices" 3 5 8 10 12 14 17 19 21 23 26 28
  "--model.state-drop-prob=$STATE_DROP_PROB"
  "--model.state-as-action-token"
  "--model.state-null-token"
  "--model.pooled-text-encoder=clip"
  "--model.pooled-text-encoder-path=openai/clip-vit-large-patch14"
  "--model.pooled-projection-dim=768"
  "--model.pooled-cache-path=clip_pooled_cache.pt"
  "--model.no-rtc"
  "--model.max-delay=8"
)
# No --model.tune-vlm on purpose: the VLM stays frozen and only the action expert
# trains (plan §13.4).  Adding it would need 8x the VRAM this card does not have.
if [ "$AUG" = "1" ]; then
  args+=( "--data.transform.model.img-aug" "--data.transform.model.view-aug" )
fi
if [ -n "$RESUME" ]; then
  args+=( "--train.resume_from_checkpoint=$RESUME" )
fi

ARGS_FILE="$(mktemp -t psi0-unitree-dex3-sonic-v1.XXXXXX.args)"
trap 'rm -f "$ARGS_FILE"' EXIT
printf '%s\n' "${args[@]}" > "$ARGS_FILE"

if [ "$MODE" = "print" ]; then
  printf '%s\n' "${args[@]}"
  exit 0
fi

# --- preflight ---------------------------------------------------------------
[ -d "$PSI0_SRC" ] || fail "Psi0 checkout is missing: $PSI0_SRC (git submodule update --init third_party/Psi0)"
EXPECTED_COMMIT="${PSI0_EXPECTED_COMMIT:-4f3720d45e102b36d7c3e9465ab8062274170518}"
ACTUAL_COMMIT="$(git -C "$PSI0_SRC" rev-parse HEAD 2>/dev/null)" || fail "cannot read the Psi0 revision at $PSI0_SRC"
[ "$ACTUAL_COMMIT" = "$EXPECTED_COMMIT" ] || fail "Psi0 is at $ACTUAL_COMMIT, expected the pinned $EXPECTED_COMMIT"
[ -z "$(git -C "$PSI0_SRC" status --porcelain --untracked-files=no)" ] || \
  fail "Psi0 has tracked modifications; this wrapper drives it through flags only (see $PSI0_SRC)"

# The pack is produced by the dataset owner; its contract is checked by the
# verifier below, so the preflight here only proves the pieces exist.
[ -d "$DATASET_ROOT/$TRAIN_ID" ] || fail "train split is missing: $DATASET_ROOT/$TRAIN_ID (pack not produced yet, or DATASET_ROOT is wrong)"
[ -d "$DATASET_ROOT/$VAL_ID" ] || fail "val split is missing: $DATASET_ROOT/$VAL_ID (pack not produced yet, or DATASET_ROOT is wrong)"
[ -s "$STATS" ] || fail "normalisation stats are missing: $STATS (the pack must ship meta/stats_psi0.json)"
for file in config.json model.safetensors action_header.safetensors; do
  [ -s "$INIT_DIR/$file" ] || fail "warm-start checkpoint is missing $INIT_DIR/$file (./dev.sh fetch-psi0-ckpt)"
done
# The trainer writes its CLIP pooled cache into output_dir while it constructs
# the model, so the run directory must be writable before the weights load.
mkdir -p "$OUTPUT_DIR" 2>/dev/null || fail "cannot create the run directory: $OUTPUT_DIR"
[ -w "$OUTPUT_DIR" ] || fail "the run directory is not writable: $OUTPUT_DIR"

echo "Psi0 fine-tune -> $OUTPUT_DIR"
echo "  pack      : $DATASET_ROOT ($TRAIN_ID/$VAL_ID)"
echo "  warm start: $INIT_DIR"
echo "  steps     : $MAX_STEPS (batch $BATCH x accum $ACCUM), lr $LR, aug=$AUG"
echo "  wandb     : $WANDB_MODE / $WANDB_PROJECT"
[ -z "$RESUME" ] && echo "  resume    : fresh run" || echo "  resume    : $RESUME"

"$PYTHON_BIN" "$VERIFY" --args-file "$ARGS_FILE" --check contract ckpt config || exit $?
if [ "$MODE" = "check" ]; then
  echo "PASS: preflight only (--check-only)"
  exit 0
fi

# --- launch Psi0's own entry point -------------------------------------------
MAIN_PORT="$("$PYTHON_BIN" - <<'PY'
import socket

for port in range(29500, 30500):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            continue
    print(port)
    break
else:
    raise SystemExit("no free port in 29500..30500")
PY
)" || exit 1

# train.py loads third_party/Psi0/.env from the current directory and resolves
# its relative paths (deepspeed config, run dirs) from there, so run from there.
cd "$PSI0_SRC"
echo "Running: torchrun --nproc_per_node=1 --master_port=$MAIN_PORT scripts/train.py ${args[*]}"
torchrun --nproc_per_node=1 --master_port="$MAIN_PORT" scripts/train.py "${args[@]}"
