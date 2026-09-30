#!/usr/bin/env bash
#
# GR00T N1.7 fine-tuning of the Unitree Dex3 SONIC v1 pack on one 32 GB RTX 5090.
#
# The pack is a stock-embodiment LeRobot v2.1 dataset: meta/modality.json maps
# the official UNITREE_G1_SONIC modality config (gr00t/configs/data/embodiment_configs.py
# at the pinned commit) onto the exporter's storage layout and one ego camera.
# Storage stays SONIC-native -- observation.state holds the measured body in
# exporter order (hands beside their arms), observation.projected_gravity is its
# own [3] column, and the action is action.motion_token [64] plus
# teleop.left/right_hand_joints [7] each -- while the model consumes the
# registered key order, i.e. 46D state and 78D action over a 40-step horizon.
# Nothing here reimplements the dataset or the recipe: the launcher, the modality
# config and the statistics writer are the pinned GR00T source in
# /opt/src/isaac-groot, driven only through flags.
#
# Flag syntax comes from the pinned launcher itself (tyro over FinetuneConfig):
# booleans are --flag/--no-flag pairs (never "--flag True") and
# --color-jitter-params takes alternating key/value pairs.  The dataset/model/
# source side of the contract lives in scripts/groot-unitree-dex3-sonic-preflight.py.
#
# Modes (exactly one):
#   --check-only              fail-closed preflight: source/model revisions,
#                             dataset contract, stats, GPU, output directory
#   --print-command           print the resolved launcher argv and exit
#   --stats                   run the official gr00t/data/stats.py for the splits
#                             and validate the resulting statistics
#   --validate-artifacts DIR  validate a finished run directory and exit
#   (default)                 run the stage below on the pinned launcher
#
# Modifiers:
#   --dry-run   preflight + provenance, then print the command instead of launching
#   --smoke     two-optimizer-step stage (MAX_STEPS=2, ACCUM=1) + artifact validation
#   --run       full stage (default): MAX_STEPS=40000, BATCH=1 x ACCUM=32
#
# Single-GPU envelope: one RTX 5090, batch 1 x accumulation 32 = effective 32.
#
# The 32 GB default freezes the pretrained state/action projection modules and
# trains the diffusion head (plus the upstream-default VLLN path, which
# FinetuneConfig does not expose):
#
#   TUNE_PROJECTOR=0  -> --no-tune-projector (state encoder, action encoder,
#                        action decoder and position embedding stay frozen)
#   TUNE_DIFFUSION=1  -> --tune-diffusion-model (the 1.091B DiT trains)
#
# That shape comes from a measured failure, not a preference: with
# --tune-projector --tune-diffusion-model the pretrained smoke reached the first
# optimizer.step with 1,620,515,968 trainable parameters and OOMed at batch 1
# (30.47 GiB in use, 88.88 MiB free on a 32607 MiB card).  Batch is already 1
# and cannot go lower, so the shortfall is static (parameters + AdamW moments
# allocated during the step), not activation memory: only removing trainable
# parameters shrinks it.  Machines with 40 GiB or more can set TUNE_PROJECTOR=1;
# below TUNE_PROJECTOR_MIN_VRAM_MIB it is refused unless ALLOW_FULL_TUNE=1 says
# so explicitly, and the override is recorded in the run provenance.
#
# The optimizer is the second measured shortfall.  The 1,293,159,424 parameters
# this profile trains are loaded fp32 (the pinned launcher calls
# AutoModel.from_pretrained without a dtype, so transformers keeps
# torch.get_default_dtype()), i.e. 5,173 MiB of weights + 5,173 MiB of gradients
# + 9,866 MiB of AdamW moments (exp_avg + exp_avg_sq).  The pretrained smoke hit
# that wall inside the first optimizer.step while a 6,443 MiB neighbour held the
# rest of the card.  Neither the batch nor the trainable set can absorb it, and
# the pinned FinetuneConfig exposes no optimizer flag (the launcher hard-codes
# adamw_torch), so OPTIM=adafactor is the default here: Adafactor keeps a
# factored row/col second moment and no first moment, i.e. 7 MiB of state for
# this profile, while training exactly the same parameters.  OPTIM=adamw_torch
# restores the pinned optimizer (for a card with room for it, e.g. TUNE_PROJECTOR=1
# on 40 GiB+).  The override runs through scripts/groot-finetune-launcher.py,
# which executes the pinned launcher verbatim and is verified by the artifact
# check against the run's own experiment_cfg/conf.yaml.
#
# Measured with OPTIM=adafactor: the two-step pretrained smoke completed
# (checkpoint-2, train_loss 1.1624) with a 19,740 MiB peak on the 32607 MiB
# card, and its checkpoint-2/optimizer.pt holds 7.1 MiB of Adafactor state --
# 1,866,496 elements, exactly the factored row/col plus 1D second moments of
# the 1,293,159,424 trainable parameters.
#
# A batch above 1 is refused unless ALLOW_BATCH_OVERRIDE=1 (also recorded).
#
# Paths (inside the dev container):
#   DATASET_ROOT  $HUMANOID_DATA_ROOT/datasets/groot/unitree-dex3-sonic-v1
#   MODEL_DIR     $HUMANOID_DATA_ROOT/models/groot_n17_base
#   SOURCE_DIR    /opt/src/isaac-groot
#   OUTPUT_ROOT   /outputs/groot-unitree-dex3-sonic-v1/finetune
#
# Knobs: BATCH ACCUM MAX_STEPS LR SAVE_STEPS SAVE_TOTAL_LIMIT WORKERS SHARD_SIZE
#        NUM_SHARDS_PER_EPOCH EPISODE_SAMPLING_RATE WEIGHT_DECAY WARMUP_RATIO
#        RESUME OUTPUT_DIR OUTPUT_ROOT STATS_ROOT EXP STAMP WANDB WANDB_PROJECT
#        WANDB_MODE SEED STATE_DROPOUT COLOR_JITTER SKIP_WEIGHT_LOADING
#        DATASET_ROOT TRAIN_ID VAL_ID MODEL_DIR SOURCE_DIR EGO_KEY SPLIT
#        GROOT_MODEL_DIR GROOT_SOURCE_DIR GROOT_PYTHON GROOT_JSON_PYTHON
#        GPU_SAMPLE_SECONDS MIN_VRAM_MIB EXPECTED_GPU OMP_NUM_THREADS
#        CUDA_VISIBLE_DEVICES ALLOW_BATCH_OVERRIDE ALLOW_LOW_VRAM ALLOW_OTHER_GPU
#        GROOT_SKIP_GPU GROOT_SKIP_ENV_IMPORT
# Optimizer: OPTIM (adafactor = 32 GB default, adamw_torch/adamw_torch_fused =
#        pinned-upstream behaviour once the card can hold the moments)
# Tuning profile: TUNE_PROJECTOR (0 = freeze the pretrained state/action
#        projection modules, 1 = 40 GiB+ profile), TUNE_DIFFUSION,
#        TUNE_PROJECTOR_MIN_VRAM_MIB, ALLOW_FULL_TUNE
# GROOT_-prefixed aliases are accepted for the model/source/output roots so the
# harness composes with scripts/groot-finetune-smoke.sh conventions.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

readonly PREFLIGHT="scripts/groot-unitree-dex3-sonic-preflight.py"
readonly EXPECTED_EMBODIMENT="UNITREE_G1_SONIC"
readonly EXPECTED_HORIZON=40

MODE=""
STAGE="full"
DRY_RUN=0
ARTIFACT_DIR=""
EXPECT_STEPS=""
ALLOW_BATCH_OVERRIDE="${ALLOW_BATCH_OVERRIDE:-0}"
ALLOW_LOW_VRAM="${ALLOW_LOW_VRAM:-0}"
ALLOW_OTHER_GPU="${ALLOW_OTHER_GPU:-0}"
SKIP_GPU_CHECKS="${GROOT_SKIP_GPU:-0}"

print_help() {
  cat <<'HELP'
usage: groot-unitree-dex3-sonic.sh [MODE] [modifiers]

Modes:
  --check-only               fail-closed preflight only (no writes, no launch)
  --print-command            print the resolved launcher argv (alias --print-args)
  --stats                    run the official gr00t/data/stats.py and validate it
  --validate-artifacts DIR   validate a finished run directory (--expect-steps N)
  (default)                  launch the fine-tune

Modifiers:
  --run                      full stage (default)
  --smoke                    exactly two optimizer steps + artifact validation
  --dry-run                  preflight + provenance + command, no launch
  --expect-steps N           expected optimizer steps for artifact validation
  --allow-batch-override     allow BATCH > 1 (recorded in provenance)
  --allow-low-vram           continue below MIN_VRAM_MIB (recorded)
  --allow-other-gpu          continue when nvidia-smi reports another GPU model
  --allow-full-tune          accept TUNE_PROJECTOR=1 below its VRAM floor, or a
                             profile that would train only the VLLN path
  --skip-gpu-checks          skip nvidia-smi/torch checks (tests, CI)
  -h, --help                 show this help

Tuning profile (32 GB default freezes the pretrained projections):
  TUNE_PROJECTOR=0           --no-tune-projector: state encoder, action encoder,
                             action decoder and position embedding stay frozen
  TUNE_PROJECTOR=1           --tune-projector: 40 GiB+ profile; refused below
                             TUNE_PROJECTOR_MIN_VRAM_MIB unless --allow-full-tune
  TUNE_DIFFUSION=1           --tune-diffusion-model: the 1.091B DiT trains
  TUNE_DIFFUSION=0           --no-tune-diffusion-model
  tune_vlln is not exposed by the pinned FinetuneConfig and stays at its
  upstream default (true).

Optimizer (the pinned launcher hard-codes adamw_torch; FinetuneConfig exposes
no flag, so scripts/groot-finetune-launcher.py overrides training.optim):
  OPTIM=adafactor            default: factored second moment, no first moment
                             (~7 MiB of state instead of 9,866 MiB AdamW moments
                             for the 1,293,159,424 fp32 trainable parameters)
  OPTIM=adamw_torch          pinned-upstream optimizer; use on a card that can
                             hold the two fp32 moments per trainable parameter
  OPTIM=adamw_torch_fused    same memory, fused torch kernel

Examples:
  ./dev.sh groot-dex3-check
  SPLIT=train ./dev.sh groot-dex3-stats
  ./dev.sh groot-dex3-smoke
  DATASET_ROOT=/data/datasets/groot/unitree-dex3-sonic-v1-mini ./dev.sh groot-dex3-smoke
  OPTIM=adamw_torch ALLOW_FULL_TUNE=1 ./dev.sh groot-dex3-run --dry-run
  RESUME=1 ./dev.sh groot-dex3-run OUTPUT_DIR=/outputs/groot-unitree-dex3-sonic-v1/finetune/unitree-dex3-sonic-v1/run-20260919T000000Z
  ./dev.sh groot-dex3-run --print-command
HELP
}

fail() { echo "FAIL: $*" >&2; exit 2; }

set_mode() { # $1 = mode
  if [ -n "$MODE" ] && [ "$MODE" != "$1" ]; then
    fail "only one mode may be selected (already $MODE, then $1)"
  fi
  MODE="$1"
}

while (($#)); do
  case "$1" in
    --check-only) set_mode check ;;
    --print-command|--print-args) set_mode print ;;
    --stats) set_mode stats ;;
    --validate-artifacts)
      (($# >= 2)) || fail "--validate-artifacts needs a run directory"
      ARTIFACT_DIR="$2"
      set_mode validate
      shift
      ;;
    --smoke) STAGE=smoke ;;
    --run) STAGE=full ;;
    --dry-run) DRY_RUN=1 ;;
    --expect-steps)
      (($# >= 2)) || fail "--expect-steps needs a number"
      EXPECT_STEPS="$2"
      shift
      ;;
    --allow-batch-override) ALLOW_BATCH_OVERRIDE=1 ;;
    --allow-low-vram) ALLOW_LOW_VRAM=1 ;;
    --allow-other-gpu) ALLOW_OTHER_GPU=1 ;;
    --allow-full-tune) ALLOW_FULL_TUNE=1 ;;
    --skip-gpu-checks) SKIP_GPU_CHECKS=1 ;;
    --help|-h) print_help; exit 0 ;;
    *) fail "unknown argument: $1 (see --help)" ;;
  esac
  shift
done
[ -n "$MODE" ] || MODE=run
if [ -n "$EXPECT_STEPS" ]; then
  case "$EXPECT_STEPS" in *[!0-9]*) fail "--expect-steps must be a number: $EXPECT_STEPS" ;; esac
fi

# --- stage defaults ----------------------------------------------------------
if [ "$STAGE" = "smoke" ]; then
  DEFAULT_MAX_STEPS=2; DEFAULT_ACCUM=1; DEFAULT_SAVE_STEPS=2; DEFAULT_SAVE_TOTAL_LIMIT=1
  DEFAULT_WORKERS=0; DEFAULT_SHARD_SIZE=64; DEFAULT_NUM_SHARDS_PER_EPOCH=1
else
  DEFAULT_MAX_STEPS=40000; DEFAULT_ACCUM=32; DEFAULT_SAVE_STEPS=2000; DEFAULT_SAVE_TOTAL_LIMIT=5
  DEFAULT_WORKERS=8; DEFAULT_SHARD_SIZE=1024; DEFAULT_NUM_SHARDS_PER_EPOCH=100000
fi

MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-$DEFAULT_ACCUM}"
LR="${LR:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-$DEFAULT_SAVE_STEPS}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-$DEFAULT_SAVE_TOTAL_LIMIT}"
WORKERS="${WORKERS:-$DEFAULT_WORKERS}"
SHARD_SIZE="${SHARD_SIZE:-$DEFAULT_SHARD_SIZE}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-$DEFAULT_NUM_SHARDS_PER_EPOCH}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
STATE_DROPOUT="${STATE_DROPOUT:-}"
COLOR_JITTER="${COLOR_JITTER:-brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08}"
RESUME="${RESUME:-0}"
WANDB="${WANDB:-0}"
SEED="${SEED:-42}"
SKIP_WEIGHT_LOADING="${SKIP_WEIGHT_LOADING:-0}"
# The pinned launcher hard-codes training.optim="adamw_torch" and FinetuneConfig
# has no flag for it, so the optimizer is applied by
# scripts/groot-finetune-launcher.py (see the header).  Adafactor is the 32 GB
# default because AdamW needs 9,866 MiB of fp32 moments for this profile and
# OOMed inside the first optimizer.step; both train the same parameters.  Keep
# this list in sync with ALLOWED_OPTIMIZERS in the launcher shim.
OPTIM="${OPTIM:-adafactor}"
# 32 GB low-VRAM profile: keep the pretrained projection modules frozen and the
# diffusion head trainable.  tune_vlln stays at the upstream default (true); the
# pinned FinetuneConfig has no flag for it, so this harness does not pretend to
# control it.
TUNE_PROJECTOR="${TUNE_PROJECTOR:-0}"
TUNE_DIFFUSION="${TUNE_DIFFUSION:-1}"
TUNE_PROJECTOR_MIN_VRAM_MIB="${TUNE_PROJECTOR_MIN_VRAM_MIB:-40000}"
ALLOW_FULL_TUNE="${ALLOW_FULL_TUNE:-0}"

for knob in MAX_STEPS BATCH ACCUM SAVE_STEPS SAVE_TOTAL_LIMIT WORKERS SHARD_SIZE \
  NUM_SHARDS_PER_EPOCH; do
  value="${!knob}"
  case "$value" in ''|*[!0-9]*) fail "$knob must be a non-negative integer, got $value" ;; esac
done
for knob in TUNE_PROJECTOR TUNE_DIFFUSION; do
  value="${!knob}"
  case "$value" in 0|1) ;; *) fail "$knob must be 0 or 1, got $value" ;; esac
done
case "$OPTIM" in
  adafactor|adamw_torch|adamw_torch_fused) ;;
  *) fail "OPTIM must be adafactor (the 32 GB default), adamw_torch or adamw_torch_fused; got '$OPTIM' (see scripts/groot-finetune-launcher.py for why no 8-bit/paged/ZeRO optimizer is selectable in groot-n17)" ;;
esac
if [ "$TUNE_PROJECTOR" = 0 ] && [ "$TUNE_DIFFUSION" = 0 ] && [ "$ALLOW_FULL_TUNE" != 1 ]; then
  fail "TUNE_PROJECTOR=0 with TUNE_DIFFUSION=0 would leave only the upstream-default VLLN path trainable; set TUNE_DIFFUSION=1 (the 32 GB default) or ALLOW_FULL_TUNE=1 to run anyway"
fi

# The smoke stage exists to prove one forward/backward/optimizer cycle, so it
# pins the step count instead of inheriting the full-run knobs.
if [ "$STAGE" = "smoke" ]; then
  [ "$MAX_STEPS" = 2 ] || \
    fail "smoke stage is exactly 2 optimizer steps; MAX_STEPS=$MAX_STEPS (use --run for a longer run)"
  [ "$ACCUM" = 1 ] || \
    fail "smoke stage is exactly 2 optimizer steps; ACCUM=$ACCUM would change the optimizer-step count"
fi
if [ "$BATCH" -gt 1 ] && [ "$ALLOW_BATCH_OVERRIDE" != 1 ]; then
  fail "BATCH=$BATCH would exceed the 32 GB envelope (default BATCH=1 x ACCUM=32); set ALLOW_BATCH_OVERRIDE=1 to override and record it"
fi

# --- paths -------------------------------------------------------------------
DATAROOT="${HUMANOID_DATA_ROOT:-/data}"
DATASET_ROOT="${DATASET_ROOT:-$DATAROOT/datasets/groot/unitree-dex3-sonic-v1}"
TRAIN_ID="${TRAIN_ID:-train}"
VAL_ID="${VAL_ID:-val}"
MODEL_DIR="${GROOT_MODEL_DIR:-$DATAROOT/models/groot_n17_base}"
SOURCE_DIR="${GROOT_SOURCE_DIR:-/opt/src/isaac-groot}"
SOURCE_COMMIT="${GROOT_EXPECTED_SOURCE_COMMIT:-1a1837f20538b7d7e21f977a11a5aee14f99803c}"
MODEL_REPO="${GROOT_EXPECTED_MODEL_REPO:-nvidia/GR00T-N1.7-3B}"
MODEL_REVISION="${GROOT_EXPECTED_MODEL_REVISION:-2fc962b973bccdd5d8ce4f67cc63b264d6886495}"
PYTHON_BIN="${GROOT_PYTHON:-/opt/venvs/groot-n17/bin/python}"
JSON_PYTHON="${GROOT_JSON_PYTHON:-python3}"
# The pinned launcher runs through the local shim, which applies OPTIM and
# otherwise executes the pinned source verbatim.
LAUNCHER="$SOURCE_DIR/gr00t/experiment/launch_finetune.py"
LAUNCH_WRAPPER="$REPO_ROOT/scripts/groot-finetune-launcher.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GROOT_OUTPUT_ROOT:-/outputs/groot-unitree-dex3-sonic-v1/finetune}}"
STATS_ROOT="${STATS_ROOT:-${GROOT_STATS_ROOT:-/outputs/groot-unitree-dex3-sonic-v1/stats}}"
EXP="${EXP:-unitree-dex3-sonic-v1}"
EMBODIMENT="${EMBODIMENT:-$EXPECTED_EMBODIMENT}"
EGO_KEY="${EGO_KEY:-ego_view}"
SPLIT="${SPLIT:-both}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-30000}"
EXPECTED_GPU="${EXPECTED_GPU:-5090}"
GPU_SAMPLE_SECONDS="${GPU_SAMPLE_SECONDS:-1}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# The launcher shim reads both: GROOT_OPTIM is the override it applies to
# config.training.optim, GROOT_LAUNCHER is the pinned source it executes.
export GROOT_OPTIM="$OPTIM"
export GROOT_LAUNCHER="$LAUNCHER"
# GR00T only reaches for wandb when --use-wandb is passed; keep the process
# offline-safe either way so a stray import cannot block on a login prompt.
export WANDB_MODE="${WANDB_MODE:-offline}"

[ "$EMBODIMENT" = "$EXPECTED_EMBODIMENT" ] || \
  fail "this harness is wired for the stock $EXPECTED_EMBODIMENT embodiment; got EMBODIMENT=$EMBODIMENT"
case "$SPLIT" in
  train) STATS_SPLITS=(train) ;;
  val) STATS_SPLITS=(val) ;;
  both) STATS_SPLITS=(train val) ;;
  *) fail "SPLIT must be train, val or both; got $SPLIT" ;;
esac
STATS_SPLITS_CSV="$(
  printf '%s,' "${STATS_SPLITS[@]}"
)"
STATS_SPLITS_CSV="${STATS_SPLITS_CSV%,}"

TRAIN_SPLIT="$DATASET_ROOT/$TRAIN_ID"
VAL_SPLIT="$DATASET_ROOT/$VAL_ID"

if [ -n "${OUTPUT_DIR:-}" ]; then
  RUN_DIR="$OUTPUT_DIR"
else
  RUN_DIR="$OUTPUT_ROOT/$EXP/${STAGE}-${STAMP}"
fi

# --- helpers -----------------------------------------------------------------
split_root() { # $1 = split name
  case "$1" in
    train) printf '%s\n' "$TRAIN_SPLIT" ;;
    val) printf '%s\n' "$VAL_SPLIT" ;;
    *) fail "unknown split: $1" ;;
  esac
}

latest_checkpoint() {
  local dir="$1" candidate step best_step=-1 best=""
  for candidate in "$dir"/checkpoint-*; do
    [ -d "$candidate" ] || continue
    step="${candidate##*/checkpoint-}"
    case "$step" in ''|*[!0-9]*) continue ;; esac
    if [ "$step" -gt "$best_step" ]; then
      best_step="$step"
      best="$candidate"
    fi
  done
  [ -n "$best" ] || return 1
  printf '%s\n' "$best"
}

record_gpu() { # $1 = destination csv; best effort, never fatal
  [ "$SKIP_GPU_CHECKS" = 1 ] && return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader >"$1" 2>/dev/null || true
}

check_gpu() {
  if [ "$SKIP_GPU_CHECKS" = 1 ]; then
    echo "WARN: GPU checks skipped (GROOT_SKIP_GPU=1)" >&2
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable; run this inside the GPU container"
  nvidia-smi -L >/dev/null 2>&1 || fail "nvidia-smi cannot see a GPU; run this inside the GPU container"
  local name memory
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/[[:space:]]*$//')"
  memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  [ -n "$name" ] || fail "nvidia-smi reported no GPU name"
  [[ "$memory" =~ ^[0-9]+$ ]] || fail "invalid GPU VRAM reported by nvidia-smi: $memory"
  echo "GPU: $name (${memory} MiB total), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  if [ -n "$EXPECTED_GPU" ] && ! printf '%s' "$name" | grep -Fq "$EXPECTED_GPU"; then
    [ "$ALLOW_OTHER_GPU" = 1 ] || fail "GPU is '$name' but EXPECTED_GPU='$EXPECTED_GPU'; set ALLOW_OTHER_GPU=1 to accept it"
    echo "WARN: GPU '$name' does not match EXPECTED_GPU='$EXPECTED_GPU' (allowed explicitly)" >&2
  fi
  if [ "$memory" -lt "$MIN_VRAM_MIB" ]; then
    [ "$ALLOW_LOW_VRAM" = 1 ] || fail "GPU has ${memory} MiB VRAM; this harness needs ${MIN_VRAM_MIB} MiB (set ALLOW_LOW_VRAM=1 for a best-effort attempt)"
    echo "WARN: ${memory} MiB VRAM is below MIN_VRAM_MIB=${MIN_VRAM_MIB} (allowed explicitly)" >&2
  fi
  if [ "$TUNE_PROJECTOR" = 1 ] && [ "$memory" -lt "$TUNE_PROJECTOR_MIN_VRAM_MIB" ]; then
    [ "$ALLOW_FULL_TUNE" = 1 ] || fail "TUNE_PROJECTOR=1 is the ${TUNE_PROJECTOR_MIN_VRAM_MIB} MiB+ profile (this GPU has ${memory} MiB): training the pretrained state/action projection modules OOMed at the first optimizer step on this class of card. Keep the 32 GB default TUNE_PROJECTOR=0, or set ALLOW_FULL_TUNE=1 to attempt it anyway (recorded in the run provenance)"
    echo "WARN: TUNE_PROJECTOR=1 on ${memory} MiB is below TUNE_PROJECTOR_MIN_VRAM_MIB=${TUNE_PROJECTOR_MIN_VRAM_MIB} (allowed explicitly with ALLOW_FULL_TUNE=1)" >&2
  fi
  if [ "${GROOT_SKIP_ENV_IMPORT:-0}" != "1" ]; then
    "$PYTHON_BIN" - <<'PY' || fail "the groot-n17 interpreter cannot see CUDA (torch.cuda.is_available() is False)"
import torch

assert torch.cuda.is_available(), "CUDA is unavailable to groot-n17"
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
PY
  fi
}

start_gpu_sampler() {
  [ "$SKIP_GPU_CHECKS" = 1 ] && return 0
  GPU_MEMORY_CSV="$RUN_DIR/gpu_memory.csv"
  printf 'timestamp,memory_used_mib,utilization_gpu_pct\n' >"$GPU_MEMORY_CSV"
  (
    while :; do
      nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits \
        >>"$GPU_MEMORY_CSV" 2>/dev/null || true
      sleep "$GPU_SAMPLE_SECONDS"
    done
  ) &
  GPU_SAMPLER_PID=$!
}

stop_gpu_sampler() {
  if [ -n "${GPU_SAMPLER_PID:-}" ]; then
    kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    wait "$GPU_SAMPLER_PID" 2>/dev/null || true
    GPU_SAMPLER_PID=""
  fi
}

write_peak_vram() {
  [ -n "${GPU_MEMORY_CSV:-}" ] && [ -s "$GPU_MEMORY_CSV" ] || return 0
  awk -F, 'NR > 1 { gsub(/ /, "", $2); if ($2 + 0 > peak) peak = $2 + 0 } END { print peak + 0 }' \
    "$GPU_MEMORY_CSV" >"$RUN_DIR/peak_vram_mib" 2>/dev/null || true
}

# --- output directory safety -------------------------------------------------
verify_run_dir_safety() {
  case "$RUN_DIR" in
    ''|/)
      fail "refusing to use '$RUN_DIR' as an output directory" ;;
    "$DATASET_ROOT"|"$DATASET_ROOT"/*|"$MODEL_DIR"|"$MODEL_DIR"/*|"$SOURCE_DIR"|"$SOURCE_DIR"/*|"$REPO_ROOT")
      fail "refusing to write the run into an input tree: $RUN_DIR" ;;
  esac
  if [ -e "$RUN_DIR" ]; then
    [ -d "$RUN_DIR" ] || fail "output path exists and is not a directory: $RUN_DIR"
    if [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
      if [ "$RESUME" = 1 ] || [ "$RESUME" = latest ]; then
        latest_checkpoint "$RUN_DIR" >/dev/null || \
          fail "RESUME=1 was requested but $RUN_DIR holds no checkpoint-* directory"
      else
        fail "output directory is already populated: $RUN_DIR (pass RESUME=1 to continue from its latest checkpoint, or pick a fresh OUTPUT_DIR/STAMP)"
      fi
    elif [ "$RESUME" = 1 ] || [ "$RESUME" = latest ]; then
      fail "RESUME=1 was requested but $RUN_DIR is empty"
    fi
  elif [ "$RESUME" = 1 ] || [ "$RESUME" = latest ]; then
    fail "RESUME=1 was requested but the run directory does not exist: $RUN_DIR"
  fi
}

prepare_run_dir() {
  verify_run_dir_safety
  [ -d "$RUN_DIR" ] || mkdir -p "$RUN_DIR" || fail "cannot create the run directory: $RUN_DIR"
  [ -w "$RUN_DIR" ] || fail "the run directory is not writable: $RUN_DIR"
}

# --- resolved launcher argv (single source of truth for the flags) -----------
build_train_args() {
  # The shim is the argv[0] for every training mode, so a missing shim is a
  # broken command rather than a runtime surprise.
  [ -f "$LAUNCH_WRAPPER" ] || fail "the launcher shim is missing: $LAUNCH_WRAPPER"
  local -a jitter=()
  local index=0
  # tyro parses dict[str, float] as alternating key/value pairs, so a stray
  # token would silently bind the next flag as a dict key.
  read -r -a jitter <<<"$COLOR_JITTER"
  if [ "$(( ${#jitter[@]} % 2 ))" -ne 0 ]; then
    fail "COLOR_JITTER needs key/value pairs, got: $COLOR_JITTER"
  fi
  while [ "$index" -lt "${#jitter[@]}" ]; do
    case "${jitter[$index]}" in
      brightness|contrast|saturation|hue) ;;
      *) fail "COLOR_JITTER key '${jitter[$index]}' is not a torchvision ColorJitter key" ;;
    esac
    index=$((index + 2))
  done

  local -a resume_args=()
  if [ "$RESUME" = 1 ] || [ "$RESUME" = latest ]; then
    # The pinned launcher takes a boolean and resumes from the latest
    # checkpoint-* in output_dir, so this can never merge two runs.
    resume_args=( --resume-from-checkpoint )
  fi
  local -a wandb_args=( --no-use-wandb )
  if [ "$WANDB" = 1 ]; then
    wandb_args=( --use-wandb --wandb-project "${WANDB_PROJECT:-groot-unitree-dex3-sonic-v1}" )
  fi
  local -a dropout_args=()
  if [ -n "$STATE_DROPOUT" ]; then
    dropout_args=( --state-dropout-prob "$STATE_DROPOUT" )
  fi
  local -a skip_args=()
  if [ "$SKIP_WEIGHT_LOADING" = 1 ]; then
    skip_args=( --skip-weight-loading )
  fi
  local -a projector_args=( --no-tune-projector )
  if [ "$TUNE_PROJECTOR" = 1 ]; then
    projector_args=( --tune-projector )
  fi
  local -a diffusion_args=( --no-tune-diffusion-model )
  if [ "$TUNE_DIFFUSION" = 1 ]; then
    diffusion_args=( --tune-diffusion-model )
  fi

  TRAIN_ARGS=(
    "$LAUNCH_WRAPPER"
    --base-model-path "$MODEL_DIR"
    --dataset-path "$TRAIN_SPLIT"
    --embodiment-tag "$EMBODIMENT"
    --num-gpus 1
    --output-dir "$RUN_DIR"
    --global-batch-size "$BATCH"
    --gradient-accumulation-steps "$ACCUM"
    --learning-rate "$LR"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-ratio "$WARMUP_RATIO"
    --dataloader-num-workers "$WORKERS"
    --shard-size "$SHARD_SIZE"
    --num-shards-per-epoch "$NUM_SHARDS_PER_EPOCH"
    --episode-sampling-rate "$EPISODE_SAMPLING_RATE"
    --use-percentiles
    --no-tune-llm
    --no-tune-visual
    "${projector_args[@]}"
    "${diffusion_args[@]}"
    --color-jitter-params "${jitter[@]}"
    "${dropout_args[@]}"
    "${resume_args[@]}"
    "${skip_args[@]}"
    "${wandb_args[@]}"
  )
}

build_stats_args() { # $1 = split root
  STATS_ARGS=(
    "$SOURCE_DIR/gr00t/data/stats.py"
    --dataset-path "$1"
    --embodiment-tag "$EMBODIMENT"
  )
}

print_args() { printf '%s\n' "$@"; }

# --- preflight ---------------------------------------------------------------
run_preflight() { # remaining args: extra --check flags
  [ -x "$PYTHON_BIN" ] || fail "groot-n17 Python is missing: $PYTHON_BIN"
  local -a checks=("$@")
  ((${#checks[@]})) || checks=(--check source --check model --check dataset --check stats)
  "$JSON_PYTHON" "$PREFLIGHT" "${checks[@]}" \
    --source-dir "$SOURCE_DIR" \
    --expected-source-commit "$SOURCE_COMMIT" \
    --model-dir "$MODEL_DIR" \
    --expected-model-repo "$MODEL_REPO" \
    --expected-model-revision "$MODEL_REVISION" \
    --dataset-root "$DATASET_ROOT" \
    --train-id "$TRAIN_ID" \
    --val-id "$VAL_ID" \
    --ego-key "$EGO_KEY" \
    --expected-fps 50 || exit $?
  echo "Embodiment   : $EMBODIMENT (horizon $EXPECTED_HORIZON)"
  echo "Dataset      : $DATASET_ROOT ($TRAIN_ID/$VAL_ID)"
  echo "Base model   : $MODEL_DIR ($MODEL_REPO@${MODEL_REVISION:0:12})"
  echo "Source       : $SOURCE_DIR (${SOURCE_COMMIT:0:12})"
  echo "Run directory: $RUN_DIR"
  echo "Steps        : max $MAX_STEPS, batch $BATCH x accum $ACCUM (effective $((BATCH * ACCUM))), lr $LR"
  echo "Trainable    : $(
    [ "$TUNE_PROJECTOR" = 1 ] && printf 'state/action projections + diffusion head' || printf 'diffusion head only (pretrained projections frozen)'
  ) (LLM + visual tower frozen; VLLN at the upstream default)"
  echo "Optimizer    : $OPTIM (scripts/groot-finetune-launcher.py override; adamw_torch would need 9,866 MiB of moments for this profile)"
  echo "Checkpoints  : every $SAVE_STEPS, keep $SAVE_TOTAL_LIMIT"
  echo "Dataloader   : $WORKERS worker(s), shard $SHARD_SIZE, $NUM_SHARDS_PER_EPOCH shard(s)/epoch"
  echo "wandb        : $([ "$WANDB" = 1 ] && echo "on (${WANDB_PROJECT:-groot-unitree-dex3-sonic-v1}, $WANDB_MODE)" || echo "off")"
  echo "resume       : $([ "$RESUME" = 1 ] || [ "$RESUME" = latest ] && echo "latest checkpoint in $RUN_DIR" || echo "fresh run")"
}

write_provenance() {
  # config.json is the pre-launch record.  A successful run then has the pinned
  # launcher save its HF model into the same directory, which overwrites this
  # file with the model config; env.json and experiment_cfg/conf.yaml are the
  # records that survive (and the artifact check reads the effective optimizer
  # from conf.yaml, not from here).
  printf '%s\n' "$SOURCE_COMMIT" >"$RUN_DIR/source_revision"
  printf '%q ' "$PYTHON_BIN" "${TRAIN_ARGS[@]}" >"$RUN_DIR/command.sh"
  printf '\n' >>"$RUN_DIR/command.sh"
  record_gpu "$RUN_DIR/gpu_before.csv"

  "$JSON_PYTHON" - "$RUN_DIR" "$PREFLIGHT" "$DATASET_ROOT" "$TRAIN_ID" "$VAL_ID" "$MODEL_DIR" \
    "$SOURCE_DIR" "$SOURCE_COMMIT" "$MODEL_REPO" "$MODEL_REVISION" <<'PY'
import datetime
import json
import pathlib
import socket
import subprocess
import sys

(run_dir, preflight, dataset_root, train_id, val_id, model_dir, source_dir,
 source_commit, model_repo, model_revision) = sys.argv[1:11]
run_dir = pathlib.Path(run_dir)


def run_json(argv):
    proc = subprocess.run(
        [sys.executable, preflight, *argv, "--json"], capture_output=True, text=True, check=False
    )
    text = proc.stdout.strip()
    if not text:
        return {"ok": False, "error": (proc.stderr.strip() or "no output")[:400]}
    try:
        return json.loads(text)
    except ValueError as exc:
        return {"ok": False, "error": f"unparsable preflight output: {exc}", "stdout": text[-400:]}


checks = {
    "dataset": ["--check", "hashes", "--dataset-root", dataset_root, "--train-id", train_id,
                "--val-id", val_id, "--splits", f"{train_id},{val_id}"],
    "model": ["--check", "model", "--model-dir", model_dir, "--expected-model-repo", model_repo,
              "--expected-model-revision", model_revision],
    "source": ["--check", "source", "--source-dir", source_dir,
               "--expected-source-commit", source_commit],
}
results = {name: run_json(argv) for name, argv in checks.items()}
dataset = results["dataset"]
model_type = None
try:
    model_type = json.loads((pathlib.Path(model_dir) / "config.json").read_text()).get("model_type")
except (OSError, ValueError):
    pass
provenance = {}
try:
    provenance = json.loads((pathlib.Path(model_dir) / "MODEL_PROVENANCE.json").read_text())
except (OSError, ValueError):
    pass
split_meta = {}
for split in (train_id, val_id):
    try:
        info = json.loads((pathlib.Path(dataset_root) / split / "meta" / "info.json").read_text())
        split_meta[split] = {
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "robot_type": info.get("robot_type"),
        }
    except (OSError, ValueError):
        split_meta[split] = None

payload = {
    "kind": "groot-unitree-dex3-sonic-provenance",
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hostname": socket.gethostname(),
    "cwd": str(pathlib.Path.cwd()),
    "source": {"dir": source_dir, "commit": source_commit},
    "model": {
        "repo": provenance.get("repo"),
        "revision": provenance.get("revision"),
        "variant": provenance.get("variant"),
        "fetched_at_utc": provenance.get("fetched_at_utc"),
        "model_type": model_type,
    },
    "dataset": {"root": dataset_root, "splits": split_meta, "hashes": dataset.get("records", {})},
    "preflight": {
        name: {
            key: result.get(key)
            for key in ("ok", "error", "errors", "stdout")
        }
        for name, result in results.items()
    },
}
(run_dir / "dataset_provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

env_values = {
    key: __import__("os").environ.get(key)
    for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "WANDB_MODE", "HF_HOME", "HUMANOID_DATA_ROOT",
                "GROOT_OPTIM", "GROOT_LAUNCHER")
}
(run_dir / "env.json").write_text(json.dumps(env_values, indent=2, sort_keys=True) + "\n")
PY

  "$JSON_PYTHON" - "$RUN_DIR" "$MODE" "$STAGE" "$MAX_STEPS" "$BATCH" "$ACCUM" "$LR" \
    "$SAVE_STEPS" "$SAVE_TOTAL_LIMIT" "$WORKERS" "$SHARD_SIZE" "$NUM_SHARDS_PER_EPOCH" \
    "$EPISODE_SAMPLING_RATE" "$RESUME" "$WANDB" "$SEED" "$ALLOW_BATCH_OVERRIDE" \
    "$ALLOW_LOW_VRAM" "$ALLOW_OTHER_GPU" "$CUDA_VISIBLE_DEVICES" "$EGO_KEY" "$EMBODIMENT" \
    "$COLOR_JITTER" "$STATE_DROPOUT" "$TUNE_PROJECTOR" "$TUNE_DIFFUSION" \
    "$ALLOW_FULL_TUNE" "$OPTIM" <<'PY'
import json
import pathlib
import sys

(run_dir, mode, stage, max_steps, batch, accum, lr, save_steps, save_total_limit,
 workers, shard_size, num_shards_per_epoch, episode_sampling_rate, resume, wandb, seed,
 allow_batch_override, allow_low_vram, allow_other_gpu, cuda_visible_devices, ego_key,
 embodiment, color_jitter, state_dropout, tune_projector, tune_diffusion,
 allow_full_tune, optimizer) = sys.argv[1:29]
payload = {
    "mode": mode,
    "stage": stage,
    "optimizer": optimizer,
    "launcher": "scripts/groot-finetune-launcher.py",
    "max_steps": int(max_steps),
    "batch": int(batch),
    "gradient_accumulation_steps": int(accum),
    "effective_batch": int(batch) * int(accum),
    "learning_rate": float(lr),
    "save_steps": int(save_steps),
    "save_total_limit": int(save_total_limit),
    "dataloader_num_workers": int(workers),
    "shard_size": int(shard_size),
    "num_shards_per_epoch": int(num_shards_per_epoch),
    "episode_sampling_rate": float(episode_sampling_rate),
    "resume": resume in ("1", "latest"),
    "wandb": wandb == "1",
    "seed": int(seed),
    "embodiment_tag": embodiment,
    "ego_camera_key": ego_key,
    "color_jitter": color_jitter,
    "state_dropout_prob": float(state_dropout) if state_dropout else None,
    "cuda_visible_devices": cuda_visible_devices,
    "tuning": {
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": tune_projector == "1",
        "tune_diffusion_model": tune_diffusion == "1",
        # FinetuneConfig has no flag for this; the pinned upstream default is true.
        "tune_vlln": "upstream default (true, not exposed)",
        "profile": "40gb+" if tune_projector == "1" else "32gb-low-vram",
    },
    "overrides": {
        "batch": allow_batch_override == "1",
        "low_vram": allow_low_vram == "1",
        "other_gpu": allow_other_gpu == "1",
        "full_tune": allow_full_tune == "1",
        # The pinned launcher hard-codes adamw_torch; anything else is the shim.
        "optimizer": optimizer != "adamw_torch",
    },
}
pathlib.Path(run_dir, "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

# --- logging / signals -------------------------------------------------------
LOG_FILE=""
GPU_MEMORY_CSV=""
GPU_SAMPLER_PID=""
LAUNCH_PID=""

start_logging() {
  LOG_FILE="$1"
  exec > >(tee -a "$LOG_FILE") 2>&1
}

finish() {
  local rc=$?
  { stop_gpu_sampler; } || true
  # Only a run that actually started logging owns a run directory; check/print/
  # stats/validate must never drop files into an unrelated tree.
  if [ -n "${LOG_FILE:-}" ] && [ -d "$RUN_DIR" ]; then
    { write_peak_vram; } || true
    printf '%s\n' "$rc" >"$RUN_DIR/exit_code" || true
    if [ -s "$RUN_DIR/peak_vram_mib" ]; then
      echo "Peak GPU memory (sampled every ${GPU_SAMPLE_SECONDS}s): $(<"$RUN_DIR/peak_vram_mib") MiB"
    fi
  fi
  return "$rc"
}

on_signal() { # $1 = signal name, $2 = exit code
  echo "received SIG$1: stopping the fine-tune" >&2
  if [ -n "$LAUNCH_PID" ]; then
    kill "-$1" "$LAUNCH_PID" 2>/dev/null || true
  fi
  exit "$2"
}

trap finish EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT

# --- modes -------------------------------------------------------------------
if [ "$MODE" = "validate" ]; then
  [ -n "$ARTIFACT_DIR" ] || fail "--validate-artifacts needs a run directory"
  validate_extra=(--expect-optim "$OPTIM")
  [ -n "$EXPECT_STEPS" ] && validate_extra+=(--expect-steps "$EXPECT_STEPS")
  "$JSON_PYTHON" "$PREFLIGHT" --check artifacts --run-dir "$ARTIFACT_DIR" "${validate_extra[@]}" || exit $?
  echo "PASS: valid fine-tune artifacts: $ARTIFACT_DIR"
  exit 0
fi

# The argv is built before the preflight so --print-command stays cheap: the
# tests and the operator both use it as the single source of flag truth.  The
# output-directory guard is read-only, so printing an unusable resume command
# is refused here rather than at launch time.
build_train_args
verify_run_dir_safety

if [ "$MODE" = "print" ]; then
  print_args "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
  exit 0
fi

if [ "$MODE" = "stats" ]; then
  # stats.py needs the pack metadata but not the statistics it is about to write,
  # so the preflight runs the contract checks only.  It is a CPU job: no GPU gate.
  run_preflight --check source --check model --check dataset
  if [ "$DRY_RUN" = 1 ]; then
    for split in "${STATS_SPLITS[@]}"; do
      build_stats_args "$(split_root "$split")"
      print_args "$PYTHON_BIN" "${STATS_ARGS[@]}"
    done
    exit 0
  fi
  mkdir -p "$STATS_ROOT/$STAMP" || fail "cannot create the stats provenance directory: $STATS_ROOT/$STAMP"
  for split in "${STATS_SPLITS[@]}"; do
    root="$(split_root "$split")"
    [ -d "$root/meta" ] || fail "$split split meta is missing: $root/meta (the pack must exist before stats can be generated)"
    build_stats_args "$root"
    printf '%q ' "$PYTHON_BIN" "${STATS_ARGS[@]}" >"$STATS_ROOT/$STAMP/stats-$split.command.sh"
    printf '\n' >>"$STATS_ROOT/$STAMP/stats-$split.command.sh"
    echo "Running: $PYTHON_BIN ${STATS_ARGS[*]}"
    "$PYTHON_BIN" "${STATS_ARGS[@]}" 2>&1 | tee "$STATS_ROOT/$STAMP/stats-$split.log" || \
      fail "gr00t/data/stats.py failed for the $split split (see $STATS_ROOT/$STAMP/stats-$split.log)"
  done
  "$JSON_PYTHON" "$PREFLIGHT" --check stats --dataset-root "$DATASET_ROOT" \
    --train-id "$TRAIN_ID" --val-id "$VAL_ID" --splits "$STATS_SPLITS_CSV" || exit $?
  echo "PASS: official statistics present and validated for: $STATS_SPLITS_CSV"
  echo "stats provenance: $STATS_ROOT/$STAMP"
  exit 0
fi

run_preflight
check_gpu
if [ "$MODE" = "check" ]; then
  verify_run_dir_safety
  echo "PASS: preflight only (--check-only); nothing was launched"
  exit 0
fi

# train / smoke / dry-run
prepare_run_dir
start_logging "$RUN_DIR/run.log"
echo "run directory: $RUN_DIR"
echo "log file     : $RUN_DIR/run.log"
printf '%s\n' "$$" >"$RUN_DIR/wrapper.pid"
record_gpu "$RUN_DIR/gpu_before.csv"
write_provenance

if [ "$DRY_RUN" = 1 ]; then
  echo "dry run: not launching"
  print_args "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
  exit 0
fi

echo "Running: $PYTHON_BIN ${TRAIN_ARGS[*]}"
start_gpu_sampler
"$PYTHON_BIN" "${TRAIN_ARGS[@]}" &
LAUNCH_PID=$!
rc=0
wait "$LAUNCH_PID" || rc=$?
LAUNCH_PID=""
stop_gpu_sampler
if [ "$rc" -ne 0 ]; then
  # The wrapper's own exit_code (written by the EXIT trap) records how this
  # script ended; the launcher's code is kept separately for triage.
  printf '%s\n' "$rc" >"$RUN_DIR/launcher_exit_code"
  if grep -qi 'CUDA out of memory' "$RUN_DIR/run.log" 2>/dev/null; then
    # Sampling can miss the true peak when the step dies between samples, so the
    # allocator's own report is kept next to it.
    grep -i 'CUDA out of memory' "$RUN_DIR/run.log" | tail -n 1 >"$RUN_DIR/oom.txt"
    echo "WARN: the launcher hit CUDA OOM; torch's own memory report is in $RUN_DIR/oom.txt (the sampled peak may understate it)" >&2
  fi
  if [ -f "$RUN_DIR/run.log" ] && grep -qiE 'GatedRepoError|gated repository|401 Client Error|403 Client Error|Access to model .* is restricted' "$RUN_DIR/run.log"; then
    echo "BLOCKED: Cosmos backbone access is gated; accept access for nvidia/Cosmos-Reason2-2B, run ./dev.sh hf-login, then retry." >&2
    exit 3
  fi
  fail "GR00T fine-tune failed with exit code $rc; see $RUN_DIR/run.log"
fi

if [ "$STAGE" = "smoke" ]; then
  # The smoke proves what actually ran: the run's own experiment_cfg/conf.yaml
  # must record the OPTIM this harness asked the shim for.
  "$JSON_PYTHON" "$PREFLIGHT" --check artifacts --run-dir "$RUN_DIR" --expect-steps 2 \
    --expect-optim "$OPTIM" || exit $?
  echo "PASS: smoke run completed with exactly 2 optimizer steps: $RUN_DIR"
else
  echo "PASS: fine-tune completed: $RUN_DIR"
  echo "validate artifacts with: ./dev.sh groot-dex3-run --validate-artifacts $RUN_DIR"
fi
