#!/usr/bin/env bash
# Opt-in GR00T N1.7 two-step fine-tuning smoke workflow.
set -euo pipefail

readonly EXPECTED_SOURCE_COMMIT="${GROOT_EXPECTED_SOURCE_COMMIT:-1a1837f20538b7d7e21f977a11a5aee14f99803c}"
readonly EXPECTED_MODEL_REPO="${GROOT_EXPECTED_MODEL_REPO:-nvidia/GR00T-N1.7-3B}"
readonly EXPECTED_MODEL_REVISION="${GROOT_EXPECTED_MODEL_REVISION:-2fc962b973bccdd5d8ce4f67cc63b264d6886495}"
readonly MIN_PRETRAINED_VRAM_MIB="${GROOT_MIN_PRETRAINED_VRAM_MIB:-40960}"
readonly SOURCE_DIR="${GROOT_SOURCE_DIR:-/opt/src/isaac-groot}"
readonly PYTHON_BIN="${GROOT_PYTHON:-/opt/venvs/groot-n17/bin/python}"
readonly JSON_PYTHON="${GROOT_JSON_PYTHON:-python3}"
readonly MODEL_DIR="${GROOT_MODEL_DIR:-${HUMANOID_DATA_ROOT:-/data}/models/groot_n17_base}"
readonly OUTPUT_ROOT="${GROOT_OUTPUT_ROOT:-/outputs/gr00t-n17-finetune-smoke}"
readonly DATASET_DIR="${GROOT_DATASET_DIR:-$SOURCE_DIR/demo_data/cube_to_bowl_5}"
readonly MODALITY_CONFIG="${GROOT_MODALITY_CONFIG:-$SOURCE_DIR/examples/SO100/so100_config.py}"

MODE="pipeline"
MODE_SET=0
OUTPUT_DIR=""
CHECKPOINT_DIR=""
ALLOW_LOW_VRAM=0
DRY_RUN=0
CHECK_TRAINING_ARTIFACTS=""
CHECK_EVAL_ARTIFACTS=""

usage() {
  cat <<'EOF'
usage: groot-finetune-smoke.sh [pipeline|pretrained|eval] [options]

Run an opt-in, single-GPU GR00T N1.7 fine-tuning smoke test.

modes:
  pipeline    Two optimizer steps without loading pretrained model weights (default).
  pretrained  Two optimizer steps with pretrained weights. Blocks below 40 GiB VRAM
              unless --allow-low-vram is passed.
  eval        Evaluate the newest pipeline checkpoint-2, or --checkpoint-dir, with
              the official open-loop evaluator.

options:
  --output-dir PATH       Create this otherwise non-existent output directory.
  --checkpoint-dir PATH   Pipeline checkpoint-2 to evaluate (eval only).
  --allow-low-vram        Explicitly attempt pretrained mode below 40 GiB VRAM.
  --dry-run               Validate prerequisites and print the command without running it.
  --check-training-artifacts PATH
                          Validate a completed checkpoint-2 directory and exit.
  --check-eval-artifacts PATH
                          Validate completed open-loop evaluation artifacts and exit.
  -h, --help              Show this help.

Environment overrides are provided for automated shell tests and isolated deployments:
GROOT_SOURCE_DIR, GROOT_PYTHON, GROOT_MODEL_DIR, GROOT_OUTPUT_ROOT,
GROOT_DATASET_DIR, GROOT_MODALITY_CONFIG, GROOT_EXPECTED_SOURCE_COMMIT,
GROOT_EXPECTED_MODEL_REPO, GROOT_EXPECTED_MODEL_REVISION.
EOF
}

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

blocked() {
  echo "BLOCKED: $*" >&2
  exit 3
}

require_file() {
  local path="$1"
  local description="$2"
  [[ -f "$path" ]] || fail "$description is missing: $path"
}

require_directory() {
  local path="$1"
  local description="$2"
  [[ -d "$path" ]] || fail "$description is missing: $path"
}

contains_lfs_pointer() {
  local path="$1"
  head -c 42 "$path" 2>/dev/null | cmp -s - <(printf '%s' 'version https://git-lfs.github.com/spec/v1')
}

require_non_pointer_assets() {
  local pattern="$1"
  local description="$2"
  local first
  local asset

  first="$(find "$DATASET_DIR" -type f -name "$pattern" -print -quit)"
  [[ -n "$first" ]] || fail "$description is missing below $DATASET_DIR"
  while IFS= read -r asset; do
    contains_lfs_pointer "$asset" && fail "$description is an unresolved Git LFS pointer: $asset"
  done < <(find "$DATASET_DIR" -type f -name "$pattern" -print)
  return 0
}

json_field_equals() {
  local path="$1"
  local field="$2"
  local expected="$3"
  "$JSON_PYTHON" - "$path" "$field" "$expected" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
actual = value.get(sys.argv[2])
if actual != sys.argv[3]:
    print(f"{sys.argv[1]}: expected {sys.argv[2]}={sys.argv[3]!r}, got {actual!r}", file=sys.stderr)
    raise SystemExit(1)
PY
}

validate_source() {
  require_directory "$SOURCE_DIR" "GR00T source checkout"
  require_file "$SOURCE_DIR/gr00t/experiment/launch_finetune.py" "GR00T fine-tuning launcher"
  local actual
  actual="$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null)" || fail "cannot read GR00T source revision at $SOURCE_DIR"
  [[ "$actual" == "$EXPECTED_SOURCE_COMMIT" ]] || fail "GR00T source revision is $actual; expected $EXPECTED_SOURCE_COMMIT"
}

validate_python_environment() {
  [[ -x "$PYTHON_BIN" ]] || fail "groot-n17 Python is missing: $PYTHON_BIN"
  "$PYTHON_BIN" --version >/dev/null
  if [[ "${GROOT_SMOKE_SKIP_ENV_IMPORT:-0}" != "1" ]]; then
    "$PYTHON_BIN" - <<'PY'
import gr00t
import torch
assert torch.cuda.is_available(), "CUDA is unavailable to groot-n17"
PY
  fi
}

validate_model() {
  require_directory "$MODEL_DIR" "GR00T N1.7 model directory"
  require_file "$MODEL_DIR/config.json" "GR00T N1.7 model config"
  require_file "$MODEL_DIR/MODEL_PROVENANCE.json" "GR00T N1.7 model provenance"
  json_field_equals "$MODEL_DIR/MODEL_PROVENANCE.json" repo "$EXPECTED_MODEL_REPO" || \
    fail "GR00T N1.7 model provenance repository does not match the lock"
  json_field_equals "$MODEL_DIR/MODEL_PROVENANCE.json" revision "$EXPECTED_MODEL_REVISION" || \
    fail "GR00T N1.7 model provenance revision does not match the lock"
  json_field_equals "$MODEL_DIR/config.json" model_type "Gr00tN1d7" || \
    fail "GR00T N1.7 model config is not a Gr00tN1d7 checkpoint"
}

validate_dataset() {
  require_directory "$DATASET_DIR" "GR00T demo dataset in the pinned source image"
  require_file "$DATASET_DIR/meta/modality.json" "GR00T dataset modality metadata"
  require_non_pointer_assets '*.parquet' 'GR00T demo parquet data'
  require_non_pointer_assets '*.mp4' 'GR00T demo video data'
  require_file "$MODALITY_CONFIG" "GR00T modality config"
}

vram_mib() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]'
}

validate_cuda_and_record_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable; run this inside the GPU container"
  nvidia-smi -L >/dev/null 2>&1 || fail "nvidia-smi cannot see a GPU; run this inside the GPU container"
  local memory
  memory="$(vram_mib)" || fail "cannot measure GPU VRAM with nvidia-smi"
  [[ "$memory" =~ ^[0-9]+$ ]] || fail "invalid GPU VRAM reported by nvidia-smi: $memory"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >"$OUTPUT_DIR/gpu.csv"
  printf '%s\n' "$memory" >"$OUTPUT_DIR/vram_mib"
}

prepare_output() {
  local timestamp
  if [[ -n "$OUTPUT_DIR" ]]; then
    [[ ! -e "$OUTPUT_DIR" ]] || fail "output directory must not already exist: $OUTPUT_DIR"
  else
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTPUT_DIR="$OUTPUT_ROOT/${MODE}-${timestamp}"
    [[ ! -e "$OUTPUT_DIR" ]] || fail "timestamped output directory already exists: $OUTPUT_DIR"
  fi
  mkdir -p "$OUTPUT_DIR"
  LOG_FILE="$OUTPUT_DIR/run.log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  printf 'mode=%s\n' "$MODE" >"$OUTPUT_DIR/mode.env"
}

write_metadata() {
  git -C "$SOURCE_DIR" rev-parse HEAD >"$OUTPUT_DIR/source_revision"
  "$JSON_PYTHON" - "$MODEL_DIR/MODEL_PROVENANCE.json" >"$OUTPUT_DIR/model_provenance.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
json.dump({key: value.get(key) for key in ("repo", "revision", "variant", "fetched_at_utc")}, sys.stdout, indent=2)
print()
PY
  cp "$MODEL_DIR/config.json" "$OUTPUT_DIR/base_model_config.json"
}

write_command() {
  local output="$1"
  shift
  printf '%q ' "$@" >"$output"
  printf '\n' >>"$output"
}

validate_training_artifacts() {
  local checkpoint="$1"
  require_directory "$checkpoint" "checkpoint-2"
  require_directory "$checkpoint/experiment_cfg" "checkpoint-2 experiment_cfg"
  if [[ -d "$checkpoint/processor" ]]; then
    require_file "$checkpoint/processor/processor_config.json" "checkpoint-2 processor config"
    require_file "$checkpoint/processor/statistics.json" "checkpoint-2 processor statistics"
  else
    # Isaac-GR00T N1.7 flattens copied processor files into checkpoint-2.
    require_file "$checkpoint/processor_config.json" "checkpoint-2 processor config"
    require_file "$checkpoint/statistics.json" "checkpoint-2 processor statistics"
  fi
  find "$checkpoint" -type f -name config.json -print -quit | grep -q . || \
    fail "checkpoint-2 has no model/config artifact: $checkpoint"
  find "$checkpoint" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name 'pytorch_model*.pt' \) -print -quit | grep -q . || \
    fail "checkpoint-2 has no model weight artifact: $checkpoint"
}

validate_eval_artifacts() {
  local output="$1"
  require_file "$output/open_loop_eval.log" "open-loop evaluation log"
  "$JSON_PYTHON" - "$output/open_loop_eval.log" <<'PY'
import math
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
for metric in ("MSE", "MAE"):
    match = re.search(rf"\b{metric}\b[^0-9+\-.]*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)", text, re.I)
    if not match:
        raise SystemExit(f"missing {metric} in {sys.argv[1]}")
    if not math.isfinite(float(match.group(1))):
        raise SystemExit(f"non-finite {metric} in {sys.argv[1]}")
PY
  find "$output" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) -print -quit | grep -q . || \
    fail "open-loop evaluation did not create a plot below $output"
}

latest_pipeline_checkpoint() {
  find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -type d -path '*/pipeline-*/checkpoint-2' -print 2>/dev/null | sort | tail -n 1
}

is_gated_error() {
  local log="$1"
  grep -qiE 'GatedRepoError|gated repository|401 Client Error|403 Client Error|Access to model .* is restricted' "$log"
}

run_training() {
  local -a command=(
    "$PYTHON_BIN" "$SOURCE_DIR/gr00t/experiment/launch_finetune.py"
    --base-model-path "$MODEL_DIR"
    --dataset-path "$DATASET_DIR"
    --embodiment-tag NEW_EMBODIMENT
    --modality-config-path "$MODALITY_CONFIG"
    --num-gpus 1
    --output-dir "$OUTPUT_DIR"
    --save-steps 2
    --save-total-limit 1
    --max-steps 2
    --global-batch-size 2
    --dataloader-num-workers 0
    --shard-size 64
    --num-shards-per-epoch 1
  )
  if [[ "$MODE" == "pipeline" ]]; then
    command+=(--skip-weight-loading)
  fi
  write_command "$OUTPUT_DIR/command.sh" "${command[@]}"
  printf 'command: '
  printf '%q ' "${command[@]}"
  printf '\n'
  [[ "$DRY_RUN" == 1 ]] && return 0

  if "${command[@]}"; then
    validate_training_artifacts "$OUTPUT_DIR/checkpoint-2"
  else
    local rc=$?
    if is_gated_error "$LOG_FILE"; then
      blocked "Cosmos backbone access is gated. Accept access for nvidia/Cosmos-Reason2-2B, run ./dev.sh hf-login, then retry."
    fi
    fail "GR00T $MODE fine-tuning command failed with exit code $rc; see $LOG_FILE"
  fi
}

run_eval() {
  local checkpoint="$1"
  local -a command=(
    "$PYTHON_BIN" "$SOURCE_DIR/gr00t/eval/open_loop_eval.py"
    --dataset-path "$DATASET_DIR"
    --embodiment-tag NEW_EMBODIMENT
    --model-path "$checkpoint"
    --traj-ids 0
    --execution-horizon 16
    --steps 5
    --modality-keys single_arm gripper
    --save-plot-path "$OUTPUT_DIR/open_loop_eval"
  )
  write_command "$OUTPUT_DIR/command.sh" "${command[@]}"
  printf 'command: '
  printf '%q ' "${command[@]}"
  printf '\n'
  [[ "$DRY_RUN" == 1 ]] && return 0

  if "${command[@]}" > >(tee "$OUTPUT_DIR/open_loop_eval.log") 2>&1; then
    validate_eval_artifacts "$OUTPUT_DIR"
  else
    local rc=$?
    if is_gated_error "$OUTPUT_DIR/open_loop_eval.log"; then
      blocked "Cosmos backbone access is gated. Accept access for nvidia/Cosmos-Reason2-2B, run ./dev.sh hf-login, then retry."
    fi
    fail "GR00T open-loop evaluation failed with exit code $rc; see $OUTPUT_DIR/open_loop_eval.log"
  fi
}

while (($#)); do
  case "$1" in
    pipeline|pretrained|eval)
      [[ "$MODE_SET" == 0 ]] || fail "only one mode may be supplied"
      MODE="$1"
      MODE_SET=1
      ;;
    --output-dir)
      (($# >= 2)) || fail "--output-dir needs a path"
      OUTPUT_DIR="$2"
      shift
      ;;
    --checkpoint-dir)
      (($# >= 2)) || fail "--checkpoint-dir needs a path"
      CHECKPOINT_DIR="$2"
      shift
      ;;
    --allow-low-vram) ALLOW_LOW_VRAM=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --check-training-artifacts)
      (($# >= 2)) || fail "--check-training-artifacts needs a checkpoint directory"
      CHECK_TRAINING_ARTIFACTS="$2"
      shift
      ;;
    --check-eval-artifacts)
      (($# >= 2)) || fail "--check-eval-artifacts needs an evaluation output directory"
      CHECK_EVAL_ARTIFACTS="$2"
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
  shift
done

if [[ -n "$CHECK_TRAINING_ARTIFACTS" || -n "$CHECK_EVAL_ARTIFACTS" ]]; then
  [[ -z "$CHECK_TRAINING_ARTIFACTS" || -z "$CHECK_EVAL_ARTIFACTS" ]] || \
    fail "artifact validation accepts one target at a time"
  if [[ -n "$CHECK_TRAINING_ARTIFACTS" ]]; then
    validate_training_artifacts "$CHECK_TRAINING_ARTIFACTS"
    echo "PASS: valid training artifacts: $CHECK_TRAINING_ARTIFACTS"
  else
    validate_eval_artifacts "$CHECK_EVAL_ARTIFACTS"
    echo "PASS: valid evaluation artifacts: $CHECK_EVAL_ARTIFACTS"
  fi
  exit 0
fi

[[ "$MODE" == "eval" || -z "$CHECKPOINT_DIR" ]] || fail "--checkpoint-dir is only valid for eval"
[[ "$MODE" == "pretrained" || "$ALLOW_LOW_VRAM" == 0 ]] || fail "--allow-low-vram is only valid for pretrained"

if [[ "$MODE" == "eval" ]]; then
  if [[ -z "$CHECKPOINT_DIR" ]]; then
    CHECKPOINT_DIR="$(latest_pipeline_checkpoint)"
  fi
  [[ -n "$CHECKPOINT_DIR" ]] || fail "no pipeline checkpoint-2 found below $OUTPUT_ROOT; run pipeline first or pass --checkpoint-dir"
  validate_training_artifacts "$CHECKPOINT_DIR"
fi

validate_source
validate_python_environment
validate_model
validate_dataset
prepare_output
write_metadata
validate_cuda_and_record_gpu

if [[ "$MODE" == "pretrained" ]]; then
  VRAM_MIB="$(<"$OUTPUT_DIR/vram_mib")"
  if (( VRAM_MIB < MIN_PRETRAINED_VRAM_MIB && ALLOW_LOW_VRAM == 0 )); then
    blocked "GPU has ${VRAM_MIB} MiB VRAM; pretrained fine-tuning requires ${MIN_PRETRAINED_VRAM_MIB} MiB (40 GiB) by default. To make an explicit best-effort attempt, re-run with --allow-low-vram."
  fi
fi

if [[ "$MODE" == "eval" ]]; then
  run_eval "$CHECKPOINT_DIR"
else
  run_training
fi

echo "PASS: GR00T N1.7 ${MODE} smoke completed: $OUTPUT_DIR"
