#!/usr/bin/env bash
# Offline tests for scripts/groot-unitree-dex3-sonic.sh.
#
# Everything the harness talks to is faked here: the pinned source checkout is a
# throwaway git repo, the dataset/model/GPU are fixtures, and the groot-n17
# interpreter is a recorder that writes the argv it was handed.  No training,
# no stats generation, no GPU needed -- the tests assert the resolved launcher
# argv and the fail-closed refusals.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/groot-unitree-dex3-sonic.sh"
PREFLIGHT="$ROOT/scripts/groot-unitree-dex3-sonic-preflight.py"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/groot-unitree-dex3-sonic.XXXXXX")"
cleanup() {
  [[ "${KEEP_TEST_TMP:-0}" == 1 ]] || rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

PASS=0
FAIL=0

pass() {
  printf 'PASS: %s\n' "$1"
  PASS=$((PASS + 1))
}

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  FAIL=$((FAIL + 1))
}

assert_file_contains() {
  local path="$1" needle="$2" message="$3"
  if grep -Fq -- "$needle" "$path"; then
    pass "$message"
  else
    fail "$message (missing: $needle)"
  fi
}

assert_file_lacks_line() {
  local path="$1" needle="$2" message="$3"
  if grep -Fxq -- "$needle" "$path"; then
    fail "$message (unexpected: $needle)"
  else
    pass "$message"
  fi
}

assert_argv_pairs() { # $1 = argv file, rest = flag value flag value ...
  local file="$1"
  shift
  local flag value
  while (($#)); do
    flag="$1"
    value="$2"
    if [[ "$(grep -A 1 -Fx -- "$flag" "$file" | tail -n 1)" == "$value" ]]; then
      pass "argv $(basename "$file"): $flag $value"
    else
      fail "argv $(basename "$file"): expected $flag $value"
      sed 's/^/  | /' "$file" >&2 || true
    fi
    shift 2
  done
}

expect_rc() {
  local expected="$1" log="$2"
  shift 2
  set +e
  "$@" >"$log" 2>&1
  local actual=$?
  set -e
  if [[ "$actual" == "$expected" ]]; then
    pass "exit status $expected: $*"
  else
    fail "expected exit $expected, got $actual: $*"
    sed 's/^/  | /' "$log" >&2 || true
  fi
}

# --- fixtures ----------------------------------------------------------------
SOURCE="$TMP_ROOT/source"
MODEL="$TMP_ROOT/model"
DATASET="$TMP_ROOT/dataset"
OUTPUT_ROOT="$TMP_ROOT/outputs"
STATS_ROOT="$TMP_ROOT/stats"
FAKE_BIN="$TMP_ROOT/bin"
mkdir -p "$SOURCE/gr00t/experiment" "$SOURCE/gr00t/data" "$MODEL" "$DATASET" \
  "$OUTPUT_ROOT" "$STATS_ROOT" "$FAKE_BIN"
printf '# placeholder launcher\n' >"$SOURCE/gr00t/experiment/launch_finetune.py"
printf '# placeholder stats writer\n' >"$SOURCE/gr00t/data/stats.py"
printf '{"model_type":"Gr00tN1d7"}\n' >"$MODEL/config.json"
printf '{}\n' >"$MODEL/processor_config.json"
printf '{}\n' >"$MODEL/statistics.json"
printf '{}\n' >"$MODEL/embodiment_id.json"
printf 'weights\n' >"$MODEL/model-00001-of-00002.safetensors"
printf '{"repo":"nvidia/GR00T-N1.7-3B","revision":"2fc962b973bccdd5d8ce4f67cc63b264d6886495","variant":""}\n' \
  >"$MODEL/MODEL_PROVENANCE.json"
git -C "$SOURCE" init -q
git -C "$SOURCE" config user.email groot-dex3-test@example.invalid
git -C "$SOURCE" config user.name groot-dex3-test
git -C "$SOURCE" add .
git -C "$SOURCE" commit -qm 'test source'
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"

write_split() { # $1 = split directory, $2 = state width, $3 = episode length
  local split="$1" simd="${2:-46}" episode_length="${3:-60}"
  mkdir -p "$split/meta" "$split/data/chunk-000" "$split/videos/chunk-000/observation.images.ego_view"
  python3 - "$split" "$simd" "$episode_length" <<'PY'
import json
import pathlib
import sys

split, simd, episode_length = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
# Official SONIC exporter storage: the hands sit next to their arms in
# observation.state, gravity is a separate column and the action is split into
# its three SONIC fields.  The modality groups are listed in the registered
# order the model consumes, each with the storage slice it reads.
storage = [("left_leg", 0, 6), ("right_leg", 6, 12), ("waist", 12, 15),
           ("left_arm", 15, 22), ("left_hand", 22, 29), ("right_arm", 29, 36),
           ("right_hand", 36, 43)]
registered = ["left_leg", "right_leg", "waist", "left_arm", "right_arm",
              "left_hand", "right_hand", "projected_gravity"]
slices = {key: (start, end) for key, start, end in storage}
gravity_dim = simd - 43
features = {
    "observation.state": {"dtype": "float32", "shape": [43]},
    "observation.projected_gravity": {"dtype": "float32", "shape": [gravity_dim]},
    "action.motion_token": {"dtype": "float32", "shape": [64]},
    "teleop.left_hand_joints": {"dtype": "float32", "shape": [7]},
    "teleop.right_hand_joints": {"dtype": "float32", "shape": [7]},
    "observation.images.ego_view": {"dtype": "video", "shape": [480, 640, 3]},
}
(split / "meta").mkdir(parents=True, exist_ok=True)
json.dump({
    "codebase_version": "v2.1", "robot_type": "unitree_g1", "total_episodes": 2,
    "total_frames": 2 * episode_length, "total_tasks": 1, "chunks_size": 1000, "fps": 50,
    "splits": {"train": "0:2"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "total_chunks": 1, "total_videos": 2, "features": features,
}, open(split / "meta/info.json", "w"), indent=2)
modality = {
    "state": {}, "action": {},
    "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
    "annotation": {"human.task_description": {"original_key": "annotation.human.task_description"}},
}
for key in registered:
    if key == "projected_gravity":
        modality["state"][key] = {"start": 0, "end": gravity_dim,
                                  "original_key": "observation.projected_gravity"}
    else:
        start, end = slices[key]
        modality["state"][key] = {"start": start, "end": end}
for key, column in (("motion_token", "action.motion_token"),
                    ("left_hand_joints", "teleop.left_hand_joints"),
                    ("right_hand_joints", "teleop.right_hand_joints")):
    modality["action"][key] = {"start": 0, "end": 64 if key == "motion_token" else 7,
                               "original_key": column}
json.dump(modality, open(split / "meta/modality.json", "w"), indent=2)
stat_columns = {"observation.state": 43, "observation.projected_gravity": gravity_dim,
                "action.motion_token": 64, "teleop.left_hand_joints": 7,
                "teleop.right_hand_joints": 7}
stats = {column: {field: [0.0] * dim for field in ("mean", "std", "min", "max", "q01", "q99")}
         for column, dim in stat_columns.items()}
stats["__fingerprints__"] = {column: "sha256:" + column for column in stat_columns}
json.dump(stats, open(split / "meta/stats.json", "w"))
json.dump({"__fingerprints__": {}}, open(split / "meta/relative_stats.json", "w"))
with open(split / "meta/episodes.jsonl", "w") as handle:
    for index in range(2):
        handle.write(json.dumps({"episode_index": index, "length": episode_length, "tasks": ["pick"]}) + "\n")
with open(split / "meta/tasks.jsonl", "w") as handle:
    handle.write(json.dumps({"task_index": 0, "task": "pick"}) + "\n")
for index in range(2):
    (split / f"data/chunk-000/episode_{index:06d}.parquet").write_bytes(b"PAR1payload")
    (split / f"videos/chunk-000/observation.images.ego_view/episode_{index:06d}.mp4").write_bytes(b"v")
PY
}

write_split "$DATASET/train" 46 60
write_split "$DATASET/val" 46 60

cat >"$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
gpu_name="${FAKE_GPU_NAME:-NVIDIA GeForce RTX 5090}"
gpu_vram="${FAKE_GPU_VRAM:-32768}"
case "${1:-}" in
  -L) echo "GPU 0: $gpu_name (UUID: GPU-test)";;
  --query-gpu=name) printf '%s\n' "$gpu_name";;
  --query-gpu=memory.total) printf '%s\n' "$gpu_vram";;
  --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu)
    printf '0, %s, 580.65.06, %s MiB, 12 MiB, 0 %%\n' "$gpu_name" "$gpu_vram";;
  --query-gpu=timestamp,memory.used,utilization.gpu)
    printf '2026-09-19T00:00:00Z, 2048 MiB, 55 %%\n';;
  *) echo "unexpected nvidia-smi arguments: $*" >&2; exit 2;;
esac
EOF
chmod 0755 "$FAKE_BIN/nvidia-smi"

# Stands in for /opt/venvs/groot-n17/bin/python: records the argv it was handed
# and materializes the artifacts the pinned launcher would have written.
FAKE_PYTHON="$FAKE_BIN/fake-groot-python"
cat >"$FAKE_PYTHON" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "-" ]]; then
  exec "$(command -v python3)" "\$@"
fi
target="\${1:-}"
shift || true
case "\$target" in
  *stats.py)
    printf '%s\n' "\$@" >"\$FAKE_STATS_ARGV"
    [[ "\${FAKE_STATS_FAIL:-0}" == 1 ]] && exit 5
    [[ "\${FAKE_STATS_CORRUPT:-0}" == 1 ]] && rm -f "\${FAKE_STATS_CORRUPT_TARGET:?}"
    exit 0
    ;;
  *launch_finetune.py|*groot-finetune-launcher.py)
    printf '%s\n' "\$@" >"\$FAKE_TRAIN_ARGV"
    [[ "\${FAKE_TRAIN_FAIL:-0}" == 1 ]] && exit "\${FAKE_TRAIN_RC:-7}"
    if [[ "\${FAKE_TRAIN_OOM:-0}" == 1 ]]; then
      printf 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 31.32 GiB of which 88.88 MiB is free. Including non-PyTorch memory, this process has 30.47 GiB memory in use.\n'
      exit 1
    fi
    sleep "\${FAKE_TRAIN_SLEEP:-0}"
    ;;
  *) echo "unexpected python target: \$target" >&2; exit 2;;
esac
run_dir=""
steps=2
while ((\$#)); do
  case "\$1" in
    --output-dir) run_dir="\$2"; shift 2;;
    --max-steps) steps="\$2"; shift 2;;
    *) shift;;
  esac
done
[[ -n "\$run_dir" ]] || { echo "fake launcher saw no --output-dir" >&2; exit 2; }
step="\${FAKE_TRAIN_STEP:-\$steps}"
mkdir -p "\$run_dir/experiment_cfg" "\$run_dir/processor" "\$run_dir/checkpoint-\$step/experiment_cfg" "\$run_dir/checkpoint-\$step/processor"
printf '{}\n' >"\$run_dir/experiment_cfg/config.yaml"
# The real launcher shim exports GROOT_OPTIM and the pinned experiment.run()
# records the effective value in conf.yaml; FAKE_OPTIM_EFFECTIVE stands in for
# a shim that failed to apply it.
effective_optim="\${FAKE_OPTIM_EFFECTIVE:-\${GROOT_OPTIM:-adamw_torch}}"
printf 'training:\n  optim: %s\n' "\$effective_optim" >"\$run_dir/experiment_cfg/conf.yaml"
printf '{}\n' >"\$run_dir/processor/processor_config.json"
printf '{}\n' >"\$run_dir/processor/statistics.json"
printf '{"global_step": %s}\n' "\$step" >"\$run_dir/checkpoint-\$step/trainer_state.json"
printf '{}\n' >"\$run_dir/checkpoint-\$step/processor_config.json"
printf '{}\n' >"\$run_dir/checkpoint-\$step/statistics.json"
printf 'weights\n' >"\$run_dir/checkpoint-\$step/model.safetensors"
EOF
chmod 0755 "$FAKE_PYTHON"

wrapper_env() { # run a command with the wrapper's test environment
  PATH="$FAKE_BIN:$PATH" \
  GROOT_SOURCE_DIR="$SOURCE" \
  GROOT_EXPECTED_SOURCE_COMMIT="$SOURCE_COMMIT" \
  GROOT_MODEL_DIR="$MODEL" \
  GROOT_EXPECTED_MODEL_REPO="nvidia/GR00T-N1.7-3B" \
  GROOT_EXPECTED_MODEL_REVISION="2fc962b973bccdd5d8ce4f67cc63b264d6886495" \
  GROOT_PYTHON="$FAKE_PYTHON" \
  GROOT_JSON_PYTHON="$(command -v python3)" \
  GROOT_OUTPUT_ROOT="$OUTPUT_ROOT" \
  GROOT_STATS_ROOT="$STATS_ROOT" \
  DATASET_ROOT="$DATASET" \
  EXP=test STAMP=20260919T000000Z \
  MIN_VRAM_MIB="${MIN_VRAM_MIB:-30000}" \
  EXPECTED_GPU="${EXPECTED_GPU:-5090}" \
  GPU_SAMPLE_SECONDS=1 \
  GROOT_SKIP_ENV_IMPORT=1 \
  FAKE_TRAIN_ARGV="${FAKE_TRAIN_ARGV:-$TMP_ROOT/train-argv}" \
  FAKE_STATS_ARGV="${FAKE_STATS_ARGV:-$TMP_ROOT/stats-argv}" \
  "$@"
}

run_wrapper() {
  # NAME=value arguments are treated as environment overrides for the wrapper;
  # everything else is passed through as a script argument.
  local -a script_args=() env_args=()
  local arg
  for arg in "$@"; do
    case "$arg" in
      [A-Za-z_]*=*) env_args+=("$arg") ;;
      *) script_args+=("$arg") ;;
    esac
  done
  wrapper_env env "${env_args[@]}" bash "$SCRIPT" "${script_args[@]}"
}

# --- print-command: full-stage argv ------------------------------------------
# --print-command prints exactly the argv it would exec, one argument per line,
# so the log itself is the argv under test.
PRINT_LOG="$TMP_ROOT/print.log"
expect_rc 0 "$PRINT_LOG" run_wrapper --print-command
assert_argv_pairs "$PRINT_LOG" \
  --base-model-path "$MODEL" \
  --dataset-path "$DATASET/train" \
  --embodiment-tag UNITREE_G1_SONIC \
  --num-gpus 1
assert_argv_pairs "$PRINT_LOG" \
  --global-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 1e-4 \
  --max-steps 40000 \
  --save-steps 2000 \
  --save-total-limit 5 \
  --dataloader-num-workers 8 \
  --shard-size 1024 \
  --num-shards-per-epoch 100000 \
  --episode-sampling-rate 0.1 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5
assert_file_contains "$PRINT_LOG" '--use-percentiles' 'full argv keeps percentile statistics'
assert_file_contains "$PRINT_LOG" '--no-tune-llm' 'full argv freezes the LLM'
assert_file_contains "$PRINT_LOG" '--no-tune-visual' 'full argv freezes the visual tower'
assert_file_contains "$PRINT_LOG" '--no-tune-projector' 'the 32 GB default freezes the pretrained projections'
assert_file_contains "$PRINT_LOG" '--tune-diffusion-model' 'the 32 GB default trains the diffusion head'
assert_file_lacks_line "$PRINT_LOG" '--tune-projector' 'the 32 GB default does not train the projector'
assert_file_lacks_line "$PRINT_LOG" '--no-tune-diffusion-model' 'the 32 GB default keeps the diffusion head trainable'
assert_file_lacks_line "$PRINT_LOG" '--tune-vlln' 'no unsupported tune_vlln flag is emitted'
assert_file_contains "$PRINT_LOG" '--no-use-wandb' 'full argv keeps wandb off by default'
assert_file_lacks_line "$PRINT_LOG" '--use-wandb' 'full argv does not enable wandb'
assert_file_lacks_line "$PRINT_LOG" '--resume-from-checkpoint' 'full argv does not resume by default'
assert_file_lacks_line "$PRINT_LOG" '--experiment-name' 'full argv lets output-dir name the experiment'
assert_file_contains "$PRINT_LOG" '--color-jitter-params' 'full argv carries the official SONIC color jitter'
assert_file_contains "$PRINT_LOG" 'scripts/groot-finetune-launcher.py' \
  'the pinned launcher runs through the optimizer shim'

JITTER_INDEX="$(grep -n -Fx -- '--color-jitter-params' "$PRINT_LOG" | cut -d: -f1)"
JITTER_TOKENS="$(sed -n "$((JITTER_INDEX + 1)),$((JITTER_INDEX + 8))p" "$PRINT_LOG" | tr '\n' ' ' | sed 's/ $//')"
if [[ "$JITTER_TOKENS" == "brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08" ]]; then
  pass 'color jitter is the official SONIC recipe in tyro key/value pair syntax'
else
  fail "unexpected color jitter tokens: $JITTER_TOKENS"
fi

# --- print-command: smoke stage ----------------------------------------------
SMOKE_PRINT_LOG="$TMP_ROOT/smoke-print.log"
expect_rc 0 "$SMOKE_PRINT_LOG" run_wrapper --smoke --print-command
assert_argv_pairs "$SMOKE_PRINT_LOG" \
  --max-steps 2 \
  --gradient-accumulation-steps 1 \
  --save-steps 2 \
  --save-total-limit 1 \
  --dataloader-num-workers 0

# --- stage/knob refusals -----------------------------------------------------
expect_rc 2 "$TMP_ROOT/smoke-steps.log" run_wrapper --smoke --print-command MAX_STEPS=40000
assert_file_contains "$TMP_ROOT/smoke-steps.log" 'smoke stage is exactly 2 optimizer steps' \
  'smoke refuses a longer run'
expect_rc 2 "$TMP_ROOT/smoke-accum.log" run_wrapper --smoke --print-command ACCUM=32
assert_file_contains "$TMP_ROOT/smoke-accum.log" 'smoke stage is exactly 2 optimizer steps' \
  'smoke refuses accumulation that changes the optimizer-step count'
expect_rc 2 "$TMP_ROOT/batch.log" run_wrapper --print-command BATCH=2
assert_file_contains "$TMP_ROOT/batch.log" 'ALLOW_BATCH_OVERRIDE=1' \
  'batch above 1 is refused with the explicit override hint'
expect_rc 0 "$TMP_ROOT/batch-override.log" run_wrapper --print-command --allow-batch-override BATCH=2
assert_argv_pairs "$TMP_ROOT/batch-override.log" --global-batch-size 2
expect_rc 2 "$TMP_ROOT/embodiment.log" run_wrapper --print-command EMBODIMENT=NEW_EMBODIMENT
assert_file_contains "$TMP_ROOT/embodiment.log" 'UNITREE_G1_SONIC' 'a non-stock embodiment is refused'
expect_rc 2 "$TMP_ROOT/conflict.log" run_wrapper --print-command --check-only
assert_file_contains "$TMP_ROOT/conflict.log" 'only one mode' 'two modes are refused'
expect_rc 2 "$TMP_ROOT/unknown.log" run_wrapper --frobnicate
assert_file_contains "$TMP_ROOT/unknown.log" 'unknown argument' 'unknown arguments are refused'
expect_rc 0 "$TMP_ROOT/help.log" run_wrapper --help

# --- check-only: happy path --------------------------------------------------
expect_rc 0 "$TMP_ROOT/check.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check.log" 'PASS: preflight only' 'check-only passes on a valid pack'
assert_file_contains "$TMP_ROOT/check.log" 'GPU: NVIDIA GeForce RTX 5090 (32768 MiB total)' 'check-only records GPU identity'
if [[ -e "$OUTPUT_ROOT/test/full-20260919T000000Z" ]]; then
  fail 'check-only must not create the run directory'
else
  pass 'check-only does not write the run directory'
fi

# --- check-only: fail-closed dataset refusals --------------------------------
BACKUP="$TMP_ROOT/backup"
mkdir -p "$BACKUP"
cp "$DATASET/train/meta/stats.json" "$BACKUP/stats.json"
rm -f "$DATASET/train/meta/stats.json"
expect_rc 2 "$TMP_ROOT/check-stats.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-stats.log" 'meta/stats.json is missing' 'missing stats block training'
cp "$BACKUP/stats.json" "$DATASET/train/meta/stats.json"

cp "$DATASET/train/meta/relative_stats.json" "$BACKUP/relative.json"
rm -f "$DATASET/train/meta/relative_stats.json"
expect_rc 2 "$TMP_ROOT/check-relative.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-relative.log" 'meta/relative_stats.json is missing' \
  'missing relative_stats block training'
cp "$BACKUP/relative.json" "$DATASET/train/meta/relative_stats.json"

printf '0-byte\n' >"$DATASET/train/meta/.stats.json.abc.tmp"
expect_rc 2 "$TMP_ROOT/check-partial.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-partial.log" 'stale partial markers' 'stale stats tmp files block training'
rm -f "$DATASET/train/meta/.stats.json.abc.tmp"

cp "$DATASET/train/meta/info.json" "$BACKUP/info.json"
python3 - "$DATASET/train/meta/info.json" <<'PY'
import json
import sys

path = sys.argv[1]
info = json.load(open(path))
info["fps"] = 30
json.dump(info, open(path, "w"))
PY
expect_rc 2 "$TMP_ROOT/check-fps.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-fps.log" 'fps is 30' 'a non-50 Hz pack is refused'
cp "$BACKUP/info.json" "$DATASET/train/meta/info.json"

cp "$DATASET/train/meta/modality.json" "$BACKUP/modality.json"
python3 - "$DATASET/train/meta/modality.json" <<'PY'
import json
import sys

path = sys.argv[1]
modality = json.load(open(path))
del modality["state"]["projected_gravity"]
json.dump(modality, open(path, "w"))
PY
expect_rc 2 "$TMP_ROOT/check-modality.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-modality.log" 'projected_gravity' 'a renamed state key is refused'
cp "$BACKUP/modality.json" "$DATASET/train/meta/modality.json"

cp "$DATASET/train/meta/episodes.jsonl" "$BACKUP/episodes.jsonl"
printf '{"episode_index": 0, "length": 12, "tasks": ["pick"]}\n' >"$DATASET/train/meta/episodes.jsonl"
expect_rc 2 "$TMP_ROOT/check-short.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-short.log" 'shorter than the 40-step action horizon' \
  'episodes shorter than the action horizon are refused'
cp "$BACKUP/episodes.jsonl" "$DATASET/train/meta/episodes.jsonl"

# --- check-only: official storage layout -------------------------------------
# The pack keeps the SONIC exporter's storage order (hands next to their arms)
# while the model consumes the registered order, so the slices are the contract.
cp "$DATASET/train/meta/modality.json" "$BACKUP/modality.json"
python3 - "$DATASET/train/meta/modality.json" <<'PY'
import json
import sys

path = sys.argv[1]
modality = json.load(open(path))
# Swap the hands and arms back to registered-order storage while keeping the
# declared slices: hand channels would be fed to an arm.
for key, start, end in (("left_arm", 22, 29), ("left_hand", 15, 22),
                        ("right_arm", 36, 43), ("right_hand", 29, 36)):
    modality["state"][key] = {"start": start, "end": end}
json.dump(modality, open(path, "w"))
PY
expect_rc 2 "$TMP_ROOT/check-storage-order.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-storage-order.log" 'official storage slice' \
  'registered-order storage with official slices is refused'
cp "$BACKUP/modality.json" "$DATASET/train/meta/modality.json"

cp "$DATASET/train/meta/modality.json" "$BACKUP/modality.json"
python3 - "$DATASET/train/meta/modality.json" <<'PY'
import json
import sys

path = sys.argv[1]
modality = json.load(open(path))
del modality["state"]["projected_gravity"]["original_key"]
json.dump(modality, open(path, "w"))
PY
expect_rc 2 "$TMP_ROOT/check-gravity-key.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-gravity-key.log" 'observation.projected_gravity' \
  'gravity without its own column is refused'
cp "$BACKUP/modality.json" "$DATASET/train/meta/modality.json"

# A pack whose gravity column is 2D cannot concatenate to the model-facing 46D.
mv "$DATASET/train" "$TMP_ROOT/train-good"
write_split "$DATASET/train" 45 60
expect_rc 2 "$TMP_ROOT/check-dim.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-dim.log" 'must be 46D' 'a 45D model-facing state is refused'
rm -rf "$DATASET/train"
mv "$TMP_ROOT/train-good" "$DATASET/train"

# --- check-only: the pack's own layout is accepted ---------------------------
expect_rc 0 "$TMP_ROOT/check-layout.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-layout.log" 'official SONIC storage layout; registered-order concatenation is 46D' \
  'the official storage layout is accepted'
assert_file_contains "$TMP_ROOT/check-layout.log" 'action.motion_token' \
  'the SONIC action fields are validated as separate columns'

# --- check-only: source and model revision gates -----------------------------
printf '# tampered\n' >>"$SOURCE/gr00t/experiment/launch_finetune.py"
expect_rc 2 "$TMP_ROOT/check-dirty-source.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-dirty-source.log" 'tracked modifications' 'a patched source tree is refused'
git -C "$SOURCE" checkout -- gr00t/experiment/launch_finetune.py

expect_rc 2 "$TMP_ROOT/check-source.log" run_wrapper --check-only GROOT_EXPECTED_SOURCE_COMMIT=0000000000000000000000000000000000000000
assert_file_contains "$TMP_ROOT/check-source.log" 'expected the pinned' 'a source revision mismatch is refused'

cp "$MODEL/MODEL_PROVENANCE.json" "$BACKUP/model-provenance.json"
printf '{"repo":"nvidia/GR00T-N1.7-3B","revision":"wrong"}\n' >"$MODEL/MODEL_PROVENANCE.json"
expect_rc 2 "$TMP_ROOT/check-model.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-model.log" 'provenance revision' 'a model revision mismatch is refused'
cp "$BACKUP/model-provenance.json" "$MODEL/MODEL_PROVENANCE.json"

cp "$MODEL/config.json" "$BACKUP/model-config.json"
printf '{"model_type":"Gr00tN1d6"}\n' >"$MODEL/config.json"
expect_rc 2 "$TMP_ROOT/check-model-type.log" run_wrapper --check-only
assert_file_contains "$TMP_ROOT/check-model-type.log" 'Gr00tN1d7' 'a non-N1.7 base model is refused'
cp "$BACKUP/model-config.json" "$MODEL/config.json"

# --- check-only: GPU gates ---------------------------------------------------
expect_rc 2 "$TMP_ROOT/check-vram.log" run_wrapper --check-only FAKE_GPU_VRAM=24576
assert_file_contains "$TMP_ROOT/check-vram.log" 'ALLOW_LOW_VRAM=1' 'a small GPU is refused with the override hint'
expect_rc 0 "$TMP_ROOT/check-vram-ok.log" run_wrapper --check-only --allow-low-vram FAKE_GPU_VRAM=24576
expect_rc 2 "$TMP_ROOT/check-gpu.log" run_wrapper --check-only FAKE_GPU_NAME='NVIDIA RTX A6000'
assert_file_contains "$TMP_ROOT/check-gpu.log" 'ALLOW_OTHER_GPU=1' 'an unexpected GPU model is refused with the override hint'
expect_rc 0 "$TMP_ROOT/check-gpu-ok.log" run_wrapper --check-only --allow-other-gpu FAKE_GPU_NAME='NVIDIA RTX A6000'

# --- output directory safety -------------------------------------------------
POPULATED="$TMP_ROOT/populated"
mkdir -p "$POPULATED/checkpoint-2"
printf '{}\n' >"$POPULATED/checkpoint-2/trainer_state.json"
expect_rc 2 "$TMP_ROOT/check-populated.log" run_wrapper --check-only OUTPUT_DIR="$POPULATED"
assert_file_contains "$TMP_ROOT/check-populated.log" 'already populated' 'a populated output directory is refused'
expect_rc 2 "$TMP_ROOT/resume-nockpt.log" run_wrapper --print-command RESUME=1 OUTPUT_DIR="$TMP_ROOT/empty-target"
assert_file_contains "$TMP_ROOT/resume-nockpt.log" 'does not exist' 'RESUME=1 without a run directory is refused'
expect_rc 2 "$TMP_ROOT/inputs.log" run_wrapper --check-only OUTPUT_DIR="$DATASET"
assert_file_contains "$TMP_ROOT/inputs.log" 'refusing to write the run into an input tree' \
  'writing into the dataset tree is refused'

# --- dry-run: provenance + resume --------------------------------------------
DRY_OUTPUT="$TMP_ROOT/dry-run"
expect_rc 0 "$TMP_ROOT/dry.log" run_wrapper --dry-run OUTPUT_DIR="$DRY_OUTPUT"
assert_file_contains "$TMP_ROOT/dry.log" 'dry run: not launching' 'dry-run does not launch the launcher'
for artifact in command.sh config.json dataset_provenance.json env.json source_revision gpu_before.csv run.log; do
  if [[ -s "$DRY_OUTPUT/$artifact" ]]; then
    pass "dry-run writes provenance: $artifact"
  else
    fail "dry-run did not write provenance: $artifact"
  fi
done
assert_file_contains "$DRY_OUTPUT/config.json" '"effective_batch": 32' 'provenance records the effective batch'
assert_file_contains "$DRY_OUTPUT/config.json" '"overrides"' 'provenance records the override block'
assert_file_contains "$DRY_OUTPUT/config.json" '"optimizer": "adafactor"' \
  'provenance records the default optimizer'
assert_file_contains "$DRY_OUTPUT/config.json" '"launcher": "scripts/groot-finetune-launcher.py"' \
  'provenance records the launcher shim'
assert_file_contains "$DRY_OUTPUT/env.json" '"GROOT_OPTIM": "adafactor"' \
  'the launcher shim sees GROOT_OPTIM'
assert_file_contains "$DRY_OUTPUT/env.json" 'gr00t/experiment/launch_finetune.py' \
  'the launcher shim is pointed at the pinned launcher'
assert_file_contains "$DRY_OUTPUT/dataset_provenance.json" "$SOURCE_COMMIT" 'provenance records the source commit'
assert_file_contains "$DRY_OUTPUT/dataset_provenance.json" 'sha256' 'provenance records dataset metadata hashes'
assert_file_contains "$DRY_OUTPUT/source_revision" "$SOURCE_COMMIT" 'provenance records the pinned revision'

RESUME_OUTPUT="$TMP_ROOT/resume-run"
mkdir -p "$RESUME_OUTPUT/checkpoint-2"
printf '{}\n' >"$RESUME_OUTPUT/checkpoint-2/trainer_state.json"
expect_rc 0 "$TMP_ROOT/resume.log" run_wrapper --dry-run RESUME=1 OUTPUT_DIR="$RESUME_OUTPUT"
assert_file_contains "$TMP_ROOT/resume.log" '--resume-from-checkpoint' 'RESUME=1 resumes from the latest checkpoint'
expect_rc 2 "$TMP_ROOT/noresume.log" run_wrapper --dry-run RESUME=0 OUTPUT_DIR="$RESUME_OUTPUT"
assert_file_contains "$TMP_ROOT/noresume.log" 'already populated' 'RESUME=0 never merges into an existing run'

# --- stats mode --------------------------------------------------------------
STATS_ARGV="$TMP_ROOT/stats-both-argv"
FAKE_STATS_ARGV="$STATS_ARGV" expect_rc 0 "$TMP_ROOT/stats-dry.log" run_wrapper --stats --dry-run
assert_file_contains "$TMP_ROOT/stats-dry.log" 'gr00t/data/stats.py' 'stats dry-run prints the official stats writer'
assert_argv_pairs "$TMP_ROOT/stats-dry.log" --embodiment-tag UNITREE_G1_SONIC
if [[ "$(grep -c -Fx -- '--dataset-path' "$TMP_ROOT/stats-dry.log")" == 2 ]]; then
  pass 'stats dry-run covers both splits by default'
else
  fail 'stats dry-run did not print one command per split'
fi
expect_rc 0 "$TMP_ROOT/stats-train.log" run_wrapper --stats --dry-run SPLIT=train
assert_file_contains "$TMP_ROOT/stats-train.log" "$DATASET/train" 'SPLIT=train selects the train split'
if [[ "$(grep -c -Fx -- '--dataset-path' "$TMP_ROOT/stats-train.log")" == 1 ]]; then
  pass 'SPLIT=train prints a single stats command'
else
  fail 'SPLIT=train printed more than one command'
fi

FAKE_STATS_ARGV="$STATS_ARGV" expect_rc 0 "$TMP_ROOT/stats-run.log" run_wrapper --stats SPLIT=train
assert_argv_pairs "$STATS_ARGV" \
  --dataset-path "$DATASET/train" \
  --embodiment-tag UNITREE_G1_SONIC
assert_file_contains "$TMP_ROOT/stats-run.log" 'PASS: official statistics present and validated' \
  'stats mode validates the generated statistics'

FAKE_STATS_ARGV="$STATS_ARGV" FAKE_STATS_CORRUPT=1 FAKE_STATS_CORRUPT_TARGET="$DATASET/train/meta/stats.json" \
  expect_rc 2 "$TMP_ROOT/stats-corrupt.log" run_wrapper --stats SPLIT=train
assert_file_contains "$TMP_ROOT/stats-corrupt.log" 'meta/stats.json is missing' \
  'stats mode fails closed when the writer produced nothing'
cp "$BACKUP/stats.json" "$DATASET/train/meta/stats.json"

FAKE_STATS_ARGV="$STATS_ARGV" FAKE_STATS_FAIL=1 expect_rc 2 "$TMP_ROOT/stats-fail.log" run_wrapper --stats SPLIT=train
assert_file_contains "$TMP_ROOT/stats-fail.log" 'gr00t/data/stats.py failed' 'a failing stats writer is reported'

# --- smoke launch + artifact validation --------------------------------------
SMOKE_OUTPUT="$TMP_ROOT/smoke-run"
FAKE_TRAIN_ARGV="$TMP_ROOT/smoke-run-argv" expect_rc 0 "$TMP_ROOT/smoke-run.log" \
  run_wrapper --smoke OUTPUT_DIR="$SMOKE_OUTPUT"
assert_file_contains "$TMP_ROOT/smoke-run.log" 'PASS: smoke run completed with exactly 2 optimizer steps' \
  'smoke run validates its checkpoint'
assert_file_contains "$SMOKE_OUTPUT/run.log" 'groot-finetune-launcher.py' \
  'the smoke run launches through the optimizer shim'
assert_file_contains "$TMP_ROOT/smoke-run-argv" '--output-dir' 'the launcher receives an output directory'
assert_file_contains "$SMOKE_OUTPUT/exit_code" '0' 'the run records its exit code'
if [[ -s "$SMOKE_OUTPUT/peak_vram_mib" ]]; then
  pass 'the run records peak GPU memory'
else
  fail 'the run did not record peak GPU memory'
fi

MISMATCH_OUTPUT="$TMP_ROOT/smoke-mismatch"
FAKE_TRAIN_ARGV="$TMP_ROOT/smoke-mismatch-argv" FAKE_TRAIN_STEP=3 expect_rc 2 "$TMP_ROOT/smoke-mismatch.log" \
  run_wrapper --smoke OUTPUT_DIR="$MISMATCH_OUTPUT"
assert_file_contains "$TMP_ROOT/smoke-mismatch.log" 'expected exactly 2 optimizer steps' \
  'a checkpoint that is not step 2 fails the smoke validator'

FAILING_OUTPUT="$TMP_ROOT/run-failure"
FAKE_TRAIN_ARGV="$TMP_ROOT/run-failure-argv" FAKE_TRAIN_FAIL=1 FAKE_TRAIN_RC=7 expect_rc 2 "$TMP_ROOT/run-failure.log" \
  run_wrapper --smoke OUTPUT_DIR="$FAILING_OUTPUT"
assert_file_contains "$TMP_ROOT/run-failure.log" 'exit code 7' 'a failing launcher is reported with its code'
assert_file_contains "$FAILING_OUTPUT/launcher_exit_code" '7' 'a failed run records the launcher exit code'
assert_file_contains "$FAILING_OUTPUT/exit_code" '2' 'a failed run records the wrapper exit code'
assert_file_contains "$FAILING_OUTPUT/run.log" 'exit code 7' 'the failing run is still fully logged'

# --- artifact validator (standalone) -----------------------------------------
expect_rc 0 "$TMP_ROOT/validate.log" "$PREFLIGHT" --check artifacts --run-dir "$SMOKE_OUTPUT" --expect-steps 2
expect_rc 0 "$TMP_ROOT/validate-optim.log" "$PREFLIGHT" --check artifacts --run-dir "$SMOKE_OUTPUT" \
  --expect-steps 2 --expect-optim adafactor
expect_rc 2 "$TMP_ROOT/validate-optim-mismatch.log" "$PREFLIGHT" --check artifacts \
  --run-dir "$SMOKE_OUTPUT" --expect-steps 2 --expect-optim adamw_torch
assert_file_contains "$TMP_ROOT/validate-optim-mismatch.log" 'effective optimizer is' \
  'the validator compares the effective optimizer'
expect_rc 0 "$TMP_ROOT/validate-cli.log" bash "$SCRIPT" --validate-artifacts "$SMOKE_OUTPUT" --expect-steps 2
assert_file_contains "$TMP_ROOT/validate-cli.log" 'PASS: valid fine-tune artifacts' 'the CLI validates a good run directory'
OPTIM=adamw_torch expect_rc 2 "$TMP_ROOT/validate-cli-optim.log" bash "$SCRIPT" \
  --validate-artifacts "$SMOKE_OUTPUT" --expect-steps 2
assert_file_contains "$TMP_ROOT/validate-cli-optim.log" 'effective optimizer is' \
  'the CLI validates with the requested OPTIM'
expect_rc 2 "$TMP_ROOT/validate-empty.log" bash "$SCRIPT" --validate-artifacts "$TMP_ROOT/empty-target"
assert_file_contains "$TMP_ROOT/validate-empty.log" 'run directory is missing' 'the CLI refuses a missing run directory'

# --- preflight script direct checks ------------------------------------------
expect_rc 2 "$TMP_ROOT/preflight-json.log" "$PREFLIGHT" --check dataset --dataset-root "$TMP_ROOT/nope" --json
expect_rc 0 "$TMP_ROOT/preflight-ok.log" "$PREFLIGHT" --check dataset --check stats --check hashes \
  --dataset-root "$DATASET" --train-id train --val-id val

# --- tuning profile: default, override, refusals -----------------------------
TUNE_PRINT="$TMP_ROOT/tune-print.log"
expect_rc 0 "$TUNE_PRINT" run_wrapper --print-command
assert_argv_pairs "$TUNE_PRINT" --dataloader-num-workers 8

OVERRIDE_PRINT="$TMP_ROOT/tune-override-print.log"
expect_rc 0 "$OVERRIDE_PRINT" run_wrapper --print-command TUNE_PROJECTOR=1 TUNE_DIFFUSION=0
assert_file_contains "$OVERRIDE_PRINT" '--tune-projector' 'TUNE_PROJECTOR=1 trains the projector'
assert_file_contains "$OVERRIDE_PRINT" '--no-tune-diffusion-model' 'TUNE_DIFFUSION=0 freezes the diffusion head'

expect_rc 2 "$TMP_ROOT/tune-invalid.log" run_wrapper --print-command TUNE_PROJECTOR=maybe
assert_file_contains "$TMP_ROOT/tune-invalid.log" 'TUNE_PROJECTOR must be 0 or 1' \
  'a non-binary tuning knob is refused'
expect_rc 2 "$TMP_ROOT/tune-invalid-diffusion.log" run_wrapper --print-command TUNE_DIFFUSION=yes
assert_file_contains "$TMP_ROOT/tune-invalid-diffusion.log" 'TUNE_DIFFUSION must be 0 or 1' \
  'a non-binary diffusion knob is refused'

expect_rc 2 "$TMP_ROOT/tune-frozen.log" run_wrapper --print-command TUNE_PROJECTOR=0 TUNE_DIFFUSION=0
assert_file_contains "$TMP_ROOT/tune-frozen.log" 'VLLN' \
  'freezing every optional module is refused with the VLLN explanation'
expect_rc 0 "$TMP_ROOT/tune-frozen-ok.log" run_wrapper --print-command --allow-full-tune \
  TUNE_PROJECTOR=0 TUNE_DIFFUSION=0

# TUNE_PROJECTOR=1 is the 40 GiB+ profile: refused on this 32 GiB fixture unless
# the override says so, and the refusal must not pretend batch size is the knob.
expect_rc 2 "$TMP_ROOT/tune-vram.log" run_wrapper --check-only TUNE_PROJECTOR=1
assert_file_contains "$TMP_ROOT/tune-vram.log" 'ALLOW_FULL_TUNE=1' \
  'projector tuning below its VRAM floor is refused with the override hint'
PROJECTOR_REFUSAL="$(grep -F 'FAIL:' "$TMP_ROOT/tune-vram.log" | grep -F 'TUNE_PROJECTOR=1' || true)"
if [[ "$PROJECTOR_REFUSAL" == *batch* ]]; then
  fail 'the projector-tune refusal must not suggest reducing the batch'
else
  pass 'the projector-tune refusal does not suggest reducing the batch'
fi
if [[ "$PROJECTOR_REFUSAL" == *'optimizer step'* ]]; then
  pass 'the projector-tune refusal names the static optimizer-step shortfall'
else
  fail "the projector-tune refusal should name the optimizer step: $PROJECTOR_REFUSAL"
fi
expect_rc 0 "$TMP_ROOT/tune-vram-ok.log" run_wrapper --check-only --allow-full-tune TUNE_PROJECTOR=1
assert_file_contains "$TMP_ROOT/tune-vram-ok.log" 'below TUNE_PROJECTOR_MIN_VRAM_MIB' \
  'the projector-tune override is announced'
assert_file_contains "$TMP_ROOT/tune-vram-ok.log" 'state/action projections + diffusion head' \
  'the preflight echo reports the projector-tuned profile'

# --- tuning profile: provenance ----------------------------------------------
TUNE_DRY="$TMP_ROOT/tune-dry-run"
expect_rc 0 "$TMP_ROOT/tune-dry.log" run_wrapper --dry-run OUTPUT_DIR="$TUNE_DRY"
assert_file_contains "$TUNE_DRY/config.json" '"tune_projector": false' \
  'provenance records the frozen projector'
assert_file_contains "$TUNE_DRY/config.json" '"tune_diffusion_model": true' \
  'provenance records the trainable diffusion head'
assert_file_contains "$TUNE_DRY/config.json" '"profile": "32gb-low-vram"' \
  'provenance records the 32 GB profile'
assert_file_contains "$TUNE_DRY/config.json" 'upstream default' \
  'provenance records that tune_vlln is upstream-controlled'

TUNE_DRY_BIG="$TMP_ROOT/tune-dry-run-big"
expect_rc 0 "$TMP_ROOT/tune-dry-big.log" run_wrapper --dry-run --allow-full-tune TUNE_PROJECTOR=1 \
  OUTPUT_DIR="$TUNE_DRY_BIG"
assert_file_contains "$TUNE_DRY_BIG/config.json" '"tune_projector": true' \
  'provenance records the projector-tuned override'
assert_file_contains "$TUNE_DRY_BIG/config.json" '"profile": "40gb+"' \
  'provenance records the 40 GiB+ profile'
assert_file_contains "$TUNE_DRY_BIG/config.json" '"full_tune": true' \
  'provenance records the explicit override'

# --- optimizer knob ----------------------------------------------------------
# Adafactor is the 32 GB default because AdamW's two fp32 moments need 9,866 MiB
# for the 1,293,159,424 trainable parameters; the knob is validated against the
# shim's allowlist and the effective value is verified from the run itself.
expect_rc 2 "$TMP_ROOT/optim-invalid.log" run_wrapper --print-command OPTIM=paged_adamw_8bit
assert_file_contains "$TMP_ROOT/optim-invalid.log" 'OPTIM must be adafactor' \
  'an 8-bit optimizer name is refused (no bitsandbytes in groot-n17)'

OPTIM_DRY="$TMP_ROOT/optim-dry-run"
expect_rc 0 "$TMP_ROOT/optim-dry.log" run_wrapper --dry-run OPTIM=adamw_torch OUTPUT_DIR="$OPTIM_DRY"
assert_file_contains "$OPTIM_DRY/config.json" '"optimizer": "adamw_torch"' \
  'OPTIM=adamw_torch is recorded in the provenance'
assert_file_contains "$OPTIM_DRY/config.json" '"optimizer": false' \
  'the pinned upstream optimizer is not flagged as an override'
assert_file_contains "$OPTIM_DRY/env.json" '"GROOT_OPTIM": "adamw_torch"' \
  'the shim receives the overridden optimizer'

MISMATCH_OUTPUT="$TMP_ROOT/optimizer-mismatch"
FAKE_TRAIN_ARGV="$TMP_ROOT/optimizer-mismatch-argv" FAKE_OPTIM_EFFECTIVE=adamw_torch \
  expect_rc 2 "$TMP_ROOT/optimizer-mismatch.log" run_wrapper --smoke OUTPUT_DIR="$MISMATCH_OUTPUT"
assert_file_contains "$TMP_ROOT/optimizer-mismatch.log" 'effective optimizer' \
  'a run whose conf.yaml records another optimizer fails smoke validation'

# --- OOM evidence ------------------------------------------------------------
OOM_OUTPUT="$TMP_ROOT/oom-run"
FAKE_TRAIN_ARGV="$TMP_ROOT/oom-argv" FAKE_TRAIN_OOM=1 expect_rc 2 "$TMP_ROOT/oom.log" \
  run_wrapper --smoke OUTPUT_DIR="$OOM_OUTPUT"
assert_file_contains "$TMP_ROOT/oom.log" 'CUDA out of memory' 'an OOM failure surfaces torch wording'
assert_file_contains "$OOM_OUTPUT/oom.txt" 'CUDA out of memory' \
  'torch own memory report is kept as evidence'
assert_file_contains "$TMP_ROOT/oom.log" 'the sampled peak may understate it' \
  'the harness flags that sampling can understate a step-time OOM'

# --- signal-safe logging -----------------------------------------------------
SIGNAL_OUTPUT="$TMP_ROOT/signal-run"
mkdir -p "$SIGNAL_OUTPUT"
OUTPUT_DIR="$SIGNAL_OUTPUT" FAKE_TRAIN_ARGV="$TMP_ROOT/signal-argv" FAKE_TRAIN_SLEEP=30 \
  wrapper_env bash "$SCRIPT" --smoke >"$TMP_ROOT/signal.log" 2>&1 &
WRAPPER_PID=$!
for _ in $(seq 1 100); do
  [[ -s "$SIGNAL_OUTPUT/wrapper.pid" ]] && grep -Fq 'Running:' "$SIGNAL_OUTPUT/run.log" 2>/dev/null && break
  sleep 0.2
done
if [[ -s "$SIGNAL_OUTPUT/wrapper.pid" ]]; then
  pass 'the run records its wrapper pid'
  kill -TERM "$(<"$SIGNAL_OUTPUT/wrapper.pid")" 2>/dev/null || true
else
  fail 'the run did not record its wrapper pid'
fi
signal_rc=0
wait "$WRAPPER_PID" || signal_rc=$?
if [[ "$signal_rc" == 143 ]]; then
  pass 'SIGTERM is reported with exit status 143'
else
  fail "expected exit 143 after SIGTERM, got $signal_rc"
fi
assert_file_contains "$TMP_ROOT/signal.log" 'received SIGTERM' 'the run announces the signal it received'
assert_file_contains "$TMP_ROOT/signal.log" 'Peak GPU memory' 'the signal path still flushes GPU evidence'
assert_file_contains "$SIGNAL_OUTPUT/exit_code" '143' 'the interrupted run records its exit code'
assert_file_contains "$SIGNAL_OUTPUT/run.log" 'received SIGTERM' 'the log survives the interruption'

# --- dev.sh dispatch ---------------------------------------------------------
# The dev.sh entries only add container plumbing; a fake docker records what
# would be executed so the wiring and env forwarding are testable offline.
FAKE_DOCKER_BIN="$TMP_ROOT/docker-bin"
mkdir -p "$FAKE_DOCKER_BIN"
cat >"$FAKE_DOCKER_BIN/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${FAKE_DOCKER_LOG:?}"
case "${1:-}" in
  compose)
    case "${4:-}" in
      ps) echo fake-container;;
      exec) printf 'EXEC %s\n' "${*:5}";;
      up) : ;;
    esac
    ;;
  inspect) echo true ;;
  *) echo "unexpected docker call: $*" >&2; exit 2 ;;
esac
EOF
chmod 0755 "$FAKE_DOCKER_BIN/docker"

run_dev_sh() { # $1 = log file, rest = dev.sh arguments
  local log="$1"
  shift
  PATH="$FAKE_DOCKER_BIN:$PATH" FAKE_DOCKER_LOG="$log" bash "$ROOT/dev.sh" "$@"
}

DEV_CHECK_LOG="$TMP_ROOT/dev-check.log"
expect_rc 0 "$TMP_ROOT/dev-check.out" run_dev_sh "$DEV_CHECK_LOG" groot-dex3-check
assert_file_contains "$DEV_CHECK_LOG" 'scripts/groot-unitree-dex3-sonic.sh --check-only' \
  'dev.sh groot-dex3-check runs the check-only preflight'
assert_file_contains "$DEV_CHECK_LOG" 'use-groot' 'dev.sh selects the groot-n17 environment'

DEV_STATS_LOG="$TMP_ROOT/dev-stats.log"
expect_rc 0 "$TMP_ROOT/dev-stats.out" run_dev_sh "$DEV_STATS_LOG" groot-dex3-stats SPLIT=train
assert_file_contains "$DEV_STATS_LOG" 'scripts/groot-unitree-dex3-sonic.sh --stats' \
  'dev.sh groot-dex3-stats runs the official stats writer'

DEV_SMOKE_LOG="$TMP_ROOT/dev-smoke.log"
expect_rc 0 "$TMP_ROOT/dev-smoke.out" run_dev_sh "$DEV_SMOKE_LOG" groot-dex3-smoke --dry-run
assert_file_contains "$DEV_SMOKE_LOG" 'scripts/groot-unitree-dex3-sonic.sh --smoke' \
  'dev.sh groot-dex3-smoke runs the two-step stage'
assert_file_contains "$DEV_SMOKE_LOG" 'groot-dex3-smoke --dry-run' \
  'dev.sh groot-dex3-smoke forwards wrapper flags as container argv'

DEV_RUN_LOG="$TMP_ROOT/dev-run.log"
expect_rc 0 "$TMP_ROOT/dev-run.out" env BATCH=1 ACCUM=32 RESUME=1 OUTPUT_DIR=/tmp/groot-dex3-run \
  TUNE_PROJECTOR=1 TUNE_DIFFUSION=0 OPTIM=adamw_torch \
  FAKE_DOCKER_LOG="$DEV_RUN_LOG" PATH="$FAKE_DOCKER_BIN:$PATH" bash "$ROOT/dev.sh" groot-dex3-run --print-command
assert_file_contains "$DEV_RUN_LOG" 'groot-dex3-run --print-command' \
  'dev.sh groot-dex3-run forwards wrapper flags as container argv'
assert_file_contains "$DEV_RUN_LOG" '-e ACCUM=32' 'dev.sh forwards the accumulation knob'
assert_file_contains "$DEV_RUN_LOG" '-e OUTPUT_DIR=/tmp/groot-dex3-run' 'dev.sh forwards the output directory'
assert_file_contains "$DEV_RUN_LOG" '-e RESUME=1' 'dev.sh forwards the resume knob'
assert_file_contains "$DEV_RUN_LOG" '-e TUNE_PROJECTOR=1' 'dev.sh forwards the projector tuning knob'
assert_file_contains "$DEV_RUN_LOG" '-e TUNE_DIFFUSION=0' 'dev.sh forwards the diffusion tuning knob'
assert_file_contains "$DEV_RUN_LOG" '-e OPTIM=adamw_torch' 'dev.sh forwards the optimizer knob'

DEV_TESTS_LOG="$TMP_ROOT/dev-tests.log"
expect_rc 0 "$TMP_ROOT/dev-tests.out" run_dev_sh "$DEV_TESTS_LOG" groot-dex3-tests
assert_file_contains "$DEV_TESTS_LOG" 'tests/test_groot_unitree_dex3_sonic.sh' \
  'dev.sh groot-dex3-tests runs the mocked shell suite'
assert_file_contains "$DEV_TESTS_LOG" 'test_groot_unitree_dex3_sonic_preflight.py' \
  'dev.sh groot-dex3-tests runs the preflight unit tests'
assert_file_contains "$DEV_TESTS_LOG" 'test_groot_finetune_launcher.py' \
  'dev.sh groot-dex3-tests runs the launcher-shim unit tests'

for command in groot-dex3-stats groot-dex3-check groot-dex3-smoke groot-dex3-run groot-dex3-tests; do
  if grep -Fq -- "|$command" "$ROOT/dev.sh"; then
    pass "dev.sh usage lists $command"
  else
    fail "dev.sh usage does not list $command"
  fi
done

printf 'summary: PASS=%d FAIL=%d\n' "$PASS" "$FAIL"
(( FAIL == 0 ))
