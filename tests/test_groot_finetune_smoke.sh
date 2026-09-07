#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/groot-finetune-smoke.sh"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/groot-finetune-smoke.XXXXXX")"
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
  local path="$1"
  local needle="$2"
  local message="$3"
  if grep -Fq -- "$needle" "$path"; then
    pass "$message"
  else
    fail "$message (missing: $needle)"
  fi
}

assert_file_not_contains() {
  local path="$1"
  local needle="$2"
  local message="$3"
  if grep -Fq -- "$needle" "$path"; then
    fail "$message (unexpected: $needle)"
  else
    pass "$message"
  fi
}

expect_rc() {
  local expected="$1"
  local log="$2"
  shift 2
  set +e
  "$@" >"$log" 2>&1
  local actual=$?
  set -e
  if [[ "$actual" == "$expected" ]]; then
    pass "exit status $expected: ${*##*/}"
  else
    fail "expected exit $expected, got $actual: $*"
    sed 's/^/  | /' "$log" >&2 || true
  fi
}

SOURCE="$TMP_ROOT/source"
MODEL="$TMP_ROOT/model"
DATASET="$SOURCE/demo_data/cube_to_bowl_5"
OUTPUT_ROOT="$TMP_ROOT/outputs"
FAKE_BIN="$TMP_ROOT/bin"
mkdir -p "$SOURCE/gr00t/experiment" "$SOURCE/gr00t/eval" "$SOURCE/examples/SO100" \
  "$DATASET/meta" "$DATASET/data" "$DATASET/videos" "$MODEL" "$FAKE_BIN" "$OUTPUT_ROOT"
printf '# placeholder launcher\n' >"$SOURCE/gr00t/experiment/launch_finetune.py"
printf '# placeholder evaluator\n' >"$SOURCE/gr00t/eval/open_loop_eval.py"
printf '# placeholder modality config\n' >"$SOURCE/examples/SO100/so100_config.py"
printf '{}\n' >"$DATASET/meta/modality.json"
printf '%s\n' '{"repo":"https://github.com/NVIDIA/Isaac-GR00T.git","revision":"SOURCE_COMMIT"}' >"$DATASET/DATASET_PROVENANCE.json"
printf 'parquet payload\n' >"$DATASET/data/episode.parquet"
printf 'video payload\n' >"$DATASET/videos/episode.mp4"
printf '{"model_type":"Gr00tN1d7"}\n' >"$MODEL/config.json"
printf '%s\n' '{"repo":"nvidia/GR00T-N1.7-3B","revision":"2fc962b973bccdd5d8ce4f67cc63b264d6886495","files":[]}' >"$MODEL/MODEL_PROVENANCE.json"

cat >"$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  -L) echo 'GPU 0: Test GPU';;
  --query-gpu=memory.total) echo 24576;;
  --query-gpu=name,driver_version,memory.total) echo 'Test GPU, 555.00, 24576 MiB';;
  *) echo "unexpected nvidia-smi arguments: $*" >&2; exit 2;;
esac
EOF
chmod 0755 "$FAKE_BIN/nvidia-smi"

git -C "$SOURCE" init -q
git -C "$SOURCE" config user.email groot-smoke@example.invalid
git -C "$SOURCE" config user.name groot-smoke-test
git -C "$SOURCE" add .
git -C "$SOURCE" commit -qm 'test source'
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
printf '{"repo":"https://github.com/NVIDIA/Isaac-GR00T.git","revision":"%s"}\n' "$SOURCE_COMMIT" >"$DATASET/DATASET_PROVENANCE.json"

run_smoke() {
  PATH="$FAKE_BIN:$PATH" \
  GROOT_SOURCE_DIR="$SOURCE" \
  GROOT_PYTHON="${GROOT_PYTHON:-$(command -v python3)}" \
  GROOT_JSON_PYTHON="${GROOT_JSON_PYTHON:-$(command -v python3)}" \
  GROOT_MODEL_DIR="$MODEL" \
  GROOT_DATASET_DIR="$DATASET" \
  GROOT_OUTPUT_ROOT="$OUTPUT_ROOT" \
  GROOT_EXPECTED_SOURCE_COMMIT="$SOURCE_COMMIT" \
  GROOT_SMOKE_SKIP_ENV_IMPORT=1 \
  "$SCRIPT" "$@"
}

run_smoke_with_image_fixture() {
  PATH="$FAKE_BIN:$PATH" \
  GROOT_SOURCE_DIR="$SOURCE" \
  GROOT_PYTHON="${GROOT_PYTHON:-$(command -v python3)}" \
  GROOT_JSON_PYTHON="${GROOT_JSON_PYTHON:-$(command -v python3)}" \
  GROOT_MODEL_DIR="$MODEL" \
  GROOT_DATASET_DIR='' \
  GROOT_OUTPUT_ROOT="$OUTPUT_ROOT" \
  GROOT_EXPECTED_SOURCE_COMMIT="$SOURCE_COMMIT" \
  GROOT_SMOKE_SKIP_ENV_IMPORT=1 \
  "$SCRIPT" "$@"
}

PIPELINE_OUTPUT="$TMP_ROOT/pipeline-output"
expect_rc 0 "$TMP_ROOT/pipeline.log" run_smoke pipeline --dry-run --output-dir "$PIPELINE_OUTPUT"
assert_file_contains "$PIPELINE_OUTPUT/command.sh" '--num-gpus 1' 'pipeline command pins one GPU'
assert_file_contains "$PIPELINE_OUTPUT/command.sh" '--max-steps 2' 'pipeline command limits training to two steps'
assert_file_contains "$PIPELINE_OUTPUT/command.sh" '--global-batch-size 2' 'pipeline command uses global batch size two'
assert_file_contains "$PIPELINE_OUTPUT/command.sh" '--dataloader-num-workers 0' 'pipeline command disables loader workers'
assert_file_contains "$PIPELINE_OUTPUT/command.sh" '--shard-size 64' 'pipeline command limits shard size'
assert_file_contains "$PIPELINE_OUTPUT/command.sh" '--num-shards-per-epoch 1' 'pipeline command limits shards per epoch'
assert_file_contains "$PIPELINE_OUTPUT/command.sh" '--skip-weight-loading' 'pipeline command skips pretrained weights'
assert_file_contains "$PIPELINE_OUTPUT/run.log" 'command:' 'pipeline persists a run log'
assert_file_contains "$PIPELINE_OUTPUT/source_revision" "$SOURCE_COMMIT" 'pipeline persists the source revision'
assert_file_contains "$PIPELINE_OUTPUT/model_provenance.json" '2fc962b973bccdd5d8ce4f67cc63b264d6886495' 'pipeline persists model provenance'

IMAGE_FIXTURE_OUTPUT="$TMP_ROOT/image-fixture-output"
expect_rc 0 "$TMP_ROOT/image-fixture.log" run_smoke_with_image_fixture pipeline --dry-run --output-dir "$IMAGE_FIXTURE_OUTPUT"
assert_file_contains "$IMAGE_FIXTURE_OUTPUT/command.sh" "--dataset-path $DATASET" 'pipeline defaults to the image-contained demo fixture'

PRETRAINED_OUTPUT="$TMP_ROOT/pretrained-output"
expect_rc 3 "$TMP_ROOT/pretrained-blocked.log" run_smoke pretrained --dry-run --output-dir "$PRETRAINED_OUTPUT"
assert_file_contains "$TMP_ROOT/pretrained-blocked.log" 'BLOCKED:' 'pretrained mode blocks below 40 GiB by default'
assert_file_contains "$TMP_ROOT/pretrained-blocked.log" '--allow-low-vram' 'pretrained block documents the explicit override'

PRETRAINED_OVERRIDE_OUTPUT="$TMP_ROOT/pretrained-override-output"
expect_rc 0 "$TMP_ROOT/pretrained-override.log" run_smoke pretrained --allow-low-vram --dry-run --output-dir "$PRETRAINED_OVERRIDE_OUTPUT"
assert_file_not_contains "$PRETRAINED_OVERRIDE_OUTPUT/command.sh" '--skip-weight-loading' 'pretrained command loads real weights'
assert_file_contains "$PRETRAINED_OVERRIDE_OUTPUT/command.sh" '--max-steps 2' 'pretrained command keeps the two-step limit'

CHECKPOINT="$TMP_ROOT/checkpoint-2"
mkdir -p "$CHECKPOINT/experiment_cfg" "$CHECKPOINT/model"
printf '{}\n' >"$CHECKPOINT/model/config.json"
printf 'weights\n' >"$CHECKPOINT/model/model.safetensors"
printf '{}\n' >"$CHECKPOINT/processor_config.json"
printf '{}\n' >"$CHECKPOINT/statistics.json"
expect_rc 0 "$TMP_ROOT/training-artifacts.log" "$SCRIPT" --check-training-artifacts "$CHECKPOINT"

EVAL_OUTPUT="$TMP_ROOT/eval-output"
expect_rc 0 "$TMP_ROOT/eval.log" run_smoke eval --dry-run --checkpoint-dir "$CHECKPOINT" --output-dir "$EVAL_OUTPUT"
assert_file_contains "$EVAL_OUTPUT/command.sh" '--execution-horizon 16' 'eval command uses the N1.7 execution-horizon flag'
assert_file_contains "$EVAL_OUTPUT/command.sh" '--steps 5' 'eval command limits open-loop steps'
assert_file_contains "$EVAL_OUTPUT/command.sh" '--save-plot-path' 'eval command requests a plot artifact'

mkdir -p "$EVAL_OUTPUT/open_loop_eval"
printf 'Average MSE: 0.125\nAverage MAE: 0.25\n' >"$EVAL_OUTPUT/open_loop_eval.log"
printf 'plot\n' >"$EVAL_OUTPUT/open_loop_eval/traj_0.jpeg"
expect_rc 0 "$TMP_ROOT/eval-artifacts.log" "$SCRIPT" --check-eval-artifacts "$EVAL_OUTPUT"

expect_rc 2 "$TMP_ROOT/dirty-output.log" run_smoke pipeline --dry-run --output-dir "$PIPELINE_OUTPUT"
assert_file_contains "$TMP_ROOT/dirty-output.log" 'must not already exist' 'training output must be clean'

expect_rc 2 "$TMP_ROOT/duplicate-mode.log" "$SCRIPT" pipeline pretrained
assert_file_contains "$TMP_ROOT/duplicate-mode.log" 'only one mode may be supplied' 'duplicate mode parsing is rejected'

printf '%s\n' '{"repo":"nvidia/GR00T-N1.7-3B","revision":"wrong","files":[]}' >"$MODEL/MODEL_PROVENANCE.json"
expect_rc 2 "$TMP_ROOT/provenance.log" run_smoke pipeline --dry-run --output-dir "$TMP_ROOT/provenance-output"
assert_file_contains "$TMP_ROOT/provenance.log" 'provenance revision does not match the lock' 'provenance mismatch blocks training'
printf '%s\n' '{"repo":"nvidia/GR00T-N1.7-3B","revision":"2fc962b973bccdd5d8ce4f67cc63b264d6886495","files":[]}' >"$MODEL/MODEL_PROVENANCE.json"

printf 'version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 1\n' >"$DATASET/data/episode.parquet"
expect_rc 2 "$TMP_ROOT/lfs-pointer.log" run_smoke pipeline --dry-run --output-dir "$TMP_ROOT/lfs-pointer-output"
assert_file_contains "$TMP_ROOT/lfs-pointer.log" 'unresolved Git LFS pointer' 'unresolved demo data LFS pointer blocks training'
printf 'parquet payload\n' >"$DATASET/data/episode.parquet"

cat >"$FAKE_BIN/gated-python" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == "--version" || "\${1:-}" == "-" ]]; then
  exec "$(command -v python3)" "\$@"
fi
printf 'GatedRepoError: restricted model access\n' >&2
exit 1
EOF
chmod 0755 "$FAKE_BIN/gated-python"
run_gated_smoke() {
  GROOT_PYTHON="$FAKE_BIN/gated-python" run_smoke "$@"
}

GATED_OUTPUT="$TMP_ROOT/gated-output"
expect_rc 3 "$TMP_ROOT/gated.log" run_gated_smoke pipeline --output-dir "$GATED_OUTPUT"
assert_file_contains "$TMP_ROOT/gated.log" 'BLOCKED: Cosmos backbone access is gated' 'gated Cosmos errors are reported as blocked'
assert_file_contains "$TMP_ROOT/gated.log" './dev.sh hf-login' 'gated Cosmos errors provide login guidance'

printf 'summary: PASS=%d FAIL=%d\n' "$PASS" "$FAIL"
(( FAIL == 0 ))
