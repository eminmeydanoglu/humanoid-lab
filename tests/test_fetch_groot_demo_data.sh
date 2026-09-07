#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/fetch-groot-demo-data.sh"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fetch-groot-demo-data.XXXXXX")"
cleanup() {
  [[ "${KEEP_TEST_TMP:-0}" == 1 ]] || rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

pass() {
  printf 'PASS: %s\n' "$1"
}

REMOTE="$TMP_ROOT/remote.git"
WORKTREE="$TMP_ROOT/worktree"
FAKE_BIN="$TMP_ROOT/bin"
DATA_ROOT="$TMP_ROOT/data"
mkdir -p "$FAKE_BIN"

git init -q --bare "$REMOTE"
git clone -q "$REMOTE" "$WORKTREE"
git -C "$WORKTREE" config user.email groot-demo-test@example.invalid
git -C "$WORKTREE" config user.name groot-demo-test
mkdir -p "$WORKTREE/demo_data/cube_to_bowl_5/meta" \
  "$WORKTREE/demo_data/cube_to_bowl_5/data" \
  "$WORKTREE/demo_data/cube_to_bowl_5/videos"
printf '{}\n' >"$WORKTREE/demo_data/cube_to_bowl_5/meta/modality.json"
printf 'parquet payload\n' >"$WORKTREE/demo_data/cube_to_bowl_5/data/episode.parquet"
printf 'video payload\n' >"$WORKTREE/demo_data/cube_to_bowl_5/videos/episode.mp4"
git -C "$WORKTREE" add demo_data/cube_to_bowl_5
git -C "$WORKTREE" commit -qm 'add GR00T demo dataset fixture'
git -C "$WORKTREE" push -q origin HEAD
SOURCE_COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"

cat >"$FAKE_BIN/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == '-C' && "${3:-}" == 'lfs' && "${4:-}" == 'pull' ]]; then
  exit 0
fi
exec "$REAL_GIT" "$@"
EOF
cat >"$FAKE_BIN/git-lfs" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod 0755 "$FAKE_BIN/git" "$FAKE_BIN/git-lfs"

PATH="$FAKE_BIN:$PATH" \
REAL_GIT="$(command -v git)" \
HUMANOID_DATA_ROOT="$DATA_ROOT" \
GROOT_DATASET_REPO="$REMOTE" \
GROOT_DATASET_REVISION="$SOURCE_COMMIT" \
"$SCRIPT"

TARGET="$DATA_ROOT/datasets/groot/cube_to_bowl_5"
[[ -f "$TARGET/meta/modality.json" ]] || fail 'dataset modality metadata was not moved into the mounted dataset directory'
[[ -f "$TARGET/data/episode.parquet" ]] || fail 'dataset parquet was not moved into the mounted dataset directory'
[[ -f "$TARGET/videos/episode.mp4" ]] || fail 'dataset video was not moved into the mounted dataset directory'
[[ -f "$TARGET/DATASET_PROVENANCE.json" ]] || fail 'dataset provenance was not written at the mounted dataset root'
python3 - "$TARGET/DATASET_PROVENANCE.json" "$SOURCE_COMMIT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    provenance = json.load(handle)
assert provenance["revision"] == sys.argv[2]
assert {item["path"] for item in provenance["files"]} >= {
    "meta/modality.json",
    "data/episode.parquet",
    "videos/episode.mp4",
}
PY
pass 'pinned demo dataset and provenance are stored under the persistent dataset mount'

grep -Fq "git -c http.version=HTTP/1.1 -C \"\$target\" fetch --depth=1 origin \"\$commit\"" "$ROOT/Dockerfile" || \
  fail 'Dockerfile must use the container-compatible pinned Git fetch transport'
grep -Fq "git -C /opt/src/isaac-groot lfs pull --include 'demo_data/cube_to_bowl_5/**'" "$ROOT/Dockerfile" || \
  fail 'Dockerfile must materialize only the pinned GR00T demo fixture'
grep -Fq "test \"\$(head -c 42 \"\$asset\")\" != 'version https://git-lfs.github.com/spec/v1'" "$ROOT/Dockerfile" || \
  fail 'Dockerfile must reject unresolved LFS fixture assets'
if grep -Fq 'GIT_LFS_SKIP_SMUDGE=1 git -C /opt/src/isaac-groot checkout' "$ROOT/Dockerfile" || \
  grep -Fq 'rm -rf /opt/src/isaac-groot/.git/lfs/objects' "$ROOT/Dockerfile"; then
  fail 'Dockerfile must retain the validated fixture payload in the final image'
fi
pass 'Docker build retains a validated, materialized GR00T fixture in the final image'
