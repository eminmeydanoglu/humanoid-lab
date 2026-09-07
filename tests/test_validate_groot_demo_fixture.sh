#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATOR="$ROOT/scripts/validate-groot-demo-fixture.sh"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/validate-groot-demo-fixture.XXXXXX")"
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

assert_contains() {
  local path="$1"
  local needle="$2"
  local message="$3"
  if grep -Fq -- "$needle" "$path"; then
    pass "$message"
  else
    fail "$message (missing: $needle)"
  fi
}

make_fixture() {
  local target="$1"
  mkdir -p "$target/meta" "$target/data" "$target/videos"
  printf '{}\n' >"$target/meta/modality.json"
  printf 'valid parquet payload\n' >"$target/data/00-valid.parquet"
  printf 'valid MP4 payload\n' >"$target/videos/00-valid.mp4"
}

VALID="$TMP_ROOT/valid"
make_fixture "$VALID"
expect_rc 0 "$TMP_ROOT/valid.log" "$VALIDATOR" "$VALID"

PARQUET_POINTER="$TMP_ROOT/parquet-pointer"
cp -a "$VALID" "$PARQUET_POINTER"
printf 'version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 1\n' >"$PARQUET_POINTER/data/99-pointer.parquet"
expect_rc 2 "$TMP_ROOT/parquet-pointer.log" "$VALIDATOR" "$PARQUET_POINTER"
assert_contains "$TMP_ROOT/parquet-pointer.log" '99-pointer.parquet' 'later parquet LFS pointer is rejected'

MP4_POINTER="$TMP_ROOT/mp4-pointer"
cp -a "$VALID" "$MP4_POINTER"
printf 'version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 1\n' >"$MP4_POINTER/videos/99-pointer.mp4"
expect_rc 2 "$TMP_ROOT/mp4-pointer.log" "$VALIDATOR" "$MP4_POINTER"
assert_contains "$TMP_ROOT/mp4-pointer.log" '99-pointer.mp4' 'later MP4 LFS pointer is rejected'

MISSING_METADATA="$TMP_ROOT/missing-metadata"
cp -a "$VALID" "$MISSING_METADATA"
rm "$MISSING_METADATA/meta/modality.json"
expect_rc 2 "$TMP_ROOT/missing-metadata.log" "$VALIDATOR" "$MISSING_METADATA"
assert_contains "$TMP_ROOT/missing-metadata.log" 'modality metadata is missing' 'missing modality metadata is rejected'

printf 'summary: PASS=%d FAIL=%d\n' "$PASS" "$FAIL"
(( FAIL == 0 ))
