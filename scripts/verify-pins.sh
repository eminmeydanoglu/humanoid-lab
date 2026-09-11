#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

NEED="python3 curl git"
for c in $NEED; do command -v "$c" >/dev/null || { echo "error: $c required" >&2; exit 2; }; done
python3 -c 'import yaml' 2>/dev/null || { echo "error: PyYAML required" >&2; exit 2; }

STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

python3 scripts/render-lock-env.py --root . >/dev/null
LOCK=$(python3 - <<'PY'
import yaml
d = yaml.safe_load(open('versions.lock.yaml'))
print(yaml.safe_dump(d, sort_keys=False))
PY
)
export LOCK

have_sha() { # $1 repo, $2 sha -> 0/1 (exact SHA in refs, else GitHub API)
  if git ls-remote "$1" "$2" 2>/dev/null | grep -q "^$2"; then return 0; fi
  local api_path code
  api_path="${1#https://github.com/}"
  api_path="${api_path%.git}"
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://api.github.com/repos/$api_path/commits/$2")
  [ "$code" = "200" ]
}
have_tag() { # $1 repo, $2 tag -> 0/1
  git ls-remote --tags "$1" "refs/tags/$2" 2>/dev/null | grep -q "refs/tags/$2"
}
have_hf_rev() { # $1 repo, $2 sha -> 0/1
  curl -fsS "https://huggingface.co/api/models/$1/revision/$2" >/dev/null 2>&1
}

RC=0
warn() { echo "  [warn] $*" >&2; RC=1; }
err()  { echo "  [fail] $*" >&2; RC=2; }

python3 - <<'PY' > /tmp/pins.tsv
import os, yaml
d = yaml.safe_load(os.environ['LOCK'])
for name, r in d['repositories'].items():
    if not isinstance(r, dict):
        continue
    print('repo', name, r['url'], r.get('commit',''), r.get('release',''))
for name, m in d['models'].items():
    if not isinstance(m, dict) or 'repo' not in m:
        # A pinned local artifact (content hash), not a download.
        continue
    print('model', name, m['repo'], m.get('revision',''), str(m.get('gated','')).lower())
PY

while read -r kind name url commit extra; do
  if [ "$kind" = repo ]; then
    if [ -n "$commit" ] && ! have_sha "$url" "$commit"; then
      err "repo $name: commit $commit not found upstream ($url)"
    else
      echo "  [ok] repo $name commit $commit"
    fi
    if [ -n "$extra" ]; then
      if have_tag "$url" "$extra"; then
        echo "  [ok] repo $name tag $extra exists"
      else
        warn "repo $name: tag '$extra' not found upstream (commit pin valid)"
      fi
    fi
  elif [ "$kind" = model ]; then
    if [ -z "$commit" ]; then
      warn "model $name: no revision in lock"
    elif have_hf_rev "$url" "$commit"; then
      echo "  [ok] model $url revision $commit (metadata)"
      if [ -n "$extra" ] && [ "$extra" != "false" ] && [ "$extra" != "none" ]; then
        warn "model $url: gated=$extra — metadata OK; download needs HF license acceptance (checked by fetch-models)"
      fi
    else
      err "model $url: revision $commit not found (gated repos may need access)"
    fi
  fi
done < /tmp/pins.tsv

PENDING=$(python3 - <<'PY'
import sys; sys.path.insert(0, 'scripts')
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location('rle', Path('scripts/render-lock-env.py'))
rle = importlib.util.module_from_spec(spec)
sys.modules['rle'] = rle
spec.loader.exec_module(rle)
p = rle.collect_required_false(rle.load_lock(Path('versions.lock.yaml')))
print('\n'.join(p))
PY
)
if [ -n "$PENDING" ]; then
  echo
  echo "  verified-later fields (required:false — verify on machine, then write to lock):"
  # shellcheck disable=SC2001 # Prefixing every reported field is more readable here.
  echo "$PENDING" | sed 's/^/    - /'
  if [ "$STRICT" = 1 ]; then RC=2; fi
fi

if [ "$RC" = 0 ]; then
  echo "verify-pins: PASS"
elif [ "$RC" = 1 ]; then
  echo "verify-pins: PASS (with warnings)" >&2
else
  echo "verify-pins: FAIL" >&2
fi
exit "$RC"
