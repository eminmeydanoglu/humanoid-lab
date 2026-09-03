#!/usr/bin/env bash
# In-repo static checks (bash syntax, shellcheck, compose, lock/pin).
# Usage: scripts/ci-lint.sh   (exit 0/1/2)
set -euo pipefail
cd "$(dirname "$0")/.."
RC=0

SCRIPTS=$(find . -path ./.git -prune -o -path ./.generated -prune -o -type f \( -name '*.sh' -o -name 'setup.sh' -o -name 'dev.sh' -o -name 'doctor.sh' \) -print)
echo "== bash -n =="
for f in $SCRIPTS; do
  bash -n "$f" && echo "  ok: $f" || { echo "  FAIL: $f"; RC=2; }
done

if command -v shellcheck >/dev/null 2>&1; then
  echo "== shellcheck =="
  shellcheck -x $SCRIPTS || RC=1
else
  echo "== shellcheck missing (skipping; install: apt install shellcheck) =="
fi

echo "== python/yaml =="
python3 -m py_compile scripts/render-lock-env.py scripts/doctor-report.py && echo "  render-lock-env.py + doctor-report.py compiled"
python3 -c "import yaml; yaml.safe_load(open('versions.lock.yaml')); print('  versions.lock.yaml: valid YAML')"

echo "== render determinism =="
python3 scripts/render-lock-env.py --root . >/dev/null
cp .generated/versions.env /tmp/versions.env.1
python3 scripts/render-lock-env.py --root . >/dev/null
if cmp -s /tmp/versions.env.1 .generated/versions.env; then
  echo "  render: deterministic (same output)"
else
  echo "  FAIL: render not deterministic"; RC=2
fi

echo "== lock consistency =="
python3 - <<'PY'
import yaml, sys
d = yaml.safe_load(open('versions.lock.yaml'))
ok = True
def chk(cond, msg):
    global ok
    if not cond:
        print('  FAIL:', msg); ok = False
chk(d.get('schema') == 1, 'schema != 1')
chk(len(str(d['repositories']['isaac_lab'].get('commit',''))) == 40, 'isaac_lab commit 40h')
chk(len(str(d['repositories']['sonic'].get('commit',''))) == 40, 'sonic commit 40h')
chk(len(str(d['repositories']['isaac_groot'].get('commit',''))) == 40, 'groot commit 40h')
for name in ('sonic','groot_n17_base','cosmos_reason_backbone'):
    chk(len(str(d['models'][name].get('revision',''))) == 40, f'model {name} revision 40h')
sys.exit(0 if ok else 2)
PY
PYRC=$?
[ "$PYRC" != 0 ] && RC=$PYRC

echo "== done: RC=$RC =="
exit "$RC"