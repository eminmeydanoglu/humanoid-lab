#!/usr/bin/env bash
# Downloads pinned HF models into the persistent data dir.
# - Always `--revision <full sha>` + explicit `--local-dir`.
# - Writes MODEL_PROVENANCE.json per model (repo, revision, time, files, SHA-256s). Idempotent.
# - Gated models: report "blocked" on 401/403. Token never written to image/Git.
# Usage: ./dev.sh fetch-models (in container, after hf-login) |
#        scripts/fetch-models.sh (direct; HF_HOME must be set)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_HOME:=${HUMANOID_DATA_ROOT:-$HOME/humanoid-lab-data}/hf-cache}"
MODEL_ROOT="${HUMANOID_DATA_ROOT:-$HOME/humanoid-lab-data}/models"
export HF_HOME MODEL_ROOT

eval "$(python3 scripts/render-lock-env.py --root . >/dev/null && cat .generated/versions.env | grep '^MODELS_' )"
# The python block below re-reads the lock structurally; env output is loaded for compatibility only.
python3 - <<'PY'
import json, subprocess, sys, os, hashlib, time, shutil
import yaml

lock = yaml.safe_load(open('versions.lock.yaml'))
models = lock['models']
root = os.environ['MODEL_ROOT']
os.makedirs(root, exist_ok=True)

# Modern HF CLI: `hf` (huggingface_hub>=0.26). Legacy `huggingface-cli` is removed upstream.
CLI = shutil.which('hf') or shutil.which('huggingface-cli')
if not CLI:
    print('[blocked] HF CLI not found. Install inside the container: '
          'uv pip install --python /opt/venvs/isaac-sonic/bin/python "huggingface_hub[cli]"')
    sys.exit(2)
CLI_NAME = os.path.basename(CLI)
print(f'[info] HF CLI: {CLI_NAME}')

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

rc = 0
for name, m in models.items():
    repo, rev = m['repo'], m['revision']
    variant = m.get('variant', '')
    if not rev:
        print(f"[skip] {name}: no revision in lock")
        continue
    dest = os.path.join(root, name)
    os.makedirs(dest, exist_ok=True)
    print(f"== {name} :: {repo} @ {rev} -> {dest}")
    if variant:
        print(f"   (variant: {variant})")
    args = [CLI, 'download', repo, '--revision', rev, '--local-dir', dest]
    if variant:
        args += ['--include', f'{variant}/*']
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        print(f"   [blocked] {repo} download failed (401/403 means gated access/license acceptance needed: "
              f"https://huggingface.co/{repo})")
        rc = 2
        continue
    prov = {"repo": repo, "revision": rev, "variant": variant,
            "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": []}
    for dirpath, _dirs, files in os.walk(dest):
        for fn in sorted(files):
            fp = os.path.join(dirpath, fn)
            if fn == 'MODEL_PROVENANCE.json':
                continue
            prov["files"].append({"path": os.path.relpath(fp, dest),
                                  "sha256": sha256_of(fp)})
    with open(os.path.join(dest, 'MODEL_PROVENANCE.json'), 'w') as f:
        json.dump(prov, f, indent=2)
        f.write('\n')
    print(f"   [ok] {len(prov['files'])} files, wrote MODEL_PROVENANCE.json")
sys.exit(rc)
PY