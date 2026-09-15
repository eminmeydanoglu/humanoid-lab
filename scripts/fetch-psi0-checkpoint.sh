#!/usr/bin/env bash
# Fetch the pinned Psi0 SONIC warm-start checkpoint into PSI_HOME and record a
# per-file sha256 provenance file next to it.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HUMANOID_DATA_ROOT:=/data}"
: "${HF_HOME:=/cache/huggingface}"
export HF_HOME
# Psi0 resolves checkpoints under PSI_HOME by convention; upstream recipes
# default to /hfm, which this checkout maps to the persistent data root.
: "${PSI_HOME:=/hfm}"
export PSI_HOME

python3 - "$HUMANOID_DATA_ROOT" "$PSI_HOME" <<'PY'
import hashlib, json, os, shutil, subprocess, sys, time
import yaml

data_root, psi_home = sys.argv[1], sys.argv[2]
entry = yaml.safe_load(open('versions.lock.yaml'))['models']['psi0_sonic_warmstart']
repo, revision, variant = entry['repo'], entry['revision'], entry['variant']
local_dir = entry.get('local_dir', 'psi/cache/checkpoints')
# local_dir is data-root relative; PSI_HOME must resolve to the same tree.
dest_root = os.path.join(data_root, local_dir)
ckpt_dir = os.path.join(dest_root, variant)
os.makedirs(dest_root, exist_ok=True)

resolved_psi_home = os.path.realpath(psi_home)
if os.path.realpath(os.path.join(psi_home, 'cache/checkpoints')) != os.path.realpath(dest_root):
    print(f'[fail] PSI_HOME {psi_home} does not map to {dest_root}; the checkpoint would be '
          f'outside PSI_HOME (resolved {resolved_psi_home})', file=sys.stderr)
    sys.exit(2)

cli = shutil.which('hf') or shutil.which('huggingface-cli')
if not cli:
    print('[blocked] HF CLI not found; run this through ./dev.sh fetch-psi0-ckpt', file=sys.stderr)
    sys.exit(2)

print(f'== psi0_sonic_warmstart :: {repo} @ {revision}')
print(f'   {variant} -> {ckpt_dir}')
try:
    subprocess.run([cli, 'download', repo, '--include', f'{variant}/*',
                    '--revision', revision, '--local-dir', dest_root], check=True)
except subprocess.CalledProcessError:
    print(f'   [blocked] {repo} download failed; a 404 here means the revision or the '
          f'directory moved upstream: https://huggingface.co/{repo}/tree/{revision}/{variant}',
          file=sys.stderr)
    sys.exit(2)

def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()

provenance = {'repo': repo, 'revision': revision, 'variant': variant,
              'fetched_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'files': []}
for dirpath, _dirs, files in os.walk(ckpt_dir):
    for name in sorted(files):
        path = os.path.join(dirpath, name)
        if name == 'MODEL_PROVENANCE.json':
            continue
        provenance['files'].append({'path': os.path.relpath(path, ckpt_dir),
                                    'size': os.path.getsize(path),
                                    'sha256': sha256_of(path)})
with open(os.path.join(ckpt_dir, 'MODEL_PROVENANCE.json'), 'w') as handle:
    json.dump(provenance, handle, indent=2)
    handle.write('\n')
print(f"   [ok] {len(provenance['files'])} files, wrote MODEL_PROVENANCE.json")
PY
