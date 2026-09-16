#!/usr/bin/env bash
# Fetch unitree_sim_isaaclab's scene assets into the persistent data root.
#
# The USD assets are not git content and not in the image: they ship as a >1 GiB
# assets.zip inside the Hugging Face *dataset* repo pinned in versions.lock.yaml
# (models.unitree_sim_assets). This mirrors fetch-models.sh: everything lands
# under $HUMANOID_DATA_ROOT/models, and a provenance file records what was
# fetched. The pinned image layer /opt/src/unitree-sim/assets is then linked at
# the extracted tree so sim_main.py finds its relative assets/ path.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HUMANOID_DATA_ROOT:=/data}"
: "${HF_HOME:=/cache/huggingface}"
export HUMANOID_DATA_ROOT HF_HOME

python3 - "$HUMANOID_DATA_ROOT" <<'PY'
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
import yaml

data_root = sys.argv[1]
entry = yaml.safe_load(open('versions.lock.yaml'))['models']['unitree_sim_assets']
repo = entry['repo']
revision = entry['revision']
archive = entry.get('archive', 'assets.zip')
min_bytes = int(entry.get('min_archive_bytes', 0))

dest_dir = os.path.join(data_root, 'models', entry.get('dest_dir', 'unitree-sim-assets'))
assets_dest = os.path.join(dest_dir, 'assets')
work = os.path.join(dest_dir, '.fetch-work')

cli = shutil.which('hf') or shutil.which('huggingface-cli')
if not cli:
    print('[blocked] HF CLI not found; run this through ./dev.sh fetch-unitree-sim-assets', file=sys.stderr)
    sys.exit(2)

if os.path.exists(work):
    shutil.rmtree(work)
os.makedirs(work, exist_ok=True)
os.makedirs(dest_dir, exist_ok=True)

print(f'== unitree_sim_assets :: {repo} @ {revision}')
print(f'   {archive} -> {dest_dir}')
subprocess.run([cli, 'download', repo, archive,
                '--repo-type', 'dataset', '--revision', revision,
                '--local-dir', work], check=True)

archive_path = os.path.join(work, archive)
if not os.path.isfile(archive_path):
    # Some CLI versions nest the dataset name; locate the archive by name.
    for dirpath, _dirs, files in os.walk(work):
        if archive in files:
            archive_path = os.path.join(dirpath, archive)
            break
if not os.path.isfile(archive_path):
    print(f'[fail] {archive} not found after download', file=sys.stderr)
    sys.exit(2)

size = os.path.getsize(archive_path)
if size < min_bytes:
    print(f'[fail] {archive} is {size} bytes (< min_archive_bytes {min_bytes}); '
          f'likely a git-lfs pointer or a truncated download', file=sys.stderr)
    sys.exit(2)
print(f'   archive size: {size / 1024 / 1024:.0f} MiB')

digest = hashlib.sha256()
with open(archive_path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        digest.update(chunk)
archive_sha = digest.hexdigest()

extract = os.path.join(work, 'extract')
os.makedirs(extract, exist_ok=True)
with zipfile.ZipFile(archive_path) as zf:
    zf.extractall(extract)

# The upstream archive carries a top-level assets/ directory.
src_assets = os.path.join(extract, 'assets')
if not os.path.isdir(src_assets):
    matches = [os.path.join(r, d) for r, dirs, _f in os.walk(extract) for d in dirs if d == 'assets']
    if not matches:
        print('[fail] no assets/ directory inside the archive', file=sys.stderr)
        sys.exit(2)
    src_assets = matches[0]

if os.path.exists(assets_dest):
    shutil.rmtree(assets_dest)
shutil.move(src_assets, assets_dest)

file_count = 0
total_bytes = 0
for dirpath, _dirs, files in os.walk(assets_dest):
    for fn in files:
        fp = os.path.join(dirpath, fn)
        file_count += 1
        total_bytes += os.path.getsize(fp)

prov = {
    'repo': repo,
    'kind': entry.get('kind', 'dataset'),
    'revision': revision,
    'archive': archive,
    'archive_sha256': archive_sha,
    'archive_bytes': size,
    'assets_dir': os.path.relpath(assets_dest, dest_dir),
    'file_count': file_count,
    'total_bytes': total_bytes,
    'fetched_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
with open(os.path.join(dest_dir, 'ASSET_PROVENANCE.json'), 'w') as f:
    json.dump(prov, f, indent=2)
    f.write('\n')
print(f'   [ok] {file_count} files, {total_bytes / 1024 / 1024:.0f} MiB extracted')

shutil.rmtree(work, ignore_errors=True)

# Point the pinned image layer at the fetched tree (assets/ is gitignored
# upstream, so the source layer stays byte-identical).
link = '/opt/src/unitree-sim/assets'
if os.path.isdir('/opt/src/unitree-sim'):
    if os.path.islink(link) or not os.path.exists(link):
        if os.path.islink(link):
            os.remove(link)
        os.symlink(assets_dest, link)
        print(f'   [ok] linked {link} -> {assets_dest}')
    else:
        print(f'   [warn] {link} exists and is not a symlink; leaving it untouched', file=sys.stderr)
else:
    print('   [warn] /opt/src/unitree-sim absent; skipped the assets link', file=sys.stderr)
PY
