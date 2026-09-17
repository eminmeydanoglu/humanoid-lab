#!/usr/bin/env bash
# Replay every prepared run directory through the free SONIC pipeline.
#
# Sequential by design: the official deployment, its ZMQ port and the DDS topics
# are single-instance on this machine, so one episode runs at a time.  A failing
# episode is recorded and the batch keeps going; the batch exits non-zero at the
# end so a partial run cannot pass as complete.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

batch_root="${1:?usage: run-latent-replay-batch.sh <batch_root> [extra run-sonic-pilot-sim.sh flags]}"
shift

status=0
ok=0
failed=0
mapfile -t run_dirs < <(find "$batch_root" -mindepth 2 -maxdepth 2 -type d -name 'episode_*' | sort)
if [ "${#run_dirs[@]}" -eq 0 ]; then
  echo "error: no prepared run directories under $batch_root" >&2
  exit 2
fi
echo "[latent-batch] ${#run_dirs[@]} episodes under $batch_root"

for run_dir in "${run_dirs[@]}"; do
  echo "[latent-batch] $(basename "$(dirname "$run_dir")")/$(basename "$run_dir")"
  if bash scripts/run-sonic-pilot-sim.sh "$run_dir" --no-direct --no-kinematic "$@"; then
    ok=$((ok + 1))
  else
    failed=$((failed + 1))
    status=1
    echo "[latent-batch] FAILED $run_dir" >&2
  fi
  echo "[latent-batch] progress ok=$ok failed=$failed"
done

echo "[latent-batch] complete ok=$ok failed=$failed"
exit "$status"
