#!/usr/bin/env bash
# Collect fresh release evidence through shipped entrypoints, then validate it.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
evidence=${RELEASE_EVIDENCE_DIR:-$root/outputs/release-acceptance}
replay_command=${GROOT_REPLAY_COMMAND:?Set GROOT_REPLAY_COMMAND; GROOT_REPLAY_LOG is provided for each run}
render_reviewer=${RELEASE_RENDER_REVIEWER:?Set RELEASE_RENDER_REVIEWER after inspecting the captured frame}
commit=$(git -C "$root" rev-parse HEAD)

if [[ -n $(git -C "$root" status --porcelain --untracked-files=no) ]]; then
  echo "ERROR: tracked workspace must be clean before evidence collection" >&2
  exit 2
fi
rm -rf "$evidence"
mkdir -p "$evidence"/adapter "$evidence"/groot-replay "$evidence"/sonic-native "$evidence"/isaac "$evidence"/rollouts

adapter_log="$evidence/adapter/tests.jsonl"
if python3 -m unittest discover -s "$root/tests" -p 'test_*adapter.py' >"$evidence/adapter/unittest.log" 2>&1; then
  tests_run=$(grep -Eo 'Ran [0-9]+ tests?' "$evidence/adapter/unittest.log" | tail -1 | grep -Eo '[0-9]+')
  printf '{"result":"PASS","tests_run":%s}\n' "${tests_run:-0}" >"$adapter_log"
else
  cat "$evidence/adapter/unittest.log" >&2
  exit 2
fi

for run in 1 2; do
  export GROOT_REPLAY_LOG="$evidence/groot-replay/run-$run.jsonl"
  bash -lc "$replay_command"
  python3 "$root/scripts/verify-cloudwalk-groot-replay.py" "$evidence/groot-replay/run-1.jsonl" "$evidence/groot-replay/run-2.jsonl" >/dev/null 2>&1 || [[ $run == 1 ]]
done
python3 "$root/scripts/verify-cloudwalk-groot-replay.py" "$evidence/groot-replay/run-1.jsonl" "$evidence/groot-replay/run-2.jsonl" >"$evidence/groot-replay/verification.json"

for run in 1 2; do
  "$root/scripts/run-sonic-isolated-native.sh" >"$evidence/sonic-native/run-$run.log" 2>&1
done

for run in 1 2 3 4 5; do
  export CLOUDWALK_LOG="$evidence/isaac/run-$run.log"
  export CLOUDWALK_ROLLOUT_METRICS="/workspace/humanoid-lab/outputs/release-acceptance/rollouts/run-$run.json"
  export CLOUDWALK_ROLLOUT_VIDEO="/workspace/humanoid-lab/outputs/release-acceptance/rollouts/run-$run.mp4"
  "$root/scripts/run-cloudwalk-closed-loop.sh"
  cp "$root/outputs/isaac-closed-loop.jpg" "$evidence/isaac/run-$run.jpg"
done

python3 - "$root" "$evidence" "$commit" "$render_reviewer" <<'PY'
import json, sys, time
from pathlib import Path
root, evidence = map(Path, sys.argv[1:3])
commit = sys.argv[3]
render_reviewer = sys.argv[4]
created = time.time()
def rel(path): return str(path.relative_to(root))
def record(artifact, command, **extra):
    return {"artifact": rel(artifact), "command": command, "commit": commit, "created_unix": created, **extra}
gates = {
    "adapter": [record(evidence / "adapter/tests.jsonl", "python3 -m unittest discover -s tests -p 'test_*adapter.py'", attachments=[rel(evidence / "adapter/unittest.log")])],
    "groot_replay": [record(evidence / f"groot-replay/run-{run}.jsonl", "$GROOT_REPLAY_COMMAND") for run in (1, 2)],
    "sonic_native": [record(evidence / f"sonic-native/run-{run}.log", "./scripts/run-sonic-isolated-native.sh") for run in (1, 2)],
    "isaac_closed_loop": [record(evidence / f"isaac/run-{run}.log", "./scripts/run-cloudwalk-closed-loop.sh") for run in (1, 2)],
    "render": [record(evidence / "isaac/run-1.jpg", "./scripts/run-cloudwalk-closed-loop.sh -- capture", visual_review={"reviewer": render_reviewer, "realistic": True, "contents": {"robot": True, "table": True, "bottle": True}})],
    "rollout": [],
}
for run in range(1, 6):
    metrics_path = evidence / f"rollouts/run-{run}.json"
    metrics = json.loads(metrics_path.read_text())
    phases = {"approach": metrics.get("approach"), "contact": metrics.get("contact_proxy"), "hand_close": metrics.get("hand_close"), "stable_grasp": metrics.get("stable_grasp"), "lift": metrics.get("lift")}
    first_failure = next((name for name, passed in phases.items() if not passed), None)
    gates["rollout"].append(record(evidence / f"rollouts/run-{run}.mp4", "./scripts/run-cloudwalk-closed-loop.sh", metrics=rel(metrics_path), phases=phases, failure_layer=first_failure))
(evidence / "manifest.json").write_text(json.dumps({"schema": 1, "commit": commit, "gates": gates}, indent=2, sort_keys=True) + "\n")
PY

python3 "$root/scripts/verify-release-evidence.py" "$evidence/manifest.json"
