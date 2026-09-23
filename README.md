# humanoid-lab


## Setup from scratch

```bash
git clone git@github.com:eminmeydanoglu/humanoid-lab.git
cd humanoid-lab
git submodule update --init third_party/Psi0   # pinned Psi0 checkout, no nesting needed

docker login nvcr.io        # the Isaac Sim base image is EULA-gated on NGC
./setup.sh --verify-digests # resolves the base image digest from NGC into versions.lock.yaml
./setup.sh                  # preflight, .env, image build, container start, smoke test
```

`setup.sh --verify-digests` needs NGC access. Without a verified digest in the lock, `setup.sh`
refuses to build

Models are a separate, revision-pinned step:

```bash
./dev.sh hf-login      # once, if you need gated Hugging Face repositories; the token stays in the host HF cache
./dev.sh fetch-models  # pinned revisions into $HUMANOID_DATA_ROOT/models, plus MODEL_PROVENANCE.json with sha256 per file
```

`nvidia/Cosmos-Reason2-2B`, the GR00T N1.7 backbone, is gated: accept its license on Hugging Face
before `fetch-models`, otherwise the GR00T checks stay `BLOCKED`.

`./dev.sh fetch-psi0-ckpt` is the separate step for the pinned Psi0 SONIC checkpoints — the warm
start every fine-tune lineage is initialized from, and the released multi-task policy the unified
evaluation UI offers as its ψ-Dream option:

```bash
./dev.sh fetch-psi0-ckpt                    # both, with MODEL_PROVENANCE.json next to each
./dev.sh fetch-psi0-ckpt psi0_sonic_dream   # one lock entry by name
```


## Training environments

Two fine-tuning environments live inside the same container, isolated from each other and from the
simulation environments:

| Environment | Source | Python | What it is |
| --- | --- | --- | --- |
| `groot-n17` | `/opt/src/isaac-groot` (image layer) | 3.12 | Isaac-GR00T N1.7 fine-tuning; torch 2.9.0+cu128, flash_attn 2.8.3 |
| `psi0` | `third_party/Psi0` (git submodule) | 3.11 | Psi0 fine-tuning, `serve`/`viz`/`psi` groups; torch 2.7.0+cu128 |

`./dev.sh sync` provisions whichever environment's lock or pinned source changed. `./dev.sh psi0`
opens a shell with the Psi0 environment active, and `./dev.sh psi0-smoke` checks the interpreter,
the PyTorch build, the upstream import stack, the resolved configuration schema and the warm-start
checkpoint without running a dataset or an optimizer step.

PSI_HOME is `/hfm`, as the upstream recipes assume; the image builds it from symlinks into the
persistent data root (`/hfm/cache/checkpoints` → `/data/checkpoints`, `/hfm/data` → `/data/datasets`),
so no upstream script is edited and nothing is duplicated.


## G1 on Isaac Sim

```bash
./dev.sh isaac-g1 dex3          # Dex3 hands
./dev.sh isaac-g1 inspire-ftp   # Inspire FTP hands
./dev.sh isaac-g1 no_hands      # 29-DoF body only
```

Interactive runs are WebRTC servers by default: Isaac Sim starts without a local
window and its UI is streamed, so nothing appears on the host's physical screen.
Watch the run with `./dev.sh webrtc-client` from any machine that reaches the
host — see [Remote viewing (WebRTC)](#remote-viewing-webrtc).

Flags:

| Flag | Effect |
| --- | --- |
| `--gui` | opens the local X11 window instead of streaming; needs a display |
| `--headless` | no UI, no livestream, no X11 requirement |
| `--head-camera-window` | opens the 640x480 head camera in a second GPU viewport; costs render throughput |
| `--duration SECONDS` | bounded run |
| `--device {cpu,cuda}` | overrides the profile device; profiles default to CPU |
| `--controller none` | disables the controller a profile declares, falling back to passive |
| `--test {passive-fall\|controlled-hold\|controller-hold}` | one of the acceptance modes below |


## Remote viewing (WebRTC)

Isaac Sim's own WebRTC livestream is the supported way to watch a run: the
application renders without a local window and serves its full UI to a client, so
the host's physical screen stays free and no port has to be exposed to the
internet when both machines share a Tailscale network. There is no browser
client — NVIDIA ships a native viewer for Linux, Windows and macOS.

| Side | What to do |
| --- | --- |
| Server | `./dev.sh isaac-g1 dex3` (streams by default), or `./dev.sh isaac-stream` for the empty Isaac Sim UI |
| Address | the host's Tailscale IPv4, advertised automatically; override with `ISAAC_LIVESTREAM_ENDPOINT`, port `49100` |
| Viewer | `./scripts/install-isaac-webrtc-client.sh` once per machine, then `./dev.sh webrtc-client`: enter the address and connect |

The viewer needs no Isaac Sim installation and no GPU. One client connects at a
time; the stream ends when the run ends.


## Isaac and SONIC in two terminals

```bash
# terminal 1 — Isaac Sim: scene, physics loop, robot state on DDS, incoming joint commands applied
./dev.sh isaac-g1-sonic dex3

# terminal 2 — the official SONIC deployment: planner and keyboard on this terminal
./dev.sh sonic-controller
```

Keys in the SONIC terminal: `]` starts control, `T` plays the reference motion, `N` and `P` change
motion, `R` resets, `O` is the emergency stop. `Enter` to switch to planner to teleop.


## PSI / GR00T policy evaluation (BlockStacking)

One command starts the shared evaluation environment: Isaac BlockStacking, one
SONIC controller and one browser UI. The UI keeps running while its model
selector switches between the two PSI checkpoints and NVIDIA GR00T.

```bash
./dev.sh psi0-isaac-eval \
  --checkpoint-dir <psi-run-directory> \
  --checkpoint-step 40000 \
  --groot-checkpoint-dir <groot-checkpoint-directory>
```

`<run-directory>` is a Psi0 training output (`run_config.json`, `argv.txt`,
`checkpoints/ckpt_<step>`); host paths under the data root and the in-container
spellings (`/outputs/...`, `/data/...`) are both accepted. The command validates
the checkpoint, the action port (`:5556`) and the served `/info` contract before
anything is served, and stops with the reason instead of guessing. Serving is
CUDA-only: the bridge spawns the deployment with its canonical `cuda:0` command,
and a machine whose GPU cannot host the checkpoint cannot serve it — a CPU device
would load the model and answer `/info`, then fail on the first action, because
the deployment's inference path runs under CUDA autocast.

Open `http://localhost:8015/` for Start/Stop/Reset, model selection and the
head-camera preview, and watch the simulation through the printed WebRTC
endpoint. SONIC runs unattended in `zmq_manager` mode. `Ctrl-C` in the terminal,
or a dead shared process, ends the whole session.

The UI's **Model** panel offers these allowlisted choices (the ψ-Dream one only
when the launcher was given its run directory):

* `Fine-tuned (40k)` — the PSI run given on the command line;
* `Base` — that PSI run's training-start checkpoint: the warm start its own
  `run_config.json` names (`model.model_name_or_path`), materialized on first
  launch into a servable run dir under `<experiment root>/base/<warm start>`.
  The released warm start ships HuggingFace-style `model.safetensors` plus a
  separate `action_header.safetensors`; the materializer merges them into the
  deploy loader's key layout and disables `model.state_null_token`, a parameter
  the warm start does not contain and that inference never reads. Its provenance
  (source files, upstream revision, key counts) is written next to it as
  `BASE_ARTIFACT.json`;
* `ψ-Dream (40k)` — the released multi-task SONIC checkpoint given with
  `--psi-dream-checkpoint-dir` (`./dev.sh fetch-psi0-ckpt psi0_sonic_dream`
  downloads it into `$HUMANOID_DATA_ROOT/checkpoints/psi0/sonic-checkpoints/`).
  It ships as a complete run directory, so it is served exactly like a fine-tune
  run; its own `run_config.json` names one camera
  (`observation.images.head`) and a 270x480 transform, and the bridge sends its
  frame under whatever single key the selected run declares. Its pack declares
  no action mask (`repack.action_mask_key` is null), so the two trailing neck
  columns of its 80D action carry values the no-op check rejects (the first
  observed chunk held `[-0.025, -0.349]` against a 0.05 tolerance); the
  evaluation target has no neck field in Protocol v4, so a run with this option
  needs `--psi0-neck-policy discard`, which drops the block and records the
  dropped values in telemetry;
* `GR00T` — the checkpoint given with `--groot-checkpoint-dir`, served by
  NVIDIA's unmodified `run_gr00t_server.py` and `run_vla_inference.py` path.

Selecting an option stops the session and changes only the policy backend. PSI
restarts its policy server on `:8014`; GR00T starts NVIDIA's PolicyServer on
`:5550` and VLA client on the private router input `:5560`. Isaac, SONIC, the
camera and the UI stay alive. The shared router remains the sole owner of public
SONIC action port `:5556`.

A PSI selection is only marked ready after `/info` matches the run directory,
step, action/state width, image key, transforms and dataset the selected run's
own `run_config.json` declares. A failed switch is fail-closed: the previous
backend is started again and the session goes to `ERROR` with the reason. If the
base artifact cannot be derived, the `Base` option is shown unavailable with the
reason. The API accepts allowlisted model ids only; paths never come from the
browser.

### BlockStacking evaluation contract (existing checkpoints)

The short launch command above now defaults to **Isaac simulation time** for
both policies and the training-matched GR00T `model-independent` left-hand order.
`dev.sh` gives each session one fresh shared clock file and passes it to Isaac,
the bridge and the policy backends; the rollout driver also creates its own clock
file. No clock or GR00T hand opt-in is required. Direct invocation of
`scripts/psi0-isaac-eval.py` still requires an Isaac process writing its declared
clock file. Use `--policy-clock wall --groot-left-hand-contract compatibility`
*only* to reproduce historical runs. These corrections do not establish plant
or camera parity. Record the selected backend/checkpoint and flags with each
run; do not compare outcomes from different contracts as if they were the same
experiment. Psi0 80D checkpoints may predict
nonzero values in the final two **unsupervised neck-padding** channels: Protocol
v4 cannot carry a neck command, so the bridge fails closed by default. For a
checkpoint whose training labels confirm those two channels are masked padding,
pass `--psi0-neck-policy discard` together with `--telemetry-dir PATH`; each
discarded value is then recorded in `target_action` telemetry. Do not enable this for a checkpoint that actually
supervises neck movement.

`--gravity-feedforward` compensates the simulated body's gravity load and has
substantially improved joint-target tracking in controlled comparisons, but
changes the closed-loop plant of checkpoints collected under the old controller.
It is **opt-in**, not a universally correct switch or a task-success fix. Run
with and without it as separately labelled dynamics contracts until the
training/deployment controller is established. Existing Psi0 training data also
has four right-hand measured-state channels with outlier-corrupted min/max
statistics; correcting the converter will not repair weights already trained
with those statistics. Do not mix corrected stats with an old checkpoint and
claim an equivalent inference contract. `--initial-pose-handshake` now waits
for the simulator's applied-reset `episode_id` before settling, but does **not**
yet acknowledge subscriber readiness or that the initial-pose command itself
was applied. Current scene/camera values are reproducible simulator settings,
not calibrated measurements of the demonstration rig. The opt-in
`--scene-profile configs/profiles/isaac-g1-sonic-blockstacking-dex3-training-plant.json`
uses the SONIC training URDF's **fixed-joint-merged** masses, training-config
joint armatures and self-collision setting instead of the welded deployment
model. It is only a plant **approximation**: the USD's center-of-mass, inertia
and collision geometry are not replaced, its spawn height remains the
legacy setting, and this profile has not been exercised in a live Isaac run.
The arms' merged masses match between the two SONIC models; do not ascribe the
high-arm behavior to a fictitious heavier evaluation arm. The complete evidence,
including failed corrected rollouts and remaining unknowns, is in
`data/outputs/blockstacking-debug/followup-diagnosis/REPORT.md`; the focused
training/inference contract review and current limits are in
`docs/blockstacking-contract-audit.md`.

## If you need


| Command | What it does |
| --- | --- |
| `./dev.sh` | interactive shell in the container, no environment pre-selected |
| `./dev.sh isaac-g1-test-controller dex3` | G1 driven by the deterministic scripted test controller |
| `./dev.sh isaac-demo <demo.py>` | an Isaac Lab demo from `/opt/src/isaaclab/scripts/demos`, streamed over WebRTC |
| `./dev.sh isaac-stream` | the Isaac Sim UI with no scene, streamed over WebRTC |
| `./dev.sh webrtc-client` | Isaac Sim WebRTC client, for watching a streaming host |
| `./dev.sh doctor` | full host + container report, written to `$HUMANOID_DATA_ROOT/diagnostics/` |
| `./dev.sh smoke` | in-container environment, asset and model checks |
| `./dev.sh sync` | re-provisions the environment whose lock or pinned source changed |
| `./dev.sh fetch-models` | downloads the pinned model revisions | |
| `./dev.sh fetch-psi0-ckpt [NAME]` | downloads the pinned Psi0 SONIC checkpoints (warm start and multi-task ψ-Dream) into PSI_HOME; `NAME` selects one lock entry | |
| `./dev.sh groot-finetune-smoke [pipeline\|pretrained\|eval]` | optional GR00T N1.7 fine-tuning smoke (two optimizer steps, then open-loop eval) |
| `./dev.sh psi0-smoke` | Psi0 environment smoke: interpreter, torch build, upstream imports, config schema, warm-start checkpoint |
| `./dev.sh psi0-isaac-eval --checkpoint-dir PSI_RUN --checkpoint-step STEP --groot-checkpoint-dir GROOT_CHECKPOINT [--psi-dream-checkpoint-dir PSI_DREAM_RUN]` | one shared BlockStacking UI for Fine-tuned PSI, Base PSI, ψ-Dream and NVIDIA GR00T |
| `./dev.sh hf-login` | Hugging Face login for gated repositories |
| `./dev.sh stop` | stops the container, keeps the data root |
| `./dev.sh rebuild` | rebuilds the image with the same lock, recreates the container |
| `./dev.sh isaac` | shell with the `isaac-sonic` environment active |
| `./dev.sh sonic-sim` | shell with the `sonic-sim` environment (MuJoCo + G1) |
| `./dev.sh groot` | shell with the `groot-n17` environment |
| `./dev.sh psi0` | shell with the `psi0` environment |




## Repository layout

| Path | Contents |
| --- | --- |
| `setup.sh`, `dev.sh`, `doctor.sh` | host entry points: provisioning, container lifecycle and runs, diagnostics |
| `compose.yaml`, `Dockerfile`, `containers/` | image and container definition, entrypoint, environment selectors, venv provisioning |
| `configs/profiles/` | G1 run profiles: robot asset, hands, camera, controller, support band |
| `src/humanoid_lab/` | simulator service, profile and command contracts, controller bridge |
| `scripts/` | Isaac G1 runner, model fetching, lock rendering, checks |
| `locks/` | frozen `uv` locks for the four Python environments |
| `third_party/Psi0/` | pinned Psi0 checkout (git submodule); the `psi0` environment is built from it |
| `tests/` | `unittest` modules and shell checks |
| `docs/` | SONIC-Isaac link notes with the measured acceptance results |
| `versions.lock.yaml` | single source of every version pin |



## Data root, mounts and environments

`HUMANOID_DATA_ROOT` in `.env` defaults to `./data` and holds everything persistent:
`datasets`, `checkpoints`, `models`, `hf-cache`, `uv-cache`, `venvs`, `isaac-cache`, `outputs`,
`diagnostics`, `rosbags`, `runtime`. It is bind-mounted into the container as `/data`, `/cache`,
`/outputs` and `/opt/venvs`. The repository itself is mounted at `/workspace/humanoid-lab`.


## Requirements

| Item | Value |
| --- | --- |
| OS / arch | Ubuntu 24.04, x86_64 |
| GPU | NVIDIA GPU, driver >= 580.65.06 (newer is kept, older fails the preflight) |
| Docker | Docker Engine with the Compose v2 plugin, plus the NVIDIA Container Toolkit |
| Disk | more than 100 GiB free where the repo lives |
| Host tools | `git`, `curl`, `python3` with PyYAML |
| NGC | account with the Isaac Sim EULA accepted, and `docker login nvcr.io` performed |
| GUI (optional) | a working X11 display; headless runs need none |
| Hugging Face (optional) | token, only for the gated GR00T backbone |

