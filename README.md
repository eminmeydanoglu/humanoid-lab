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

`./dev.sh fetch-psi0-ckpt` is the separate step for the Psi0 warm start:

```bash
./dev.sh fetch-psi0-ckpt   # 11 GiB, pinned revision, MODEL_PROVENANCE.json next to the weights
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


## Psi0 policy in the loop (BlockStacking)

One command starts the whole evaluation: the Isaac BlockStacking scene, the
official SONIC Y controller and the bridge/UI (which owns the Psi0 policy
server it serves).

```bash
./dev.sh psi0-isaac-eval --checkpoint-dir <run-directory> --checkpoint-step 40000
```

`<run-directory>` is a Psi0 training output (`run_config.json`, `argv.txt`,
`checkpoints/ckpt_<step>`); host paths under the data root and the in-container
spellings (`/outputs/...`, `/data/...`) are both accepted. The command validates
the checkpoint, the action port (`:5556`) and the served `/info` contract before
anything is served, and stops with the reason instead of guessing.

The SONIC controller keeps this terminal, started in its pose-streaming mode so
it can consume the policy's Protocol v4 messages: press `Enter` to enable the
pose stream, then `]` to start control. Open `http://localhost:8015/` for
Start/Stop/Reset, the checkpoint selector and the head-camera preview, and watch
the simulation through the printed WebRTC endpoint. `Ctrl-C` in the terminal, or
a dead background process, ends the whole session.

The UI's **Policy checkpoint** panel offers exactly two allowlisted choices:

* `Fine-tuned (40k)` — the run given on the command line;
* `Base` — that run's training-start checkpoint: the warm start its own
  `run_config.json` names (`model.model_name_or_path`), materialized on first
  launch into a servable run dir under `<experiment root>/base/<warm start>`.
  The released warm start ships HuggingFace-style `model.safetensors` plus a
  separate `action_header.safetensors`; the materializer merges them into the
  deploy loader's key layout and disables `model.state_null_token`, a parameter
  the warm start does not contain and that inference never reads. Its provenance
  (source files, upstream revision, key counts) is written next to it as
  `BASE_ARTIFACT.json`.

Selecting an option stops the session, restarts the policy server on `:8014`
with the selected run dir/step, and only marks it ready after `/info` matches
the run dir, step, action/state width, transforms and dataset. A failed switch
is fail-closed: the previous checkpoint is started again and the session goes to
`ERROR` with the reason. If the base artifact cannot be derived, the `Base`
option is shown unavailable with the reason — no other checkpoint is
substituted. The API takes an allowlisted option id only; paths are not
accepted.


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
| `./dev.sh fetch-psi0-ckpt` | downloads the pinned Psi0 SONIC warm start into PSI_HOME | |
| `./dev.sh groot-finetune-smoke [pipeline\|pretrained\|eval]` | optional GR00T N1.7 fine-tuning smoke (two optimizer steps, then open-loop eval) |
| `./dev.sh psi0-smoke` | Psi0 environment smoke: interpreter, torch build, upstream imports, config schema, warm-start checkpoint |
| `./dev.sh psi0-isaac-eval --checkpoint-dir RUN_DIR --checkpoint-step STEP` | one-command Psi0-in-the-loop BlockStacking evaluation: Isaac scene, SONIC Y, policy server, bridge/UI |
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

