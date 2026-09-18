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
| `--render-interval N` | physics ticks per rendered frame (profile default 8, about 25 FPS); sets the physics/viewport balance, and did not change the achievable rate in measurement |
| `--no-dlssg` | disables DLSS frame generation; measured 206.5 → 213.0 Hz and a lower render call (9.0 → 8.0 ms) |
| `--perf-detail` | adds the per-phase step breakdown to the per-second performance line |
| `--test {passive-fall\|controlled-hold\|controller-hold}` | one of the acceptance modes below |

## Interactive performance

Both G1 paths are paced to wall clock when a controller is attached, so the loop
rate *is* the acceptance number: a run that cannot keep 200 Hz is visibly behind
real time for the controller talking to it over DDS.

Measured on this host with `scripts/benchmark-sim.py` (WebRTC streaming UI, one
Isaac process at a time, 10 s warm-up + 20 s window, medians of the samples in
that window). The paced pairs are the same command before and after the
deadline-based pacing change; the `misses` column is not comparable across it
because the counter changed meaning (see `docs/sonic-isaac.md`).

| Run | Command | Physics | RTF | Render | step | render_call |
| --- | --- | --- | --- | --- | --- | --- |
| Flat, free | `./dev.sh isaac-g1-sonic dex3 --controller none` | 207 Hz | 1.03 | 26 FPS | 3.7 ms | 9.0 ms |
| Flat, free, no DLSS-G | `... --controller none --no-dlssg` | 213 Hz | 1.06 | 27 FPS | 3.7 ms | 8.0 ms |
| Flat, headless | `... --controller none --headless` | 419 Hz | 2.09 | — | 2.4 ms | — |
| Flat, paced, before deadline pacing | `./dev.sh isaac-g1-sonic dex3` | 160 Hz | 0.80 | 20 FPS | 4.5 ms | 9.7 ms |
| Flat, paced, after deadline pacing | `./dev.sh isaac-g1-sonic dex3` | 162 Hz | 0.81 | 20 FPS | 4.5 ms | 9.6 ms |
| Rough, free | `./dev.sh isaac-g1-sonic-rough dex3 --controller none` | 195 Hz | 0.97 | 24 FPS | 4.0 ms | 8.8 ms |
| Rough, paced, before deadline pacing | `./dev.sh isaac-g1-sonic-rough dex3` | 132 Hz | 0.66 | 17 FPS | 4.5 ms | 20.3 ms |
| Rough, paced, after deadline pacing | `./dev.sh isaac-g1-sonic-rough dex3` | 139 Hz | 0.69 | 18 FPS | 4.5 ms | 17.3 ms |

Where the time goes (`--perf-detail`, flat, streamed UI): PhysX is ~3.2 ms of
the ~3.7 ms step, staging the tick ~0.3 ms, the scene's own buffer refresh
~0.05 ms; the render call is ~9 ms and runs every 8th tick. The streamed UI
costs about half the physics rate: the same scene headless runs at 2.09x RTF.

Things that were measured and did **not** help on this host, so they are not
knobs: PhysX worker threads (4 > 16 > 24; 1 and 2 unchanged), the PhysX->USD
writeback cadence, the asset's contact sensors (nothing reads them), and the
render cadence — halving the render rate doubles the cost of each render call,
leaving the per-second render cost unchanged. The one repeatable render-side
gain is `--no-dlssg`.

```
scripts/benchmark-sim.py --list                       # the case matrix
scripts/benchmark-sim.py --perf-detail                # all cases, one at a time
scripts/benchmark-sim.py --case sonic-flat-paced --extra "--no-dlssg"
```

Each case runs the canonical launcher, discards the warm-up window, aggregates
the samples the path itself prints, samples GPU/CPU alongside, and writes one
JSON artifact under `.generated/benchmarks/` (never committed). Read it as:
*paced* RTF ≥ 1 means the run held real time; `pacing_overruns` counts the ticks
that missed their deadline; the *free* cases (no controller, or `--no_realtime`)
measure the headroom the paced run has to work with. For the Instinct playback,
whose own counters are cumulative, the harness reports the windowed rate derived
from consecutive samples.

Instinct's analysis on this host is in [docs/instinct-parkour.md](docs/instinct-parkour.md).
The apparent 24-core CPU bottleneck was ONNX Runtime's two default spinning
thread pools, not PhysX: explicit serial, non-spinning sessions reduced the
process from about 22.6 CPU cores to 1.1. Removing a duplicate Kit main-loop
rate limiter then reduced streamed render cost from about 20.6 ms to 7.9 ms.
With real depth and policy enabled, streamed playback improved from about 0.28
to **0.46 RTF** (~22.8 of the 50 target policy steps/s). The remaining critical
path is four GPU PhysX steps (~21.6 ms), required depth observation processing
(~6.5 ms) and rendering; the physics steps alone exceed the 20 ms real-time
policy budget. The harness reports windowed rates, while playback's own
`loops/s` counter is cumulative.


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
| Viewer | `./dev.sh webrtc-view` starts the client on emin-1 and connects it; `./dev.sh webrtc-view raider` watches from the MSI laptop |

`webrtc-view` copies a small helper over ssh, starts the client on that machine,
points it at this host and confirms the connection from this side. NVIDIA's
client takes no server argument, so the address is written into the client's own
storage through its DevTools port. Each viewer machine needs the client
(`./scripts/install-isaac-webrtc-client.sh`) and `python3-websocket` once; the
laptop's account is named once with `ISAAC_VIEW_RAIDER=user@host`. The command
does nothing while that machine is already watching — `ISAAC_VIEW_RESTART=1`
restarts its client instead.

The viewer needs no Isaac Sim installation and no GPU. One client connects at a
time; the stream ends when the run ends.


## Isaac and SONIC in two terminals

```bash
# terminal 1 — Isaac Sim: scene, physics loop, robot state on DDS, incoming joint commands applied
./dev.sh isaac-g1-sonic dex3

# ...or the same path on the InstinctLab parkour terrain (stairs, gaps, obstacle fields)
./dev.sh isaac-g1-sonic-rough dex3

# terminal 2 — the official SONIC deployment: planner and keyboard on this terminal
./dev.sh sonic-controller
```

Keys in the SONIC terminal: `]` starts control, `T` plays the reference motion, `N` and `P` change
motion, `R` resets, `O` is the emergency stop. `Enter` to switch to planner to teleop.


## If you need


| Command | What it does |
| --- | --- |
| `./dev.sh` | interactive shell in the container, no environment pre-selected |
| `./dev.sh isaac-g1-test-controller dex3` | G1 driven by the deterministic scripted test controller |
| `./dev.sh isaac-g1-sonic-rough dex3` | the SONIC G1 on the InstinctLab parkour terrain, streamed by default |
| `./dev.sh instinct-parkour` | the Project-Instinct G1 parkour checkpoint, with the depth window, streamed by default |
| `./dev.sh instinct-parkour-drive` | keyboard driver for a local X11 Isaac window; use the WebRTC client's keyboard for streamed playback |
| `./dev.sh isaac-demo <demo.py>` | an Isaac Lab demo from `/opt/src/isaaclab/scripts/demos`, streamed over WebRTC |
| `./dev.sh isaac-stream` | the Isaac Sim UI with no scene, streamed over WebRTC |
| `./dev.sh webrtc-client` | Isaac Sim WebRTC client, for watching a streaming host |
| `./dev.sh webrtc-view [emin-1\|raider]` | start the client on that machine and connect it to this host's stream (emin-1 by default) |
| `./dev.sh doctor` | full host + container report, written to `$HUMANOID_DATA_ROOT/diagnostics/` |
| `./dev.sh smoke` | in-container environment, asset and model checks |
| `./dev.sh sync` | re-provisions the environment whose lock or pinned source changed |
| `./dev.sh fetch-models` | downloads the pinned model revisions | |
| `./dev.sh fetch-psi0-ckpt` | downloads the pinned Psi0 SONIC warm start into PSI_HOME | |
| `./dev.sh groot-finetune-smoke [pipeline\|pretrained\|eval]` | optional GR00T N1.7 fine-tuning smoke (two optimizer steps, then open-loop eval) |
| `./dev.sh psi0-smoke` | Psi0 environment smoke: interpreter, torch build, upstream imports, config schema, warm-start checkpoint |
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

