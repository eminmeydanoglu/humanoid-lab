# humanoid-lab



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

Docker group membership needs a fresh login session: `sudo usermod -aG docker $USER`, then log out
and back in.

## Setup from scratch

```bash
git clone git@github.com:eminmeydanoglu/humanoid-lab.git
cd humanoid-lab

docker login nvcr.io        # the Isaac Sim base image is EULA-gated on NGC
./setup.sh --verify-digests # resolves the base image digest from NGC into versions.lock.yaml
./setup.sh                  # preflight, .env, image build, container start, smoke test
```

`setup.sh --verify-digests` needs NGC access. Without a verified digest in the lock, `setup.sh`
refuses to build, so this is the one step that cannot be skipped on a fresh machine.

`setup.sh` itself is idempotent and does the following:

1. preflight: OS/arch, free disk, NVIDIA driver against the tested baseline, Docker,
   Compose v2, NVIDIA Container Toolkit, daemon reachability, host `input` group;
2. renders `versions.lock.yaml` into `.generated/versions.env`;
3. creates `.env` from `.env.example` on the first run and creates the data root directories.
   An existing `.env` is never overwritten;
4. builds the `dev` image and starts the `humanoid-lab-dev` container;
5. runs the short smoke test. A failing smoke test does not abort setup — it is reported, and
   `./dev.sh smoke` can be re-run afterwards.

The run is logged to `setup.log` and `hersey.log`.

Models are a separate, revision-pinned step:

```bash
./dev.sh hf-login      # once, if you need gated Hugging Face repositories; the token stays in the host HF cache
./dev.sh fetch-models  # pinned revisions into $HUMANOID_DATA_ROOT/models, plus MODEL_PROVENANCE.json with sha256 per file
```

`nvidia/Cosmos-Reason2-2B`, the GR00T N1.7 backbone, is gated: accept its license on Hugging Face
before `fetch-models`, otherwise the GR00T checks stay `BLOCKED`.

## Everyday commands

All commands run from the repo root. They start the container if it is not running.

| Command | What it does |
| --- | --- |
| `./dev.sh` | interactive shell in the container, no environment pre-selected |
| `./dev.sh isaac` | shell with the `isaac-sonic` environment active |
| `./dev.sh sonic-sim` | shell with the `sonic-sim` environment (MuJoCo + G1) |
| `./dev.sh groot` | shell with the `groot-n17` environment |
| `./dev.sh isaac-g1 dex3\inspire-ftp\no_hands` | G1 on Isaac Sim, see below |
| `./dev.sh isaac-g1-sonic dex3\inspire-ftp\no_hands` | G1 on Isaac Sim, publishing robot state to SONIC over DDS |
| `./dev.sh sonic-controller` | the official SONIC controller, run in a second terminal |
| `./dev.sh isaac-g1-test-controller dex3` | G1 driven by the deterministic scripted test controller |
| `./dev.sh isaac-demo <demo.py>` | an Isaac Lab demo from `/opt/src/isaaclab/scripts/demos` |
| `./dev.sh doctor` | full host + container report, written to `$HUMANOID_DATA_ROOT/diagnostics/` |
| `./dev.sh smoke` | in-container environment, asset and model checks |
| `./dev.sh sync` | re-provisions the environment whose lock or pinned source changed |
| `./dev.sh fetch-models` | downloads the pinned model revisions | |
| `./dev.sh groot-finetune-smoke [pipeline\|pretrained\|eval]` | optional GR00T N1.7 fine-tuning smoke (two optimizer steps, then open-loop eval) |
| `./dev.sh hf-login` | Hugging Face login for gated repositories |
| `./dev.sh stop` | stops the container, keeps the data root |
| `./dev.sh rebuild` | rebuilds the image with the same lock, recreates the container |

Inside a container shell, `use-isaac-sonic`, `use-sonic-sim`, `use-groot` and `use-none` switch
environments, and `show-env` prints the active one and its Python.


## Repository layout

| Path | Contents |
| --- | --- |
| `setup.sh`, `dev.sh`, `doctor.sh` | host entry points: provisioning, container lifecycle and runs, diagnostics |
| `compose.yaml`, `Dockerfile`, `containers/` | image and container definition, entrypoint, environment selectors, venv provisioning |
| `configs/profiles/` | G1 run profiles: robot asset, hands, camera, controller, support band |
| `src/humanoid_lab/` | simulator service, profile and command contracts, controller bridge |
| `scripts/` | Isaac G1 runner, model fetching, lock rendering, checks |
| `locks/` | frozen `uv` locks for the three Python environments |
| `tests/` | `unittest` modules and shell checks |
| `docs/` | SONIC-Isaac link notes with the measured acceptance results |
| `versions.lock.yaml` | single source of every version pin |

## G1 on Isaac Sim

```bash
./dev.sh isaac-g1 dex3          # Dex3 hands
./dev.sh isaac-g1 inspire-ftp   # Inspire FTP hands
./dev.sh isaac-g1 no_hands      # 29-DoF body only
```

The profile selects the robot asset, the hands and the head camera. The Dex3 variant uses the pinned
SONIC G1 USD (14 hand DoF), the Inspire variant the Isaac Lab asset (24 hand DoF), and `no_hands` is
the 29-DoF body alone. Hand joints are passive unless a controller drives them: on the SONIC path the
Dex3 profile applies the hand commands it receives over DDS, while the Inspire variant deliberately
leaves the hands passive.

- `dex3` needs the Dex3 USD materialized into the data root. `./dev.sh sync` does that and verifies
  it by content hash; without it the run fails and the smoke test reports it.
- The GUI opens the main RTX viewport and a `G1 Simulator` panel. `Reset Robot` restores the initial
  pose and leaves the timeline paused. Stop the run with `Ctrl-C` or by closing the window.
- Normal runs have no time limit; without an explicit duration they stay alive until stopped.

Flags, passed after the profile:

| Flag | Effect |
| --- | --- |
| `--headless` | no GUI and no X11 requirement |
| `--head-camera-window` | opens the 640x480 head camera in a second GPU viewport; costs render throughput |
| `--duration SECONDS` | bounded run |
| `--device {cpu,cuda}` | overrides the profile device; profiles default to CPU |
| `--controller none` | disables the controller a profile declares, falling back to passive |
| `--test {passive-fall\|controlled-hold\|controller-hold}` | one of the acceptance modes below |

Physics runs at 200 Hz (`physics_dt` 0.005 s) on four PhysX threads, with the main viewport rendered
every eighth physics step. Only one Isaac G1 run can be active at a time; a second one exits with an
error instead of competing for the GPU.

## Isaac and SONIC in two terminals

```bash
# terminal 1 — Isaac Sim: scene, physics loop, robot state on DDS, incoming joint commands applied
./dev.sh isaac-g1-sonic dex3

# terminal 2 — the official SONIC deployment: planner and keyboard on this terminal
./dev.sh sonic-controller
```

Keys in the SONIC terminal: `]` starts control, `T` plays the reference motion, `N` and `P` change
motion, `R` resets, `O` is the emergency stop.

The order does not matter: if SONIC starts first it waits with `LowState is not available` until
Isaac comes up. DDS is loopback-only on domain `42`, so nothing reaches a real robot network. The
robot hangs from a pelvis band until the first valid controller command, at most `20 s`, and the
release is logged as `isaac_g1_support_released`. If commands stop for longer than the 0.25 s
command TTL, the robot goes passive and falls. Running `./dev.sh sonic-controller` again replaces
the previous Isaac-targeted controller instead of leaving two of them running.

`Isaac` publishes `LowState` and applies `BodyJointCommand`; it contains no policy. `docs/sonic-isaac.md`
documents the link in detail, including the measured acceptance results and the known limits.

## 
## Data root, mounts and environments

`HUMANOID_DATA_ROOT` in `.env` defaults to `./data` and holds everything persistent:
`datasets`, `checkpoints`, `models`, `hf-cache`, `uv-cache`, `venvs`, `isaac-cache`, `outputs`,
`diagnostics`, `rosbags`, `runtime`. It is bind-mounted into the container as `/data`, `/cache`,
`/outputs` and `/opt/venvs`. The repository itself is mounted at `/workspace/humanoid-lab`.

The Python environments (`isaac-sonic`, `sonic-sim`, `groot-n17`) live in the data root, not in the
image. Each is fingerprinted by its lock file plus the pinned upstream commits, and re-provisioned
only when that fingerprint changes, so day-to-day code edits never rebuild an environment. A new
dependency must be added to the lock and applied with `./dev.sh sync`.

> `/opt/venvs` and the Isaac cache directories are writable host bind mounts. Installing packages
> manually inside the container persists on the host and breaks the lock/fingerprint expectation;
> such an environment is not repaired automatically. Use `./dev.sh sync` after a lock or source
> change and `./dev.sh rebuild` after an image change, then re-check with `./dev.sh doctor` and
> `./dev.sh smoke`.


