# humanoid-lab

Unitree G1 workstation: Isaac Sim and MuJoCo simulation, the official SONIC body
controller, Psi0 and GR00T N1.7 policy evaluation, and the Flux 3 Dex3 stack.
Everything runs in one Docker container. `./dev.sh` is the entry point.

## Requirements

| Item | Value |
| --- | --- |
| OS / arch | Ubuntu 24.04, x86_64 |
| GPU | NVIDIA GPU, driver >= 580.65.06 |
| Docker | Docker Engine, Compose v2 plugin, NVIDIA Container Toolkit |
| Disk | more than 100 GiB free where the repo lives |
| Host tools | `git`, `curl`, `python3` with PyYAML |
| NGC | Isaac Sim EULA accepted, `docker login nvcr.io` performed |
| GUI (optional) | a working X11 display; headless runs need none |
| Hugging Face (optional) | token, only for the gated GR00T backbone |

## Setup

```bash
git clone git@github.com:eminmeydanoglu/humanoid-lab.git
cd humanoid-lab
git submodule update --init third_party/Psi0   # pinned Psi0 checkout

docker login nvcr.io          # the Isaac Sim base image is EULA-gated on NGC
./setup.sh --verify-digests   # resolves the base image digest into versions.lock.yaml
./setup.sh                    # preflight, .env, image build, container start, smoke test
```

`--verify-digests` needs NGC access. A build without a verified digest is refused.

Models are a separate step:

```bash
./dev.sh hf-login                           # once, for gated Hugging Face repositories
./dev.sh fetch-models                       # pinned revisions into $HUMANOID_DATA_ROOT/models
./dev.sh fetch-psi0-ckpt                    # pinned Psi0 SONIC checkpoints
./dev.sh fetch-psi0-ckpt psi0_sonic_dream   # one lock entry by name
```

`nvidia/Cosmos-Reason2-2B`, the GR00T N1.7 backbone, is gated: accept its license on
Hugging Face before `fetch-models`.

## Container and shells

| Command | What it does |
| --- | --- |
| `./dev.sh` | interactive shell in the container, no environment pre-selected |
| `./dev.sh isaac` | shell with the `isaac-sonic` environment |
| `./dev.sh sonic-sim` | shell with the `sonic-sim` environment (MuJoCo + G1) |
| `./dev.sh groot` | shell with the `groot-n17` environment |
| `./dev.sh psi0` | shell with the `psi0` environment |
| `./dev.sh doctor` | full host + container report into `$HUMANOID_DATA_ROOT/diagnostics/` |
| `./dev.sh smoke` | in-container environment, asset and model checks |
| `./dev.sh sync` | re-provisions the environment whose lock or pinned source changed |
| `./dev.sh stop` | stops the container, keeps the data root |
| `./dev.sh rebuild` | rebuilds the image with the same lock, recreates the container |

## Isaac Sim / G1

```bash
./dev.sh isaac-g1 dex3          # Dex3 hands
./dev.sh isaac-g1 inspire-ftp   # Inspire FTP hands
./dev.sh isaac-g1 no_hands      # 29-DoF body only
./dev.sh isaac-stream           # Isaac Sim UI with no scene
./dev.sh isaac-demo <demo.py>   # an Isaac Lab demo from /opt/src/isaaclab/scripts/demos
```

Runs are WebRTC servers by default. No window appears on the host screen.

| Flag | Effect |
| --- | --- |
| `--gui` | opens the local X11 window instead of streaming |
| `--headless` | no UI, no livestream, no X11 requirement |
| `--head-camera-window` | 640x480 head camera in a second GPU viewport |
| `--duration SECONDS` | bounded run |
| `--controller none` | disables the controller the profile declares |
| `--test {passive-fall\|controlled-hold\|controller-hold}` | acceptance mode; 12 s unless `--duration` is given |

Other runners: `./dev.sh isaac-g1-test-controller dex3`,
`./dev.sh isaac-g1-sonic-fixed-base dex3`,
`./dev.sh isaac-g1-direct-reference dex3 --trajectory-reference PATH`,
`./dev.sh isaac-g1-kinematic dex3 --kinematic-reference PATH`.

## Remote viewing (WebRTC)

| Side | What to do |
| --- | --- |
| Server | `./dev.sh isaac-g1 dex3` (streams by default), or `./dev.sh isaac-stream` |
| Address | the host's Tailscale IPv4, advertised automatically; override with `ISAAC_LIVESTREAM_ENDPOINT`, port `49100` |
| Viewer | `./scripts/install-isaac-webrtc-client.sh` once per machine, then `./dev.sh webrtc-client`; enter the address and connect |

One client at a time. The stream ends with the run. The viewer needs no GPU and no
Isaac Sim installation.

## SONIC

```bash
# terminal 1 — Isaac: scene, physics, robot state on DDS, incoming joint commands
./dev.sh isaac-g1-sonic dex3

# terminal 2 — official SONIC deployment: planner and keyboard
./dev.sh sonic-controller
```

SONIC terminal keys: `]` starts control, `T` plays the reference motion, `N`/`P` change
motion, `R` resets, `O` is the emergency stop, `Enter` switches to planner teleop.
Start order does not matter. SONIC waits for Isaac and continues. DDS stays on domain 42
over loopback; the real robot network is never used.

## Psi0 / GR00T evaluation (BlockStacking)

One command starts Isaac BlockStacking, one SONIC controller and the browser UI:

```bash
./dev.sh psi0-isaac-eval \
  --checkpoint-dir <psi-run-directory> \
  --checkpoint-step 40000 \
  --groot-checkpoint-dir <groot-checkpoint-directory> \
  [--psi-dream-checkpoint-dir <psi-dream-run-directory>]
```

Open `http://localhost:8015/` for Start/Stop/Reset, model selection and the head-camera
preview. Watch the simulation over the printed WebRTC endpoint.

| Model option | What it serves |
| --- | --- |
| `Fine-tuned (40k)` | the Psi0 run given on the command line |
| `Base` | that run's training-start checkpoint, materialized on first launch |
| `ψ-Dream (40k)` | the released multi-task SONIC checkpoint, given with `--psi-dream-checkpoint-dir` |
| `GR00T` | the checkpoint given with `--groot-checkpoint-dir` |

A switch stops only the policy backend. Isaac, SONIC, the camera and the UI stay alive.
A failed switch is fail-closed: the previous backend restarts and the session goes to `ERROR`.

Tasks: `--task PickApple` or `--task PickGum` changes the scene and the prompt. The default
is BlockStacking. Extra flags go to `scripts/psi0-isaac-eval.py`.

Defaults that matter:

- Policy clock defaults to Isaac simulation time for both policies.
- GR00T left-hand order defaults to `model-independent`.
- Psi0 neck channels fail closed. Pass `--psi0-neck-policy discard --telemetry-dir PATH`
  only for a checkpoint whose neck padding is verified.
- `--gravity-feedforward` is opt-in. Run it as a separately labelled dynamics contract.
- Use `--policy-clock wall --groot-left-hand-contract compatibility` only to reproduce
  historical runs.

`Ctrl-C` in the terminal, or a dead shared process, ends the whole session.

## Flux Dex3 (suspended PickApple simulation)

Each process runs in its own terminal, from the repository root. Nothing is started for you.

One-time preparation:

```bash
docker compose --env-file .env --profile flux build flux-ros
docker compose --env-file .env up -d dev
docker compose --env-file .env --profile flux up -d flux-ros
./dev.sh flux-ros-build
```

Model environment and checkpoint (both are external artifacts):

```bash
./dev.sh flux-model-env --plan     # inspect resolved paths
./dev.sh flux-model-env            # create data/venvs/flux-model from the pinned lock
./dev.sh flux-checkpoint --source /path/to/trained/checkpoint \
  --base-model-dir /path/to/base-policy-export
./dev.sh flux-model-server --check
```

Terminals:

```bash
# 1 — GPU model server, binds 127.0.0.1:5561
./dev.sh flux-model-server

# 2 — Isaac simulator: scene, fixed-base robot, camera and DDS
./dev.sh flux-isaac configs/profiles/pick-apple-askida.json   # or pick-gum-askida.json

# 3 — ROS 2 launch: camera bridge and Flux Dex3 node
./dev.sh flux-ros ros2 launch flux_sim_camera flux_sim.launch.py \
  model_endpoint:=tcp://127.0.0.1:5561 \
  camera_endpoint:=tcp://127.0.0.1:5555 \
  motor_output_config:=/workspace/humanoid-lab/configs/flux/flux-dex3-sim-motor-config.json \
  enable_motor_commands:=true \
  network_timeout_s:=3.0 \
  max_chunk_age_s:=3.0

# 4 — operator service calls
./dev.sh flux-ros ros2 service call /flux_dex3/get_status flux_dex3_interfaces/srv/GetStatus '{}'
./dev.sh flux-ros ros2 service call /flux_dex3/start_task flux_dex3_interfaces/srv/StartTask '{prompt: "Put the apple into the plate."}'
./dev.sh flux-ros ros2 service call /flux_dex3/stop_task std_srvs/srv/Trigger '{}'
```

- Prompts must match `PROMPTS` in `flux_dex3/flux_dex3/node.py`. The gum prompt is
  `Put the gum into the plate.`
- `accepted: true` only queues the task. Confirm `state: RUNNING` and a nonempty
  `session_id` with `get_status`.
- The simulation launch remaps `/lowcmd` to `/arm_sdk`, matching the profile's
  `rt/arm_sdk` reader. Network timeout and maximum returned-observation age default
  to 3 seconds. The DDS command freshness limit remains 0.25 seconds.
- Realistic apple scene: use `pick-apple-askida-real.json` for Isaac and pass
  `profile:=configs/profiles/pick-apple-askida-real.json` to the ROS launch. Keep
  the profile's `flux_dds` controller enabled so body and hand states are published.
- Stop the task before stopping ROS or Isaac.
- The motor config is simulation-only and must not be copied to a real robot.
- The ROS launch publishes the robot and camera TF tree. For the gum scene, add
  `profile:=configs/profiles/pick-gum-askida.json` to the ROS launch so its TF
  matches the simulator profile.
- Loopback DDS uses automatic participant indices and explicit localhost discovery
  on both the simulator and ROS sides, so later ROS clients can discover state.
- RViz: `./dev.sh flux-ros ros2 launch flux_sim_viz flux_rviz.launch.py`.
- Foxglove Bridge listens on loopback `8765`; expose it with
  `tailscale serve --bg --https=8443 8765`. Alternatively, start a separate read-only bridge
  bound to the host's Tailscale IP:

  ```bash
  ./dev.sh flux-ros ros2 run foxglove_bridge foxglove_bridge --ros-args \
    -r __node:=foxglove_tailscale -p address:="$(tailscale ip -4)" -p port:=8765 \
    --params-file /workspace/humanoid-lab/configs/flux/foxglove-live.yaml
  ```

  Connect a current Foxglove client to `ws://<tailscale-ip>:8765`.
  In the Image panel, select `/camera/color/image_raw/compressed`: the simulation
  launch publishes a latest-only JPEG preview (quality 75, at most 15 FPS), while
  inference keeps its raw RGB topic. The live bridge hides the raw camera and
  limits the shared outgoing queue to 32 messages, dropping oldest queued data
  on overflow. Reconnect after changing the bridge settings to discard old TCP
  buffers. This bounds application buffering; network stalls can still delay TCP.


- Optional recordings: add
  `--tracking-output /outputs/<run>/tracking.parquet --metrics-output /outputs/<run>/summary.json`
  to the simulator command.

## Datasets and pipelines

| Command | What it does |
| --- | --- |
| `./dev.sh sonic-pilot` | builds one pilot episode and its review page |
| `./dev.sh sonic-convert` | converts a source episode to the canonical SONIC format |
| `./dev.sh sonic-convert-unitree` | bulk Unitree Dex3 production conversion |
| `./dev.sh sonic-encode` | encodes an episode with the pinned SONIC encoder |
| `./dev.sh sonic-review` / `sonic-review-serve` | builds and serves the review page |
| `./dev.sh sonic-verify` | independent arithmetic check of a converted episode |
| `./dev.sh sonic-dataset-validate` | validates a converted dataset |
| `./dev.sh psi0-dex3-dataset-split` | freezes the episode-level train/validation split |
| `./dev.sh psi0-dex3-dataset-convert` | Unitree Dex3 + SONIC to Psi0 30 Hz LeRobot pack |
| `./dev.sh psi0-dex3-dataset-validate` | re-derives the pack and opens it with the pinned loader |
| `./dev.sh psi0-dex3-check` | dataset and warm-start preflight |
| `./dev.sh psi0-dex3-run` | training gate run |
| `./dev.sh fetch-groot-demo-data` | downloads the GR00T demo data |

## Tests

| Command | What it does |
| --- | --- |
| `./dev.sh sonic-tests` | SONIC unit tests in both conversion environments |
| `./dev.sh psi0-tests` | Psi0 contract tests with the `psi0` interpreter |
| `./dev.sh psi0-smoke` | Psi0 environment smoke: interpreter, torch, imports, config, warm start |
| `./dev.sh groot-finetune-smoke [pipeline\|pretrained\|eval]` | optional GR00T N1.7 fine-tuning smoke (two optimizer steps, then open-loop eval) |

## Data root

`HUMANOID_DATA_ROOT` in `.env` defaults to `./data` and holds everything persistent:
datasets, checkpoints, models, caches, venvs and run outputs. It is bind-mounted into the
container as `/data`, `/cache`, `/outputs` and `/opt/venvs`. The repository itself is
mounted at `/workspace/humanoid-lab`.

## Repository layout

| Path | Contents |
| --- | --- |
| `setup.sh`, `dev.sh`, `doctor.sh` | host entry points: provisioning, container lifecycle and runs, diagnostics |
| `compose.yaml`, `Dockerfile`, `containers/` | image and container definition, entrypoint, environment selectors, venv provisioning |
| `configs/profiles/` | G1 run profiles: robot asset, hands, camera, controller, support band |
| `configs/flux/` | Flux Dex3 motor config and its provenance note |
| `src/humanoid_lab/` | simulator service, profile and command contracts, controller bridge, dataset pipelines |
| `scripts/` | run scripts, model fetching, lock rendering, checks |
| `locks/` | frozen `uv` locks for the Python environments |
| `third_party/Psi0/` | pinned Psi0 checkout (git submodule) |
| `third_party/flux/` | pinned Flux training and inference sources |
| `tests/` | `unittest` modules and shell checks |
| `versions.lock.yaml` | single source of every version pin |
