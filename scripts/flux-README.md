# Flux Dex3 in the suspended PickApple simulator

Run each process in its own terminal from the repository root. The simulator owns the scene and its camera/DDS streams; the model server owns the GPU model; ROS 2 owns the camera bridge and the Dex3 controller. You decide when to start and stop each process and when to call a task service. No session manager starts other processes on your behalf.

## One-time preparation

Run `./setup.sh` first if `.env` does not exist. Build the ROS container and the workspace after installing or changing ROS sources:

```bash
docker compose --env-file .env --profile flux build flux-ros
docker compose --env-file .env up -d dev
docker compose --env-file .env --profile flux up -d flux-ros
./dev.sh flux-ros-build
```

The model environment and trained weights are separate from the repository. On a new installation, inspect `./dev.sh flux-model-env --plan`, create the environment with `./dev.sh flux-model-env`, and prepare the trained adapter with:

```bash
./dev.sh flux-checkpoint --source /path/to/trained/checkpoint \
  --base-model-dir /path/to/base-policy-export
./dev.sh flux-model-server --check
```

The adapter defaults to `data/models/flux-dex3/checkpoint-2500`; the model server uses the persistent `data/venvs/flux-model` interpreter if present. `./dev.sh flux-model-server --plan` shows the resolved paths without starting anything. Neither the adapter nor its base model is downloaded automatically.

## Start the processes

**Terminal 1 — GPU model server** (leave it running; its logs stay here):

```bash
./dev.sh flux-model-server
```

It binds `127.0.0.1:5561` and checks the interpreter and prepared checkpoint before loading. The frozen Qwen text encoder stays in host RAM throughout loading and serving; the server encodes the node's supported prompts and the empty classifier-free-guidance prompt before reporting READY. This saves roughly 8 GB of resident GPU weights at the cost of a longer startup and higher host RAM use. New prompts outside the node's vocabulary require CPU text encoding on their first request. The DiT and video encoder remain on GPU, with the checkpoint's bf16 precision, sampler, guidance and action normalization unchanged. Set `FLUX_MERGE_ADAPTER=1` to merge LoRA weights in memory for faster sampling; this can increase peak load memory and slightly change bf16 rounding. Ctrl-C stops this server. Start a second copy only after the first one exits.

**Terminal 2 — Isaac simulator** (scene, fixed-base robot, camera and DDS):

```bash
./dev.sh flux-isaac configs/profiles/pick-apple-askida.json 
```

This profile inherits the apple-and-plate scene, fixes the G1 pelvis in space facing the table, places both hands above the near edge of the tabletop, and selects the simulator's `flux_dds` actuator/feedback adapter. While fresh arm commands arrive, that adapter holds the three waist joints at their standing angles with the pinned deployment gains so the torso stays aligned with the fixed pelvis. [First spawn head-camera frame](flux-head-camera-spawn.jpg) shows the apple, plate, and both hands. The simulator publishes the SONIC-format head camera on port 5555 for the ROS camera bridge. `--gui` opens the local Isaac window; omit it for the default WebRTC view or use `--headless` when a window is unnecessary. A run without `--duration` stays up until you stop it with Ctrl-C. Only one Isaac G1 run can own the simulator lock at a time. For the gum scene use `configs/profiles/pick-gum-askida.json` here and the corresponding gum prompt below.

**Terminal 3 — ROS 2 launch** (camera bridge and Flux Dex3 node):

```bash
./dev.sh flux-ros ros2 launch flux_sim_camera flux_sim.launch.py \
  model_endpoint:=tcp://127.0.0.1:5561 \
  camera_endpoint:=tcp://127.0.0.1:5555 \
  motor_output_config:=/workspace/humanoid-lab/configs/flux/flux-dex3-sim-motor-config.json \
  enable_motor_commands:=true
```

The launch logs stay in this terminal. Its camera bridge translates the simulator stream into `/camera/color/image_raw`; the Dex3 node reads this image and the simulated joint-state topics and publishes motor commands. The supplied motor config is **simulation-only**. Publishing requires both `enable_motor_commands:=true` and a valid motor config. To inspect observations without motor commands, omit the last two launch arguments; the node will create no command publishers. Stop the launch with Ctrl-C.

The simulator and ROS container use ROS domain 42 and loopback-only CycloneDDS. The ROS container uses host networking, so `127.0.0.1` reaches the model server on the host and the simulator's camera port.

## Call the ROS services yourself

**Terminal 4 — operator commands.** Check readiness before starting a task:

```bash
./dev.sh flux-ros ros2 service call /flux_dex3/get_status flux_dex3_interfaces/srv/GetStatus '{}'

./dev.sh flux-ros ros2 service call /flux_dex3/start_task flux_dex3_interfaces/srv/StartTask '{prompt: "Put the apple into the plate."}'

```

The model must report READY and camera/joint observations must be fresh. An `accepted: true` start response confirms only that the request was accepted; confirm `state: RUNNING` and a nonempty `session_id` with `get_status`. The prompt must exactly match an entry in `flux_dex3/flux_dex3/node.py`'s `PROMPTS`. For the gum profile use `Put the gum into the plate.`. A task has no automatic time limit; stop it explicitly:

```bash
./dev.sh flux-ros ros2 service call /flux_dex3/stop_task \
  std_srvs/srv/Trigger '{}'
```

Check `get_status` again for the resulting state and reason. Stop the task before stopping ROS or Isaac; then use Ctrl-C in each process's own terminal. A rejected start or a task that stops unexpectedly is explained by the service's `reason` field and the ROS launch terminal. The model and simulator terminals show their respective failures separately.

## Optional observation and recordings

For visualization, open RViz in another terminal with `./dev.sh flux-ros ros2 launch flux_sim_viz flux_rviz.launch.py`. RViz reads the simulated joint states and head-camera topic. The WebRTC client can be started separately with `./dev.sh webrtc-client` when installed.

Foxglove Bridge starts with the `flux-ros` container and listens on host loopback port 8765. It uses the same ROS domain and CycloneDDS configuration as the other ROS nodes. To expose a secure connection on a dedicated Tailscale HTTPS port without changing existing Serve routes, run `tailscale serve --bg --https=8443 8765` on the host. In Foxglove Web (`https://app.foxglove.dev`) or Foxglove Desktop, add a Foxglove WebSocket connection to `wss://<this-machine-tailnet-DNS>:8443/`. Foxglove account permissions may affect direct WebSocket access. To inspect the bridge, run `docker compose --env-file .env --profile flux logs flux-ros` and `tailscale serve status`.

Add `--tracking-output /outputs/<run>/tracking.parquet --metrics-output /outputs/<run>/summary.json` to the simulator command if you need recorded motion and a summary; these files are written when Isaac ends normally. Choose a distinct output directory for each run. These recordings and visualization have no effect on service calls or process ownership.
