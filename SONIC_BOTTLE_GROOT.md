# SONIC Isaac Lab bottle scene + GR00T N1.7

This worktree adds the Unitree small-warehouse and packing-table assets to the
existing SONIC G1/Dex3 simulator. The bottle is a dynamic rigid cylinder with
the dimensions, mass, and friction used by Unitree's pick-place scene. The
original SONIC and Unitree profiles remain unchanged.

Run every command below from:

```bash
cd /home/aksoy-msi/code/unitree_sim_isaaclab
```

## 1. Start Isaac Lab as a full 3D WebRTC stream

```bash
./dev.sh isaac-g1-sonic-bottle --camera-zmq-port 5555
```

Isaac Sim now runs its complete 3D UI in WebRTC livestream mode on port
`49100`, following the implementation in the main checkout. The launcher
prints the Tailscale address. Install and start the official client:

```bash
./scripts/install-isaac-webrtc-client.sh
DISPLAY=:0 ./dev.sh webrtc-client
```

Enter the printed host address, for example `100.126.18.76`, and port `49100`
in the client, then connect. The client shows the full warehouse, table,
robot, and bottle viewport. Port `5555` still publishes the 640x480 head RGB
frame as SONIC's `ego_view` for GR00T.

Use `--gui` for a direct X11 Isaac window or `--headless` for validation runs.

For a passive scene check without a desktop window:

```bash
./dev.sh isaac-g1-sonic-bottle --headless --duration 5 --controller none \
  --record-video /outputs/sonic-bottle-check.mp4
```

## 2. Open the SONIC camera viewer (second laptop terminal)

```bash
cd /home/aksoy-msi/code/unitree_sim_isaaclab
./dev.sh sonic-camera-viewer
```

The viewer subscribes to the same ZMQ stream used by GR00T. Press `Q` to close
it; press `R` to start or stop MP4 recording.

## 3. Start the GR00T policy server (third terminal)

```bash
cd /home/aksoy-msi/code/unitree_sim_isaaclab
./dev.sh groot
```

Inside the container shell:

```bash
cd /opt/src/isaac-groot
python -m gr00t.eval.run_gr00t_server \
  --model-path /data/models/cloudwalk-gr00t-n17-g1-grab-bottle-rh-371ep-v10-finetune/checkpoint-30000 \
  --embodiment-tag UNITREE_G1_SONIC \
  --device cuda:0 --host 0.0.0.0 --port 5550
```

## 4. Start SONIC standing/control terminal

```bash
cd /home/aksoy-msi/code/unitree_sim_isaaclab
./dev.sh sonic-stand
```

This starts the SONIC deployment with `zmq_manager`. It brings the robot to
the standing pose and keeps the simulator-facing DDS loop alive. It listens
for VLA actions on `tcp://localhost:5556`. Stop it with `Ctrl+C`, or from a
second shell use:

```bash
./dev.sh sonic-stand-stop
```

## 5. Start VLA inference in its own terminal

```bash
cd /home/aksoy-msi/code/unitree_sim_isaaclab
./dev.sh sonic-vla \
  --host localhost --port 5550 \
  --camera-host localhost --camera-port 5555 \
  --state-zmq-host localhost --state-zmq-port 5557 \
  --action-zmq-host 0.0.0.0 --action-zmq-port 5556 \
  --embodiment-tag unitree_g1_sonic \
  --prompt "grab the bottle"
```

`sonic-vla` uses the GR00T environment and forwards its latent actions to the
already-running `sonic-stand` process. The PolicyServer must already be
running on port 5550. Stop VLA independently with `Ctrl+C`; stopping VLA leaves
SONIC standing in the first terminal. If the terminal disconnects, use:

```bash
./dev.sh sonic-vla-stop
```

## Stop everything

Use `Ctrl+C` in each foreground terminal. To remove any worktree-owned
container processes afterward:

```bash
./dev.sh stop
```

## Verified on this machine

- Scene and bottle profile: 41 unit/source-invariant tests pass.
- Isaac head camera: SONIC client receives `ego_view` as 480x640 RGB on port 5555.
- GR00T checkpoint: both shards load on the RTX 5090 and `PolicyClient.ping()` returns `True` on port 5550.
- Existing SONIC, GR00T, MuJoCo, and Unitree environments were not modified.
