# flux

Two halves of the FLUX 3 Action repository (`git@github.com:black-forest-labs/flux-action.git`,
main @ `afd2d12` at the time this snapshot was taken), placed here so the sim and robot
integration in this repository runs the same inference code that ships to the robot.

| Half | Contents | Depends on |
|---|---|---|
| `flux-training/` | the `flux_action` library (`src/`), training configs, data preparation and training/export examples, research docs, library tests, `pyproject.toml` + `uv.lock`, CI workflow | — |
| `flux-inference/` | the ROS 2 packages (`ros2/flux_dex3`, `ros2/flux_dex3_interfaces`), the GPU-side serving entry points (`examples/dex3/g1_inference.py`, `examples/dex3/zmq_server.py`), the Dex3 deployment tests and docs | `flux-training` (imports `flux_action`) |

Dependency rule: **inference imports the library, the library never imports inference.** The only
edge is `flux-inference` reading `flux_action` and `flux_dex3` (`ros2/`); `zmq_server.py` imports
`flux_dex3.protocol` so both ends of the wire share one protocol definition.

These are plain directories, not git submodules (contrast `third_party/Psi0`, which is a submodule of
an upstream repository). Files keep their upstream relative paths, so `diff -r` against the source
checkout stays meaningful and re-syncing is mechanical.

## What is deliberately not here

`outputs/` (45 GB of prepared indices, checkpoints and runs) and `.venv/` stay in the source
checkout `/home/aksoy-lab/code/flux-training/flux-action`, which is also where the git history lives.
Training runs from there. For serving, the ZMQ server refuses writable or symlinked checkpoints, so
copy the checkpoint to a scratch directory and drop write permission first (see
`flux-inference/README.md`).

`flux-inference/outputs` is a relative symlink to that artifact tree: the deployment test
`test_manifest_matches_unitree_motor_order` compares the node's 28-channel order against the trained
index manifest (`outputs/dex3/index/manifest.json`). If the artifacts move, repoint that one symlink;
moving them under `data/` in this repository is the alternative.

## Re-syncing from the source checkout

```sh
FLUX=/home/aksoy-lab/code/flux-training/flux-action
DEST=/home/aksoy-lab/code/humanoid-lab-main/third_party/flux

# flux-training: everything except the inference-only files
rsync -a \
  --exclude='.git/' --exclude='.venv/' --exclude='outputs/' \
  --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='__pycache__/' \
  --exclude='ros2/' \
  --exclude='examples/dex3/g1_inference.py' --exclude='examples/dex3/zmq_server.py' \
  --exclude='tests/test_dex3_command_output.py' --exclude='tests/test_dex3_end_to_end.py' \
  --exclude='tests/test_dex3_network.py' --exclude='tests/test_dex3_node_pause.py' \
  --exclude='tests/test_dex3_reconnect.py' --exclude='tests/test_dex3_robot_bridge.py' \
  --exclude='tests/test_dex3_zmq.py' --exclude='tests/test_g1_inference.py' \
  --exclude='docs/dex3_inference.md' --exclude='docs/dex3_ros_bridge.md' --exclude='docs/dex3_ros_inference_todo.md' \
  "$FLUX/" "$DEST/flux-training/"

# flux-inference: the deployment surface
rsync -a "$FLUX/ros2/" "$DEST/flux-inference/ros2/"
cp "$FLUX/examples/__init__.py" "$DEST/flux-inference/examples/"
cp "$FLUX/examples/dex3/g1_inference.py" "$FLUX/examples/dex3/zmq_server.py" "$DEST/flux-inference/examples/dex3/"
cp $FLUX/tests/test_dex3_{command_output,end_to_end,network,node_pause,reconnect,robot_bridge,zmq}.py \
   "$FLUX/tests/test_g1_inference.py" "$DEST/flux-inference/tests/"
cp "$FLUX/docs/dex3_inference.md" "$FLUX/docs/dex3_ros_bridge.md" "$FLUX/docs/dex3_ros_inference_todo.md" "$DEST/flux-inference/docs/"
```

If the split is committed upstream, these two directories can become submodules with per-path sparse
checkout; that is the only step this snapshot cannot do on its own.
