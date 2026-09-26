# G1/Dex3 inference boundary

`examples/dex3/g1_inference.py` is the model-facing part of a future ROS node. It does not import ROS or subscribe to topics. Load it **once** in the inference process, then call `predict` for each synchronized observation:

```python
from examples.dex3.g1_inference import G1Inference

model = G1Inference.load("outputs/dex3/peft-train/checkpoint-<selected>")
model.reset()  # each new episode
commands = model.predict(image, state, task)
```

- `image`: RGB `uint8` HWC, either source `(480, 640, 3)` or training-sized `(192, 256, 3)`. Source-sized frames are resized with PyAV to the training canvas. Use the same RGB convention as the recorded videos.
- `state`: finite measured joint/hand values `(28,)`, ordered by `state_names` in `outputs/dex3/index/manifest.json`. These values and the output commands must use the dataset's joint units.
- `task`: nonempty instruction string. Initial tests should use the task labels in the training index.
- Result: `float32` `(32, 28)` **absolute** commands, ordered by `action_names` in the same manifest. At 30 Hz this chunk spans about 1.07 seconds.

The checkpoint stores the LoRA adapter and saved normalization processors; its adapter config references the full base policy. Keep the base policy, encoders and the selected checkpoint accessible to the inference process. Use an immutable copy of the selected checkpoint because training prunes older checkpoints. `reset()` clears policy and processor state at episode boundaries.

A future ROS callback must synchronize camera/state/task timestamps, map joint names into the manifest order, pass one observation to `predict`, and publish the chunk with its observation timestamp. Calls to this stateful instance must be serialized. The transport also needs stale-observation handling and robot-side command limits. The current repository requires Python 3.12; ROS Foxy commonly runs with Python 3.8, so the Foxy node and this model process can communicate across a process boundary when their Python runtimes differ.

Run a real checkpoint reload and measure warmed inference latency after training frees the GPU. The current sampling settings use four steps and two guidance passes per step. Keep these settings until action quality and latency are measured together.
