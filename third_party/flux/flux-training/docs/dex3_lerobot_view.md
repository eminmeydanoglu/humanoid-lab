# G1 Dex3 FLUX3 dataset view

`flux_action.data.lerobot.dex3_view.Dex3Flux3View` is a map-style PyTorch dataset for the exported G1 policy's LeRobot FLUX3 `history` processor. Use `split="train"` or `split="val"` on the **same** combined Dex3 index. It keeps the existing episode-level split and repaired ToastedBread row/video references, reads numeric values from the existing memory-mapped `rows.f32.npy` cache of the original Parquet files, and seeks RGB frames in the original `cam_left_high` MP4 files. It writes no media or normalization statistics. The source and index paths must remain available at runtime.

Each item contains `observation.images.cam_left_high` as uint8 RGB `(33, 3, 192, 256)`, measured `observation.state` `(1, 28)`, absolute `action` `(33, 28)` (the previous command followed by 32 targets), and task text from the validated manifest. LeRobot's saved FLUX3 preprocessor converts the actions to `(32, 28)` normalized targets. The source camera is 480×640, and the decode resize preserves its 4:3 geometry at 192×256. No joint reordering or unit conversion is applied.

The standard `lerobot-train` dataset factory currently accepts a LeRobot repository ID and constructs its own `LeRobotDataset`; it does not discover this custom view from `--dataset.repo_id`. A future training entry point must supply these train/val objects to the trainer's dataloader construction and use the exported policy's saved processors. The validation script exercises the same `DataLoader` batch format and LeRobot FLUX3 processor; it does not load or execute the model.

From the `flux-action` directory, with LeRobot installed and available on `PYTHONPATH`:

```sh
PYTHONPATH=src:$LEROBOT_CHECKOUT python examples/dex3/verify_lerobot_view.py \
  --source-root /home/aksoy-lab/code/humanoid-lab-main/data/datasets/first_tur_ham/unitree-g1-dex3 \
  --index-dir outputs/dex3/index \
  --policy outputs/dex3/g1-base-policy --samples 100
```

This verifies both train and validation (100 random windows in each), original task metadata, processor quantiles, and split isolation. The dataset does not calculate new statistics; use the `policy_preprocessor*.safetensors` saved in `outputs/dex3/g1-base-policy`.
