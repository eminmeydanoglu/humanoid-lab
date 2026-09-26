# Unitree G1 Dex3 data connection

The source collection is read from
`/home/aksoy-lab/code/humanoid-lab-main/data/datasets/first_tur_ham/unitree-g1-dex3`.
It is read-only; preparation writes only under `outputs/dex3/index` in this checkout.
The training config is `configs/dex3/train.json`. Run commands from the repository root.

```sh
uv sync --locked --extra data --extra encoders
uv run python examples/dex3/prepare.py \
  --source-root /home/aksoy-lab/code/humanoid-lab-main/data/datasets/first_tur_ham/unitree-g1-dex3 \
  --output-dir outputs/dex3/index
```

The index contains a combined `manifest.json`, `rows.f32.npy`, `statistics.json`,
`audit.json` and intermediate per-dataset rows under `parts/`. `source_root` in the
training config points to the collection, and `index_dir` points to the combined index.
All video references resolve to the original files. Each window reads only
`observation.images.cam_left_high` as `images.head`: uint8 RGB frames resized to
256 × 256, with one observed frame and 32 following frames. The state is
`observation.state` `(28,)` and the target is `action` `(32, 28)`, both in the
recorded channel order. The model uses a single 256 × 256 camera canvas.

`action_parameterization: absolute` means the 28 recorded commands are the
targets directly; the trainer does not subtract consecutive commands. The
index computes q01/q99 channel bounds for actions and states from **training
episodes only** and saves them for training and inference. The collection is
held out by episode: the last approximately 10% of eligible episodes *in each
source* form the validation split. Review the split if recording sessions
were grouped chronologically.

The 13 source folders are checked for the same 28 joint names/order, 30 Hz,
head camera and valid rows. `G1_Dex3_GraspSquare_Dataset` is excluded after
comparing its entire numeric recording and every selected head-camera video
file with `G1_Dex3_BlockStacking_Dataset`; they are byte-identical. The
preparation refuses to exclude it if those files ever differ. The
`ToastedBread` source metadata misidentifies the data file or global row offset
for its last 22 episodes. Preparation validates those episodes against actual
Parquet rows and repairs only the generated index; `audit.json` lists every
changed reference.

Task text is taken from each source's `meta/tasks.parquet`. Seven Pick* datasets
have conflicting stale episode labels mentioning a red cup; the discrepancies
are recorded per episode in `audit.json`. Inspect the task labels against the
recordings before treating them as verified demonstration instructions. The
source datasets are left untouched. Camera and action alignment should also
be spot-checked against recorded episodes before training.

`configs/dex3/train.json` supplies a complete training configuration with the
correct data fields. Other optimization and schedule values currently follow
the DROID config as a starting template; review them and the available GPU
memory before launching a training job.
