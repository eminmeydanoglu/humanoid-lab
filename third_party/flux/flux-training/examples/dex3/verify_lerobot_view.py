"""Check the Dex3 view with LeRobot's saved FLUX3 preprocessing; no policy is loaded."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.flux3.configuration_flux3 import Flux3Config
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from flux_action.data.lerobot.dex3_view import CAMERA, Dex3Flux3View


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 100:
        raise ValueError("at least 100 samples per split are required")
    manifest = json.loads((args.index_dir / "manifest.json").read_text())
    audit = json.loads((args.index_dir / "audit.json").read_text())
    source_names = set(audit["sources"])
    assert len(source_names) == 12 and "G1_Dex3_GraspSquare_Dataset" not in source_names
    assert {e["episode_id"].split("/")[0] for e in manifest["episodes"]} == source_names
    assert manifest["state_names"] == manifest["action_names"] and len(manifest["state_names"]) == 28
    assert manifest["fps"] == 30 and manifest["camera_hw"] == {"head": [480, 640]}
    assert manifest["counts"]["eligible"] == 2851
    assert manifest["counts"]["train_episodes"] == 2567 and manifest["counts"]["val_episodes"] == 284
    train_ids = {e["episode_id"] for e in manifest["episodes"] if e["split"] == "train"}
    val_ids = {e["episode_id"] for e in manifest["episodes"] if e["split"] == "val"}
    assert not train_ids & val_ids and len(train_ids) == 2567 and len(val_ids) == 284
    task_text = {}
    for name in source_names:
        data = pq.read_table(args.source_root / name / "meta/tasks.parquet").to_pydict()
        assert len(data["task_index"]) == 1
        key = "task" if "task" in data else "__index_level_0__"
        task_text[name] = str(data[key][0]).strip()
    for e in manifest["episodes"]:
        assert e["caption"] == task_text[e["episode_id"].split("/")[0]]
        assert (args.source_root / e["data_file"]).is_file()
        assert (args.source_root / e["videos"]["head"]["file"]).is_file()
    repairs = audit["row_reference_repairs"]
    assert len(repairs) == 22 and all(r["dataset"] == "G1_Dex3_ToastedBread_Dataset" for r in repairs)
    by_source_episode = {
        (e["episode_id"].split("/")[0], e["source_episode_index"]): e for e in manifest["episodes"]
    }
    for repair in repairs:
        episode = by_source_episode[(repair["dataset"], repair["episode"])]
        assert episode["data_file"] == f"{repair['dataset']}/{repair['actual_file']}"
        assert episode["from_index"] == (
            audit["row_offsets"][repair["dataset"]] + repair["actual_from_index"]
        )
    assert audit["task_source"] == "meta/tasks.parquet"
    print("index: 12 sources, 2851 episodes, 2567 train / 284 val, task metadata agrees", flush=True)
    print("ToastedBread repaired row references:", len(repairs), flush=True)

    config = Flux3Config.from_pretrained(args.policy)
    assert config.conditioning == "history" and config.action_representation == "absolute"
    assert config.camera_order == [CAMERA] and list(config.canvas_hw) == [192, 256]
    pre, _ = make_pre_post_processors(config, pretrained_path=args.policy)
    saved = load_file(
        args.policy / "policy_preprocessor_step_3_flux3_observation_history_normalizer.safetensors"
    )
    targets = load_file(args.policy / "policy_preprocessor_step_4_flux3_action_target_normalizer.safetensors")
    assert set(saved) == {"state.q01", "state.q99", "action.q01", "action.q99"}
    assert set(targets) == {"action.q01", "action.q99"}
    for key in targets:
        assert torch.equal(targets[key], saved[key])

    observed = defaultdict(set)
    for split, seed in (("train", 73001), ("val", 73002)):
        view = Dex3Flux3View(args.source_root, args.index_dir, split=split)
        indices = torch.randperm(len(view), generator=torch.Generator().manual_seed(seed))[: args.samples]
        loader = DataLoader(Subset(view, indices.tolist()), batch_size=1, num_workers=0)
        for number, batch in enumerate(loader):
            raw_state = batch["observation.state"].clone()
            raw_action = batch["action"].clone()
            assert batch[CAMERA].shape == (1, 33, 3, 192, 256)
            assert batch[CAMERA].dtype == torch.uint8
            assert raw_state.shape == (1, 1, 28) and raw_action.shape == (1, 33, 28)
            assert torch.isfinite(raw_state).all() and torch.isfinite(raw_action).all()
            ep = int(batch["episode_index"].item())
            episode = manifest["episodes"][ep]
            assert episode["split"] == split
            assert batch["task"][0] == task_text[episode["episode_id"].split("/")[0]]
            observed[split].add(ep)
            image = batch[CAMERA][0, 0].permute(1, 2, 0).numpy()
            assert image.shape[0] * 4 == image.shape[1] * 3
            batch[CAMERA] = batch[CAMERA].float() / 255.0
            processed = pre(batch)
            image_out = processed[CAMERA]
            state_out = processed["observation.state"]
            action_out = processed["action"]
            assert image_out.shape == (1, 33, 3, 192, 256)
            assert state_out.shape == (1, 1, 28) and action_out.shape == (1, 32, 28)
            assert torch.isfinite(state_out).all() and torch.isfinite(action_out).all()
            lo_s, hi_s = saved["state.q01"], saved["state.q99"]
            lo_a, hi_a = targets["action.q01"], targets["action.q99"]
            expected_s = (2 * (raw_state - lo_s) / torch.where(hi_s - lo_s > 1e-6, hi_s - lo_s, 1) - 1).clamp(
                -6, 6
            )
            expected_a = (
                2 * (raw_action[:, 1:] - lo_a) / torch.where(hi_a - lo_a > 1e-6, hi_a - lo_a, 1) - 1
            ).clamp(-6, 6)
            torch.testing.assert_close(state_out, expected_s)
            torch.testing.assert_close(action_out, expected_a)
            if number == 0 and split == "train":
                for label, values in (
                    ("raw state", raw_state[0, 0]),
                    ("normalized state", state_out[0, 0]),
                    ("raw first action", raw_action[0, 1]),
                    ("normalized first action", action_out[0, 0]),
                ):
                    print(label + ":", [round(float(v), 6) for v in values[:6]], flush=True)
                print("task:", batch["task"][0], flush=True)
            if (number + 1) % 25 == 0:
                print(f"{split}: {number + 1}/{args.samples} samples passed", flush=True)
        assert number + 1 == args.samples
    assert not observed["train"] & observed["val"]
    print(
        f"PASS: {args.samples} random train + {args.samples} random val windows through LeRobot FLUX3 preprocessing",
        flush=True,
    )


if __name__ == "__main__":
    main()
