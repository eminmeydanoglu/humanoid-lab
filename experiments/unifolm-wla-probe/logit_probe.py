#!/usr/bin/env python3
"""Probe the next-token distribution over Unitree's extended vocabulary.

    .../venvs/unifolm-wla/bin/python logit_probe.py --model-dir .../UnifoLM-ER-1 \
        --manifest .../frames/manifest.jsonl --prompts prompts.json --out .../logits --tag er1

Free generation answers in plain English and never emits POS/ROT/EEF/LOW/HAND/seg
tokens, so this measures the question one level down: after the prompt, how much
probability mass does the model put on each structured group, and which bins
dominate? A single forward pass per (image, prompt) is enough, and comparing the
same image across prompts shows whether the instruction changes the structured
distribution even when the sampled text does not.
"""

import argparse
import collections
import json
import pathlib
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

PATTERNS = [
    (re.compile(r"^<\|POS_(\d+)\|>$"), "POS"),
    (re.compile(r"^<\|ROT_(\d+)\|>$"), "ROT"),
    (re.compile(r"^<\|EEF_(\d+)\|>$"), "EEF"),
    (re.compile(r"^<\|LOW_(\d+)\|>$"), "LOW"),
    (re.compile(r"^<\|HAND_(\d+)\|>$"), "HAND"),
    (re.compile(r"^<seg(\d+)>$"), "seg"),
]
GROUPS = [g for _, g in PATTERNS]


def group_ids(tokenizer):
    ids = collections.defaultdict(list)
    for token, idx in tokenizer.get_added_vocab().items():
        for pattern, group in PATTERNS:
            if pattern.match(token):
                ids[group].append(idx)
                break
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--prompt-ids", default="", help="comma-separated subset of prompt ids")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.manifest)]
    if args.limit_samples:
        samples = samples[: args.limit_samples]
    prompts = json.load(open(args.prompts))
    if args.prompt_ids:
        wanted = set(args.prompt_ids.split(","))
        prompts = [p for p in prompts if p["id"] in wanted]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, device_map="cuda:0", dtype=torch.bfloat16
    ).eval()
    tokenizer = processor.tokenizer
    ids_by_group = group_ids(tokenizer)
    print(f"group vocabulary sizes: { {g: len(v) for g, v in ids_by_group.items()} }", flush=True)

    rows = []
    for sample in samples:
        image = Image.open(sample["image"]).convert("RGB")
        for spec in prompts:
            other = next(f for f in ("apple", "pear", "grapes", "starfruit") if f != sample["fruit"])
            prompt = spec["prompt"].format(fruit=sample["fruit"], other_fruit=other)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            row = {
                "tag": args.tag,
                "sample": sample["id"],
                "fruit": sample["fruit"],
                "prompt_id": spec["id"],
                "prompt_family": spec["family"],
                "prompt": prompt,
                "argmax_token": tokenizer.convert_ids_to_tokens(int(probs.argmax())),
                "argmax_prob": float(probs.max()),
                "group_mass": {},
                "group_top": {},
            }
            for group in GROUPS:
                if not ids_by_group[group]:
                    row["group_mass"][group] = 0.0
                    row["group_top"][group] = []
                    continue
                gids = torch.tensor(ids_by_group[group], dtype=torch.long, device=probs.device)
                gp = probs[gids]
                row["group_mass"][group] = float(gp.sum())
                top = torch.topk(gp, k=min(args.topk, len(gids)))
                row["group_top"][group] = [
                    [tokenizer.convert_ids_to_tokens(int(gids[i])), round(float(p), 6)]
                    for p, i in zip(top.values, top.indices)
                ]
            rows.append(row)
            print(
                f"{sample['id']:24s} {spec['id']:20s} argmax={row['argmax_token'][:28]:28s} "
                f"mass={ {g: round(v, 4) for g, v in row['group_mass'].items() if v > 1e-4} }",
                flush=True,
            )

    with open(out_dir / f"{args.tag}_logits.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Same image, matched vs mismatched instruction: does the structured mass move?
    lines = ["# Structured-vocabulary logit probe", "", "## Probability mass per group (first generated position)", "",
             "| frame | prompt | argmax | " + " | ".join(GROUPS) + " |", "|---|---|---|" + "---:|" * len(GROUPS)]
    for r in rows:
        mass = " | ".join(f"{r['group_mass'][g]:.2e}" for g in GROUPS)
        lines.append(f"| {r['sample']} | {r['prompt_id']} | `{r['argmax_token']}` | {mass} |")
    with open(out_dir / f"{args.tag}_logits_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
