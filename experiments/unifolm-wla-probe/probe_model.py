#!/usr/bin/env python3
"""Run a prompt battery against a local UnifoLM ER checkpoint.

    /home/aksoy-msi/code/humanoid-lab-main/data/venvs/unifolm-wla/bin/python probe_model.py \
        --model-dir /home/aksoy-msi/code/humanoid-lab-main/data/models/unifolm-wla-1.0/UnifoLM-ER-1 \
        --tag er1

Generation is greedy: the goal is to compare token streams across prompts, and
sampling would swamp prompt effects with sampling noise. Every emitted token in
Unitree's extended vocabulary (POS/ROT/EEF/LOW/HAND/seg/SEP_VQ/robot_state) is
recorded, so we can see whether a prompt elicits action or region tokens at all
before any decoder exists.
"""

import argparse
import collections
import json
import pathlib
import re
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, LogitsProcessor

GROUP_PATTERNS = [
    (re.compile(r"^<\|POS_(\d+)\|>$"), "POS"),
    (re.compile(r"^<\|ROT_(\d+)\|>$"), "ROT"),
    (re.compile(r"^<\|EEF_(\d+)\|>$"), "EEF"),
    (re.compile(r"^<\|LOW_(\d+)\|>$"), "LOW"),
    (re.compile(r"^<\|HAND_(\d+)\|>$"), "HAND"),
    (re.compile(r"^<seg(\d+)>$"), "seg"),
    (re.compile(r"^<\|SEP_VQ\|>$"), "SEP_VQ"),
    (re.compile(r"^<\|robot_state\|>$"), "robot_state"),
    (re.compile(r"^<\|robot_state_implicit_stats\|>$"), "robot_state_stats"),
    (re.compile(r"^<seg_begin>$"), "seg_begin"),
    (re.compile(r"^<seg_end>$"), "seg_end"),
]


def group_of(token_str):
    for pattern, name in GROUP_PATTERNS:
        if pattern.match(token_str):
            return name
    return None


class RestrictToGroups(LogitsProcessor):
    """Force generation into selected structured groups, plus EOS to allow stopping.

    Free generation only ever produces natural language, so restricting the
    vocabulary tests whether the structured bins are reachable at all and, if so,
    which bins the model prefers for a given image and prompt.
    """

    def __init__(self, allowed_ids):
        self.allowed = torch.tensor(sorted(allowed_ids), dtype=torch.long)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed.to(scores.device)] = 0.0
        return scores + mask


def ids_for_groups(tokenizer, groups):
    wanted = set(groups)
    allowed = set()
    for token, idx in tokenizer.get_added_vocab().items():
        if group_of(token) in wanted:
            allowed.add(idx)
    allowed.add(tokenizer.eos_token_id)
    return allowed


def load_model(model_dir):
    processor = AutoProcessor.from_pretrained(model_dir)
    kwargs = {"device_map": "cuda:0", "dtype": torch.bfloat16}
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, **kwargs)
    model.eval()
    return processor, model


def generate(processor, model, image, prompt, max_new_tokens, logits_processor=None, assistant_prefix=""):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if assistant_prefix:
        # The template closes an assistant turn with <|im_end|>, so the prefix is appended
        # to the rendered prompt instead of being passed as an assistant message.
        text += assistant_prefix
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
    prompt_len = inputs["input_ids"].shape[1]
    kwargs = {"logits_processor": [logits_processor]} if logits_processor else {}
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, **kwargs
        )
    new_ids = out[0][prompt_len:].tolist()
    return new_ids, prompt_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit-samples", type=int, default=0, help="0 = all frames in the manifest")
    ap.add_argument("--prompt-ids", default="", help="comma-separated subset of prompt ids")
    ap.add_argument("--save-token-ids", action="store_true")
    ap.add_argument("--restrict-group", default="", help="comma-separated groups to force, e.g. POS or POS,ROT")
    ap.add_argument("--assistant-prefix", default="", help="text appended after the assistant header, e.g. '<|EEF_START|>'")
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

    processor, model = load_model(args.model_dir)
    tokenizer = processor.tokenizer
    restrictor = None
    if args.restrict_group:
        groups = [g.strip() for g in args.restrict_group.split(",") if g.strip()]
        allowed = ids_for_groups(tokenizer, groups)
        restrictor = RestrictToGroups(allowed)
        print(f"restricting generation to {groups} ({len(allowed)} tokens incl. EOS)", flush=True)

    started = time.time()
    results = []
    prompt_ids = [spec["id"] for spec in prompts]
    with open(out_dir / f"{args.tag}_results.jsonl", "w") as f:
        for sample in samples:
            image = Image.open(sample["image"]).convert("RGB")
            for spec in prompts:
                other = next(f for f in ("apple", "pear", "grapes", "starfruit") if f != sample["fruit"])
                prompt = spec["prompt"].format(fruit=sample["fruit"], other_fruit=other)
                ids, prompt_len = generate(
                    processor, model, image, prompt, args.max_new_tokens, restrictor, args.assistant_prefix
                )
                names = [tokenizer.convert_ids_to_tokens(i) for i in ids]
                groups = collections.Counter(g for g in (group_of(n) for n in names) if g)
                row = {
                    "tag": args.tag,
                    "sample": sample["id"],
                    "fruit": sample["fruit"],
                    "prompt_id": spec["id"],
                    "prompt_family": spec["family"],
                    "prompt": prompt,
                    "text": tokenizer.decode(ids, skip_special_tokens=False),
                    "n_new_tokens": len(ids),
                    "hit_max_new_tokens": len(ids) >= args.max_new_tokens,
                    "special_group_counts": dict(groups),
                    "latency_s": round(time.time() - started, 1),
                }
                if args.save_token_ids:
                    row["token_ids"] = ids
                f.write(json.dumps(row) + "\n")
                f.flush()
                results.append(row)
                print(
                    f"{sample['id']:24s} {spec['id']:20s} tokens={len(ids):4d} "
                    f"groups={dict(groups) if groups else '-'}",
                    flush=True,
                )

    with open(out_dir / f"{args.tag}_summary.json", "w") as f:
        json.dump(
            {
                "model_dir": args.model_dir,
                "samples": len(samples),
                "prompts": len(prompts),
                "max_new_tokens": args.max_new_tokens,
                "restrict_group": args.restrict_group,
                "prompts_eliciting_special_groups": {
                    pid: sum(1 for r in results if r["prompt_id"] == pid and r["special_group_counts"])
                    for pid in prompt_ids
                },
                "last_prompt_len_tokens": prompt_len,
                "wall_clock_s": round(time.time() - started, 1),
            },
            f,
            indent=2,
        )
    print(f"wrote {len(results)} rows to {out_dir}")


if __name__ == "__main__":
    main()
