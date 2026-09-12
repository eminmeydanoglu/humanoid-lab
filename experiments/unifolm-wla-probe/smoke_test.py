#!/usr/bin/env python3
"""Minimal load + one-generation check for a local UnifoLM ER checkpoint.

    /home/aksoy-msi/code/humanoid-lab-main/data/venvs/unifolm-wla/bin/python smoke_test.py \
        --model-dir .../UnifoLM-ER-1

Reports load time, VRAM use, and the raw generated text (special tokens left
visible) so the tokenizer's extended vocabulary is observable directly.
"""

import argparse
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--image", default="/home/aksoy-msi/code/humanoid-lab-main/data/outputs/unifolm-wla-probe/frames/apple_ep000000_f0.png")
ap.add_argument("--prompt", default="Pick up the red apple and place it on the plate.")
ap.add_argument("--max-new-tokens", type=int, default=128)
ap.add_argument("--no-cache", action="store_true", help="recompute the prefix every step (checkpoint default)")
args = ap.parse_args()

print(f"loading {args.model_dir}", flush=True)
t0 = time.time()
processor = AutoProcessor.from_pretrained(args.model_dir)
model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir, device_map="cuda:0", dtype=torch.bfloat16)
model.eval()
print(f"loaded in {time.time() - t0:.0f}s | VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

image = Image.open(args.image).convert("RGB")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
print(f"prompt tokens: {inputs['input_ids'].shape[1]}", flush=True)

t0 = time.time()
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=not args.no_cache,
    )
new = out[0][inputs["input_ids"].shape[1]:]
elapsed = time.time() - t0
print(
    f"generated {len(new)} tokens in {elapsed:.1f}s "
    f"({len(new) / elapsed:.1f} tok/s, use_cache={not args.no_cache}) "
    f"| peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
    flush=True,
)
print("--- raw output (special tokens shown) ---")
print(processor.tokenizer.decode(new, skip_special_tokens=False))
print("--- plain text ---")
print(processor.tokenizer.decode(new, skip_special_tokens=True))
