# UnifoLM-WLA-1.0 ER probe

Experiments on Unitree's `UnifoLM-ER-1` and `UnifoLM-ER-Flow` checkpoints (released
11 Sep 2026, https://huggingface.co/collections/unitreerobotics/unifolm-wla-10).
Unitree has not published inference code for these weights, so every question here
is answered by probing the checkpoints directly.

The question that drives this work: does changing only the language instruction —
with the image held fixed — change the model's physical "intent" (predicted future
dynamic regions and discrete robot-action tokens)?

## What the checkpoints actually are

| | UnifoLM-ER-1 | UnifoLM-ER-Flow |
|---|---|---|
| Tensors | 713 | 718 |
| Architecture | `Qwen3VLForConditionalGeneration` | same |
| Base | Qwen3-VL-4B | Qwen3-VL-4B |
| Extra | — | `lm_head`, `robot_state_projector.net.{0,2}` |
| Config extras | — | `robot_state_dim: 120`, `robot_state_token: <\|robot_state\|>` |

Verified by diffing safetensors headers against `Qwen/Qwen3-VL-4B-Instruct`:
ER-1's 713 tensor names and shapes are **identical** to Qwen3-VL-4B; the only
difference is `embed_tokens` `[153088, 2560]` versus `[151936, 2560]`, the extra
rows being Unitree's added special tokens.

Consequences:

- **ER-1 loads with stock `transformers`** and no custom code.
- **ER-Flow does not**: `robot_state_projector` is not part of any Qwen3-VL class,
  so `from_pretrained` reports it as `UNEXPECTED` and silently drops it. The
  released weights therefore cannot condition on robot state, and the RVQ action
  decoders plus the dynamic-region VQ-VAE are not published at all. Token streams
  can be studied; tokens cannot be turned into trajectories or masks.

## Environment (this machine)

Use the dedicated venv, not the GR00T one:

```
/home/aksoy-msi/code/humanoid-lab-main/data/venvs/unifolm-wla/bin/python
  torch 2.9.0+cu128, torchvision 0.24.0+cu128, triton 3.5.0, transformers 5.5.3
```

Three traps found while building it, all load-bearing:

1. **`transformers 4.57.3` cannot construct the model.** It reads the RoPE settings
   from `config.rope_scaling`, while these checkpoints use the newer
   `rope_parameters` key, so the value is `None` and model construction dies with
   `AttributeError: 'NoneType' object has no attribute 'get'`. The checkpoints were
   saved with `transformers 5.5.3`; use that version.
2. **`torchvision` is required** even for image-only use: the repo's
   `processor_config.json` declares a video processor, and `AutoProcessor` builds
   it, which raises `Qwen3VLVideoProcessor requires the Torchvision library`.
3. **Newer torch/triton silently breaks generation here.** With torch 2.14/triton
   3.8 the CUDA helper module fails to JIT-compile (`/usr/include/python3.12/Python.h`
   is absent, no `python3.12-dev`). Pinning torch 2.9.0 + triton 3.5.0 avoids it.

Models and inputs live in the data root (see `AGENTS.md`; not in git):

```
<data-root>/models/unifolm-wla-1.0/UnifoLM-ER-1
<data-root>/models/unifolm-wla-1.0/UnifoLM-ER-Flow
<data-root>/outputs/unifolm-wla-probe/frames/          # 16 G1 ego_view frames + manifest
```

Both checkpoints were downloaded with a parallel chunked fetcher and verified
against the sha256 that the HF CDN reports in `x-linked-etag`. Note that HF's Xet
transfer path ran ~15x slower here than plain CDN ranges (0.17 MB/s vs 2.6 MB/s
single-connection; ~16 MB/s with 8 parallel chunks).

## Inputs

`frames/` holds 16 frames (4 fruits x 4 episodes) taken from real G1 head-camera
recordings in `<data-root>/datasets/groot/g1-fruits/`, one frame per episode:

```
<data-root>/venvs/lerobot-viz/bin/python extract_frames.py
```

Each fruit dataset carries exactly one instruction ("Pick up the <fruit> and place
it on the plate"), so the same frame can be paired with a matched or a deliberately
mismatched instruction. Every frame contains the target fruit *and* a distractor
fruit, plus the plate and both hands — which makes "same image, different prompt"
contrasts possible with real objects.

## Scripts

| Script | Purpose |
|---|---|
| `extract_frames.py` | Pull probe frames out of the LeRobot-format fruit datasets |
| `smoke_test.py` | Load a checkpoint and generate once; prints VRAM, tok/s, raw output |
| `probe_model.py` | Prompt battery. Supports `--restrict-group` (lock the vocabulary to one token group) and `--assistant-prefix` (start the assistant turn inside the structured format) |
| `logit_probe.py` | One forward pass per (image, prompt); probability mass and top bins per token group |
| `analyze_probe.py` | Elicitation table plus matched-vs-mismatched token-stream divergence |
| `analyze_structured.py` | Compares restricted-generation streams across frames and prompts |

Example runs:

```bash
V=<data-root>/venvs/unifolm-wla/bin/python
M=<data-root>/models/unifolm-wla-1.0/UnifoLM-ER-Flow
OUT=<data-root>/outputs/unifolm-wla-probe

# free generation, 18 prompts on one frame
$V probe_model.py --model-dir $M --manifest $OUT/frames/manifest.jsonl \
  --prompts prompts.json --out $OUT/pilot --tag erflow --limit-samples 1 --save-token-ids

# vocabulary locked to POS tokens
$V probe_model.py --model-dir $M --manifest $OUT/frames/manifest.jsonl --prompts prompts.json \
  --out $OUT/restricted --tag erflow_POS --restrict-group POS --max-new-tokens 32 --save-token-ids

# assistant turn primed inside the action format
$V probe_model.py --model-dir $M --manifest $OUT/frames/manifest.jsonl --prompts prompts.json \
  --out $OUT/prefix --tag erflow_pfx_eef --assistant-prefix "<|EEF_START|>" --max-new-tokens 160
```

## Findings

### 1. Free generation never uses the structured vocabulary

Neither model emits `POS/ROT/EEF/LOW/HAND/seg` tokens for any of the 18 prompts,
including prompts that name the opening tokens (`<|EEF_START|>`, `<|HAND_START|>`,
`<|LOW_START|>`, `<seg_begin>`) inside the user message. Both answer in English. This
is a statement about *inline* priming only — starting the **assistant turn** with
those same tokens does unlock the protocol, see finding 4. ER-1 on
`apple_ep000000_f0`:

- "Pick up the apple and place it on the plate." → `Reach for the apple on the table and place it on the plate.`
- "Locate the apple in the image and output its bounding box." → `[(255, 415)]`
- "Where should the robot grasp the apple? Give the grasp point." → `[(251, 471)]`
- "Pick up the pear..." (no pear in frame) → `The pear is not visible in the current scene.`
- "Pick up the cup..." (no cup in frame) → `The cup is already on the plate.` (hallucination)

Grounding answers come back as pixel-coordinate points in a `[(x, y)]` list, which
matches the "image point prediction" task in the published benchmark table.

### 2. Structured probability mass differs by four orders of magnitude

`logit_probe.py`, first generated position, 4 frames x 6 prompts (full table in
`results/*_logits_report.md`):

| Group | ER-1 | ER-Flow |
|---|---|---|
| POS | 3.7e-7 … 1.6e-10 | 1.7e-4 … 1.3e-3 |
| ROT | 2.5e-7 … 1.3e-10 | 2.5e-4 … 1.8e-3 |
| HAND | 2.5e-7 … 1.3e-10 | 9.7e-4 … 3.9e-3 |
| seg | 2.5e-7 … 1.3e-10 | 2.4e-4 … 1.9e-2 |

ER-1's mass sits at numerical noise, and its within-group ordering is by token id,
i.e. uniform — the vocabulary is effectively dead. ER-Flow retains a real, weak
(0.02%–2%) and prompt-sensitive bias: the `seg` mass for `future_region` reaches
1.9e-2 on one frame while the `point` prompt drops to ~1e-9.

### 3. Under vocabulary restriction the two models separate cleanly

Locking generation to one group plus EOS:

- **ER-1 emits EOS immediately** for POS, HAND and seg, on every prompt and frame:
  no usable preference exists.
- **ER-Flow produces long structured streams.** From `results/restricted_full_report.md`
  (16 frames x 5 prompts per group):

| prompt | POS mean bins | HAND mean bins | seg mean bins |
|---|---:|---:|---:|
| `match_instruction` | 26.2 | 14.3 | 26.2 |
| `mismatch_instruction` | 28.1 | 7.8 | 26.2 |
| `do_not_touch` | 12.6 | 1.0 | 24.2 |
| `future_region` | 2.8 | 1.0 | 4.9 |
| `point` | 6.1 | 0.9 | 8.0 |

Two behavioural regularities worth following up:

- **Matched and mismatched instructions diverge almost immediately.** On the same
  frame the POS streams share a prefix of 0–4 bins (jaccard 0.09–0.33), e.g. on
  `apple_ep000000_f0` all of match/mismatch/do-not-touch start with the same two
  bins (169, 239) and separate at the third (195 / 102 / 73). The opening appears
  to be scene-driven; the continuation follows the instruction.
- **Stream length carries meaning.** Naming an object that is not in the frame
  collapses the HAND stream to a single bin plus EOS (e.g. `apple_ep000037_f0`:
  32 bins matched, 1 bin mismatched), and `do_not_touch` suppresses the HAND
  stream in 16/16 frames.

Caveat: with structured mass this small, forcing the vocabulary amplifies a weak
prior, and long forced streams tend to degenerate into repeating bins. The first
few bins are the informative part.

### 4. Priming the assistant turn unlocks the full action protocol

Free generation and restriction both fail to reach the structured vocabulary; forcing
the assistant turn to *start* inside the format works. Appending the priming string
after the assistant header (`probe_model.py --assistant-prefix`) makes
**UnifoLM-ER-Flow** emit a complete 34-token action block:

```
<|POS_x|> x8   <|ROT_x|> x8   <|EEF_END|>   <|HAND_START|>
<|HAND_x|> x4  <|SEP_VQ|>  <|HAND_x|> x4  <|SEP_VQ|>  <|HAND_x|> x4   <seg_end>
```

Raw rows are in `results/prefix_protocol.md`; counts over 4 frames x 4 prompts per prime:

| priming token | full protocol | partial | plain English | immediate EOS |
|---|---:|---:|---:|---:|
| `<|EEF_START|>` | 12 | 0 | 0 | 4 |
| `<|LOW_START|>` | 11 | 2 | 0 | 3 |
| `<|robot_state|>` | 7 | 4 | 0 | 5 |
| `<|HAND_START|>` | 5 | 11 | 0 | 0 |
| `<seg_begin>` | 0 | 16 | 0 | 0 |

The same priming on **UnifoLM-ER-1 produces 0 structured rows out of 32**: it answers
in degenerate English, or stops immediately. The structured pathway exists only in
ER-Flow.

Four findings that shape how this can be used:

- **There is no EEF bin group** in the tokenizer, and the primed block encodes the
  end-effector pose as exactly 8 `POS` + 8 `ROT` bins. The hand block is three groups
  of four bins separated by `<|SEP_VQ|>`, consistent with a residual-VQ hierarchy.
- **The priming token is a mode switch, not a block selector.** Priming with
  `<|LOW_START|>` still yields the EEF+HAND block, and no `LOW` bins appear in any run.
- **The `future_region` prompt never enters the action protocol** — it returns
  `<|im_end|>` at once under the action primes, and a `<segNNN>` stream under
  `<seg_begin>`. Region and action pathways are separated by prompt.
- **Instruction binding is weak under priming.** `mismatch_instruction` and
  `do_not_touch` converge on nearly the same POS/ROT sequence, and `do_not_touch`
  still emits a full pick action — unlike the restricted runs, where it suppressed
  the HAND stream. Also, the same frame and instruction give different POS bins
  under restriction (`169, 239, 195, …`) than under priming (`159, 6, 164, …`), so
  bin values depend on the conditioning path and any calibration must fix that path.

## Web UI

`webapp/` holds a small local page over the same models: pick one of the extracted
frames, type a prompt, read the answer, and in point mode see the coordinates the
model printed drawn back onto the photo. See `webapp/README.md` for how to run it and
the Tailscale URLs. Setting it up produced one real finding worth keeping:

**ER-1 answers grounding prompts in a 0-1000 normalised grid, not in frame pixels.**
Checked on four frames against an independent visual estimate, the mapped answers
land within 2-13 px of the object centre; the same numbers read as pixels fall outside
the 640x480 frame in three of the four cases. Details, the phrasing comparison
(`Point to the apple.` returns no numbers at all) and the grasp-point drift are in
`results/point_calibration.md`.

## Open questions

1. **Bin to coordinate mapping is undocumented.** `POS`/`ROT`/`HAND`/`LOW`/`seg`
  each have 256 bins and no published quantisation scheme. Both calibration routes
  are available here: the frames have measured object boxes, and the target and
  distractor fruits sit at different positions in the same image.
2. **Note there is no EEF bin group** in the tokenizer — only
  `<|EEF_START|>`/`<|EEF_END|>`. The primed protocol suggests EEF poses are encoded
  as POS+ROT pairs.
3. **Robot-state conditioning is not reproducible** without the missing projector
  code. It does not affect prompt-comparison experiments (state is held constant
  across prompts), only absolute action quality.
4. **Decoders are not published**, so tokens cannot be rendered as masks or
  trajectories. Every experiment here is therefore token- and distribution-level.
5. `UnifoLM-WLA-1.0` itself (the 6B model the ER models feed into) has not been
  released; Unitree's open-source plan still lists the post-train code as pending.
