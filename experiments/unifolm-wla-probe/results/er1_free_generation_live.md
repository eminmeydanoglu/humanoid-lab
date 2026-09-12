# ER-1 free generation, live rerun (2026-09-12)

Command (from `src/unifolm-wla-probe/`):

```
venvs/unifolm-wla/bin/python probe_model.py \
  --model-dir models/unifolm-wla-1.0/UnifoLM-ER-1 \
  --manifest outputs/unifolm-wla-probe/frames/manifest_clean4.jsonl \
  --prompts prompts.json \
  --prompt-ids match_instruction,mismatch_instruction,bbox,grasp_point,next_action,free_description \
  --tag er1_live --max-new-tokens 96 --out outputs/unifolm-wla-probe/live_demo
```

Greedy decoding, no vocabulary restriction, no assistant-turn priming. 4 G1 head-camera frames (640x480, frame 0 of an episode) x 6 prompts = 24 rows. Wall clock 97.5 s for load + all 24 generations on an RTX 5090 Laptop (24 GB); first row at t+4.3 s, last at t+94.2 s, i.e. ~3.8 s per answer including image preprocessing.

Language check: 24/24 outputs are English text with no CJK characters; all 24 terminate with `<|im_end|>` and none hit the 96-token cap. No structured token (POS/ROT/EEF/LOW/HAND/seg/SEP_VQ/robot_state) appears in any free-generation output (`special_group_counts` empty in every row).

| frame | prompt id | question | answer |
|---|---|---|---|
| `apple_ep000000_f0` | `match_instruction` | Pick up the apple and place it on the plate. | `Reach for the apple on the table and place it on the plate.<\|im_end\|>` |
| `apple_ep000000_f0` | `mismatch_instruction` | Pick up the pear and place it on the plate. | `The pear is not visible in the current scene. I will need to locate it before attempting to place it on the plate.<\|im_end\|>` |
| `apple_ep000000_f0` | `bbox` | Locate the apple in the image and output its bounding box. | `[(255, 415)]<\|im_end\|>` |
| `apple_ep000000_f0` | `grasp_point` | Where should the robot grasp the apple? Give the grasp point. | `[(251, 471)]<\|im_end\|>` |
| `apple_ep000000_f0` | `next_action` | What is the next action the robot should take? Describe it briefly. | `retrieve the apple from the table<\|im_end\|>` |
| `apple_ep000000_f0` | `free_description` | Describe the scene and the manipulation task that is possible here. | `The robotic arm is positioned near the empty teal plate. The apple is on the left side of the table. The task is to place the apple onto the plate.<\|im_end\|>` |
| `pear_ep000074_f0` | `match_instruction` | Pick up the pear and place it on the plate. | `position the right hand to the yellow pear<\|im_end\|>` |
| `pear_ep000074_f0` | `mismatch_instruction` | Pick up the apple and place it on the plate. | `position the apple above the plate<\|im_end\|>` |
| `pear_ep000074_f0` | `bbox` | Locate the pear in the image and output its bounding box. | `[(816, 512)]<\|im_end\|>` |
| `pear_ep000074_f0` | `grasp_point` | Where should the robot grasp the pear? Give the grasp point. | `[(848, 426)]<\|im_end\|>` |
| `pear_ep000074_f0` | `next_action` | What is the next action the robot should take? Describe it briefly. | `reach for the pear on the table<\|im_end\|>` |
| `pear_ep000074_f0` | `free_description` | Describe the scene and the manipulation task that is possible here. | `The robotic arm is positioned near a pink plate. A yellow object lies beside the plate. The arm's grippers are open, ready to interact with the objects on the table.<\|im_end\|>` |
| `grapes_ep000000_f0` | `match_instruction` | Pick up the grapes and place it on the plate. | `retrieve the grapes from the table<\|im_end\|>` |
| `grapes_ep000000_f0` | `mismatch_instruction` | Pick up the apple and place it on the plate. | `Position the apple above the plate<\|im_end\|>` |
| `grapes_ep000000_f0` | `bbox` | Locate the grapes in the image and output its bounding box. | `[(791, 434)]<\|im_end\|>` |
| `grapes_ep000000_f0` | `grasp_point` | Where should the robot grasp the grapes? Give the grasp point. | `[(791, 450)]<\|im_end\|>` |
| `grapes_ep000000_f0` | `next_action` | What is the next action the robot should take? Describe it briefly. | `move towards the orange on the table<\|im_end\|>` |
| `grapes_ep000000_f0` | `free_description` | Describe the scene and the manipulation task that is possible here. | `The robotic arm is positioned near a pink plate. A green object is on the right side of the table. The arm's grippers are open, ready to interact with the object.<\|im_end\|>` |
| `starfruit_ep000111_f0` | `match_instruction` | Pick up the starfruit and place it on the plate. | `move the starfruit and place it on the plate<\|im_end\|>` |
| `starfruit_ep000111_f0` | `mismatch_instruction` | Pick up the apple and place it on the plate. | `position the arm over the apple<\|im_end\|>` |
| `starfruit_ep000111_f0` | `bbox` | Locate the starfruit in the image and output its bounding box. | `[(311, 836)]<\|im_end\|>` |
| `starfruit_ep000111_f0` | `grasp_point` | Where should the robot grasp the starfruit? Give the grasp point. | `[(318, 867)]<\|im_end\|>` |
| `starfruit_ep000111_f0` | `next_action` | What is the next action the robot should take? Describe it briefly. | `retrieve the starfruit from the table<\|im_end\|>` |
| `starfruit_ep000111_f0` | `free_description` | Describe the scene and the manipulation task that is possible here. | `The robotic arm is positioned near a green plate. A bunch of grapes lies to the right of the plate, ready for placement. The setup allows for easy transfer of grapes onto the plate.<\|im_end\|>` |

Notes on answer quality:

- Grounding prompts return pixel points, not boxes: `bbox` -> `[(255, 415)]`, `grasp_point` -> `[(251, 471)]` for the apple frame. `bbox` for pear gives `[(816, 512)]` and for starfruit `[(311, 836)]`, i.e. y beyond the 480-pixel image height, so the coordinates are unreliable as metric grounding even though they are visually plausible as an estimate.

- Wrong-object hallucination: asked to pick the pear on an apple frame the model says the pear is not visible; asked to pick the cup (not in any frame) it answers "The cup is already on the plate." Consistent with the earlier pilot row.

- `free_description` on `starfruit_ep000111_f0` describes "a bunch of grapes", a real misidentification on that frame.

