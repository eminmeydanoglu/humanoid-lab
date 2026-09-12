# Primed action protocol (raw evidence)

Source: `prefix2/` runs (7 x 16 generations, greedy, `--max-new-tokens 160`), frames
from `frames/manifest_clean4.jsonl`. Generated text is quoted verbatim, `<|im_end|>`
included.

## The protocol ER-Flow emits when the assistant turn starts with an action opener

34 tokens, always the same shape:

```
<|POS_x|> x8  <|ROT_x|> x8  <|EEF_END|>  <|HAND_START|>
<|HAND_x|> x4  <|SEP_VQ|>  <|HAND_x|> x4  <|SEP_VQ|>  <|HAND_x|> x4  <seg_end>
```

Two structural facts fall out of this. There is no EEF bin group in the tokenizer,
only `<|EEF_START|>`/`<|EEF_END|>` — and the primed output encodes the end-effector
pose as exactly 8 `POS` plus 8 `ROT` bins. The hand block is three groups of four
bins separated by `<|SEP_VQ|>`, consistent with a residual-VQ hierarchy.

## `<|EEF_START|>` priming, all 16 rows

| frame | prompt | generated text |
|---|---|---|
| apple_ep000000_f0 | match | `<|POS_159|><|POS_6|><|POS_164|><|POS_213|><|POS_141|><|POS_246|><|POS_162|><|POS_253|><|ROT_204|><|ROT_117|><|ROT_239|><|ROT_211|><|ROT_237|><|ROT_142|><|ROT_243|><|ROT_141|><|EEF_END|><|HAND_START|><|HAND_18|><|HAND_177|><|HAND_207|><|HAND_137|><|SEP_VQ|><|HAND_163|><|HAND_215|><|HAND_113|><|HAND_66|><|SEP_VQ|><|HAND_231|><|HAND_47|><|HAND_229|><|HAND_4|><seg_end><|im_end|>` |
| apple_ep000000_f0 | mismatch | `<|POS_159|><|POS_102|><|POS_223|><|POS_99|><|POS_141|><|POS_136|><|POS_245|><|POS_15|><|ROT_37|><|ROT_25|><|ROT_239|><|ROT_171|><|ROT_237|><|ROT_15|><|ROT_243|><|ROT_254|><|EEF_END|><|HAND_START|><|HAND_18|><|HAND_177|><|HAND_110|><|HAND_63|><|SEP_VQ|><|HAND_163|><|HAND_215|><|HAND_215|><|HAND_121|><|SEP_VQ|><|HAND_231|><|HAND_47|><|HAND_92|><|HAND_62|><seg_end><|im_end|>` |
| apple_ep000000_f0 | do_not_touch | `<|POS_159|><|POS_102|><|POS_223|><|POS_99|><|POS_141|><|POS_139|><|POS_229|><|POS_250|><|ROT_37|><|ROT_25|><|ROT_239|><|ROT_171|><|ROT_237|><|ROT_15|><|ROT_243|><|ROT_254|><|EEF_END|><|HAND_START|><|HAND_7|><|HAND_200|><|HAND_113|><|HAND_235|><|SEP_VQ|><|HAND_59|><|HAND_128|><|HAND_111|><|HAND_20|><|SEP_VQ|><|HAND_3|><|HAND_137|><|HAND_189|><|HAND_214|><seg_end><|im_end|>` |
| apple_ep000000_f0 | future_region | `<|im_end|>` |
| pear_ep000074_f0 | match | `<|POS_159|><|POS_121|><|POS_32|><|POS_215|><|POS_225|><|POS_246|><|POS_176|><|POS_254|><|ROT_28|><|ROT_123|><|ROT_119|><|ROT_197|><|ROT_251|><|ROT_15|><|ROT_221|><|ROT_193|><|EEF_END|><|HAND_START|><|HAND_43|><|HAND_98|><|HAND_44|><|HAND_163|><|SEP_VQ|><|HAND_189|><|HAND_152|><|HAND_66|><|HAND_127|><|SEP_VQ|><|HAND_142|><|HAND_114|><|HAND_103|><|HAND_214|><seg_end><|im_end|>` |
| pear_ep000074_f0 | mismatch | `<|POS_159|><|POS_121|><|POS_32|><|POS_215|><|POS_225|><|POS_253|><|POS_115|><|POS_254|><|ROT_28|><|ROT_242|><|ROT_123|><|ROT_171|><|ROT_8|><|ROT_250|><|ROT_160|><|ROT_251|><|EEF_END|><|HAND_START|><|HAND_43|><|HAND_98|><|HAND_44|><|HAND_74|><|SEP_VQ|><|HAND_189|><|HAND_152|><|HAND_150|><|HAND_150|><|SEP_VQ|><|HAND_142|><|HAND_114|><|HAND_42|><|HAND_139|><seg_end><|im_end|>` |
| pear_ep000074_f0 | do_not_touch | `<|POS_159|><|POS_102|><|POS_223|><|POS_99|><|POS_141|><|POS_139|><|POS_229|><|POS_250|><|ROT_37|><|ROT_25|><|ROT_239|><|ROT_171|><|ROT_237|><|ROT_15|><|ROT_243|><|ROT_254|><|EEF_END|><|HAND_START|><|HAND_7|><|HAND_200|><|HAND_113|><|HAND_235|><|SEP_VQ|><|HAND_59|><|HAND_128|><|HAND_111|><|HAND_20|><|SEP_VQ|><|HAND_3|><|HAND_137|><|HAND_189|><|HAND_214|><seg_end><|im_end|>` |
| pear_ep000074_f0 | future_region | `<|im_end|>` |
| grapes_ep000000_f0 | match | `<|POS_159|><|POS_121|><|POS_32|><|POS_215|><|POS_250|><|POS_184|><|POS_147|><|POS_62|><|ROT_225|><|ROT_225|><|ROT_90|><|ROT_245|><|ROT_45|><|ROT_142|><|ROT_186|><|ROT_187|><|EEF_END|><|HAND_START|><|HAND_18|><|HAND_177|><|HAND_110|><|HAND_63|><|SEP_VQ|><|HAND_163|><|HAND_215|><|HAND_215|><|HAND_121|><|SEP_VQ|><|HAND_231|><|HAND_47|><|HAND_92|><|HAND_62|><seg_end><|im_end|>` |
| grapes_ep000000_f0 | mismatch | `<|POS_159|><|POS_121|><|POS_32|><|POS_215|><|POS_225|><|POS_253|><|POS_115|><|POS_254|><|ROT_28|><|ROT_123|><|ROT_123|><|ROT_191|><|ROT_220|><|ROT_15|><|ROT_200|><|ROT_215|><|EEF_END|><|HAND_START|><|HAND_18|><|HAND_177|><|HAND_110|><|HAND_63|><|SEP_VQ|><|HAND_163|><|HAND_215|><|HAND_215|><|HAND_121|><|SEP_VQ|><|HAND_231|><|HAND_47|><|HAND_92|><|HAND_62|><seg_end><|im_end|>` |
| grapes_ep000000_f0 | do_not_touch | `<|POS_159|><|POS_102|><|POS_223|><|POS_99|><|POS_141|><|POS_139|><|POS_229|><|POS_250|><|ROT_37|><|ROT_25|><|ROT_239|><|ROT_171|><|ROT_237|><|ROT_15|><|ROT_243|><|ROT_254|><|EEF_END|><|HAND_START|><|HAND_7|><|HAND_200|><|HAND_113|><|HAND_235|><|SEP_VQ|><|HAND_59|><|HAND_128|><|HAND_111|><|HAND_20|><|SEP_VQ|><|HAND_3|><|HAND_137|><|HAND_189|><|HAND_214|><seg_end><|im_end|>` |
| grapes_ep000000_f0 | future_region | `<|im_end|>` |
| starfruit_ep000111_f0 | match | `<|POS_159|><|POS_6|><|POS_164|><|POS_213|><|POS_141|><|POS_246|><|POS_225|><|POS_251|><|ROT_204|><|ROT_117|><|ROT_173|><|ROT_182|><|ROT_240|><|ROT_80|><|ROT_250|><|ROT_172|><|EEF_END|><|HAND_START|><|HAND_82|><|HAND_11|><|HAND_120|><|HAND_31|><|SEP_VQ|><|HAND_163|><|HAND_215|><|HAND_244|><|HAND_3|><|SEP_VQ|><|HAND_117|><|HAND_194|><|HAND_14|><|HAND_81|><seg_end><|im_end|>` |
| starfruit_ep000111_f0 | mismatch | `<|POS_159|><|POS_102|><|POS_223|><|POS_99|><|POS_181|><|POS_139|><|POS_229|><|POS_250|><|ROT_37|><|ROT_25|><|ROT_239|><|ROT_171|><|ROT_237|><|ROT_15|><|ROT_243|><|ROT_254|><|EEF_END|><|HAND_START|><|HAND_34|><|HAND_148|><|HAND_248|><|HAND_8|><|SEP_VQ|><|HAND_11|><|HAND_119|><|HAND_63|><|HAND_219|><|SEP_VQ|><|HAND_231|><|HAND_172|><|HAND_8|><|HAND_56|><|SEP_VQ|><|HAND_231|><|HAND_54|><seg078>...` (degenerates into a ~160-token `<seg>` run) |
| starfruit_ep000111_f0 | do_not_touch | `<|POS_164|><|POS_158|><|POS_197|><|POS_196|><|POS_74|><|POS_230|><|POS_242|><|POS_171|><|ROT_110|><|ROT_124|><|ROT_220|><|ROT_72|><|ROT_93|><|ROT_47|><|ROT_66|><|ROT_11|><|EEF_END|><|HAND_START|><|HAND_82|><|HAND_11|><|HAND_158|><|HAND_146|><|SEP_VQ|><|HAND_163|><|HAND_215|><|HAND_244|><|HAND_26|><|SEP_VQ|><|HAND_117|><|HAND_77|><|HAND_27|><|HAND_56|><seg_end><|im_end|>` |
| starfruit_ep000111_f0 | future_region | `<|im_end|>` |

## Row counts by priming token

| priming token | full protocol | partial (structured, not the block) | plain English | immediate EOS |
|---|---:|---:|---:|---:|
| `<|EEF_START|>` (ER-Flow) | 12 | 0 | 0 | 4 |
| `<|LOW_START|>` (ER-Flow) | 11 | 2 | 0 | 3 |
| `<|robot_state|>` (ER-Flow) | 7 | 4 | 0 | 5 |
| `<|HAND_START|>` (ER-Flow) | 5 | 11 | 0 | 0 |
| `<seg_begin>` (ER-Flow) | 0 | 16 (pure `<segNNN>` streams) | 0 | 0 |
| `<|EEF_START|>` (ER-1) | 0 | 0 | 12 | 4 |
| `<|HAND_START|>` (ER-1) | 0 | 0 | 7 | 9 |

Observations that matter for interpreting this:

1. **The priming token acts as a mode switch, not a block selector.** Priming with
   `<|LOW_START|>` still produces the EEF+HAND block; no `LOW` bins are emitted in
   any run. What changes between primes is mostly how often the model enters the
   structured mode at all.
2. **`future_region` never enters the action protocol** — under `<|EEF_START|>`,
   `<|LOW_START|>` and `<|robot_state|>` it returns `<|im_end|>` immediately on all
   frames, and under `<seg_begin>` it produces a `<segNNN>` stream. The region and
   action pathways are cleanly separated by prompt.
3. **Instruction binding is weak.** `mismatch_instruction` and `do_not_touch`
   converge on nearly identical POS/ROT sequences (e.g. on `apple_ep000000_f0` they
   share `<|POS_159|><|POS_102|><|POS_223|><|POS_99|><|POS_141|>` and an identical
   ROT run), and `do_not_touch` still emits a full pick action. So under priming the
   instruction perturbs the tokens but does not gate the behaviour — unlike the
   vocabulary-restricted runs, where `do_not_touch` collapsed the HAND stream to EOS.
4. **Bin values are conditioning-path dependent.** The same frame and instruction
   yields different POS bins under restriction (`169, 239, 195, …`) versus priming
   (`159, 6, 164, …`), so any attempt to map bins to coordinates must fix the
   conditioning path first.
