# Structured token streams under vocabulary restriction

`n_bins` counts structured tokens before EOS; `eos_at` is where the model stopped.

## Group `HAND`

### Same frame: matched vs mismatched instruction

| frame | match bins | mismatch bins | common prefix | jaccard | match stopped at | mismatch stopped at |
|---|---:|---:|---:|---:|---:|---:|
| apple_ep000000_f0 | 24 | 24 | 2 | 0.11 | 24 | 24 |
| apple_ep000037_f0 | 24 | 1 | 0 | 0.00 | 24 | 2 |

### Stream length per prompt (a short stream means the model stopped early)

| prompt | runs | mean bins | runs with <=1 bin |
|---|---:|---:|---:|
| future_region | 4 | 0.5 | 4 |
| match_instruction | 4 | 12.0 | 2 |
| mismatch_instruction | 4 | 6.2 | 3 |
| point | 4 | 0.5 | 4 |

### Same prompt across frames (does the opening bin depend on the scene?)

| prompt | first bins per frame |
|---|---|
| future_region | apple=-, apple=97, apple=-, apple=97 |
| match_instruction | apple=-, apple=68, apple=-, apple=68 |
| mismatch_instruction | apple=-, apple=68, apple=-, apple=242 |
| point | apple=-, apple=142, apple=-, apple=142 |

## Group `POS`

### Same frame: matched vs mismatched instruction

| frame | match bins | mismatch bins | common prefix | jaccard | match stopped at | mismatch stopped at |
|---|---:|---:|---:|---:|---:|---:|
| apple_ep000000_f0 | 24 | 24 | 2 | 0.23 | 24 | 24 |
| apple_ep000037_f0 | 24 | 24 | 4 | 0.23 | 24 | 24 |

### Stream length per prompt (a short stream means the model stopped early)

| prompt | runs | mean bins | runs with <=1 bin |
|---|---:|---:|---:|
| future_region | 4 | 0.5 | 4 |
| match_instruction | 4 | 12.0 | 2 |
| mismatch_instruction | 4 | 12.0 | 2 |
| point | 4 | 0.5 | 4 |

### Same prompt across frames (does the opening bin depend on the scene?)

| prompt | first bins per frame |
|---|---|
| future_region | apple=-, apple=2, apple=-, apple=27 |
| match_instruction | apple=-, apple=169, apple=-, apple=169 |
| mismatch_instruction | apple=-, apple=169, apple=-, apple=169 |
| point | apple=-, apple=166, apple=-, apple=166 |

## Group `seg`

### Same frame: matched vs mismatched instruction

| frame | match bins | mismatch bins | common prefix | jaccard | match stopped at | mismatch stopped at |
|---|---:|---:|---:|---:|---:|---:|
| apple_ep000000_f0 | 24 | 24 | 0 | 0.40 | 24 | 24 |
| apple_ep000037_f0 | 24 | 24 | 0 | 0.30 | 24 | 24 |

### Stream length per prompt (a short stream means the model stopped early)

| prompt | runs | mean bins | runs with <=1 bin |
|---|---:|---:|---:|
| future_region | 4 | 6.2 | 3 |
| match_instruction | 4 | 12.0 | 2 |
| mismatch_instruction | 4 | 12.0 | 2 |
| point | 4 | 6.0 | 3 |

### Same prompt across frames (does the opening bin depend on the scene?)

| prompt | first bins per frame |
|---|---|
| future_region | apple=-, apple=27, apple=-, apple=32 |
| match_instruction | apple=-, apple=188, apple=-, apple=188 |
| mismatch_instruction | apple=-, apple=27, apple=-, apple=5 |
| point | apple=-, apple=27, apple=-, apple=- |

