# Structured token streams under vocabulary restriction

`n_bins` counts structured tokens before EOS; `stopped at` is the total number of generated tokens (1-2 means the model emitted EOS almost immediately).

## erflow / group `HAND`

### Same frame: matched vs mismatched instruction

| frame | match bins | mismatch bins | common prefix | jaccard | match stopped at | mismatch stopped at |
|---|---:|---:|---:|---:|---:|---:|
| apple_ep000000_f0 | 32 | 32 | 2 | 0.14 | 32 | 32 |
| apple_ep000037_f0 | 32 | 1 | 0 | 0.00 | 32 | 2 |
| apple_ep000074_f0 | 1 | 9 | 0 | 0.00 | 2 | 10 |
| apple_ep000111_f0 | 1 | 20 | 0 | 0.05 | 2 | 21 |
| grapes_ep000000_f0 | 32 | 1 | 0 | 0.00 | 32 | 2 |
| grapes_ep000037_f0 | 27 | 1 | 0 | 0.00 | 28 | 2 |
| grapes_ep000074_f0 | 18 | 13 | 10 | 0.27 | 19 | 14 |
| grapes_ep000111_f0 | 1 | 9 | 0 | 0.00 | 2 | 10 |
| pear_ep000000_f0 | 1 | 1 | 1 | 1.00 | 2 | 2 |
| pear_ep000037_f0 | 32 | 1 | 0 | 0.00 | 32 | 2 |
| pear_ep000074_f0 | 16 | 29 | 2 | 0.07 | 17 | 30 |
| pear_ep000111_f0 | 10 | 3 | 1 | 0.09 | 11 | 4 |
| starfruit_ep000000_f0 | 1 | 1 | 0 | 0.00 | 2 | 2 |
| starfruit_ep000037_f0 | 21 | 1 | 0 | 0.00 | 22 | 2 |
| starfruit_ep000074_f0 | 3 | 1 | 0 | 0.00 | 4 | 2 |
| starfruit_ep000111_f0 | 1 | 1 | 0 | 0.00 | 2 | 2 |

### Stream length per prompt (a short stream means the model stopped early)

| prompt | runs | mean bins | runs with <=1 bin |
|---|---:|---:|---:|
| do_not_touch | 16 | 1.0 | 16 |
| future_region | 16 | 1.0 | 16 |
| match_instruction | 16 | 14.3 | 6 |
| mismatch_instruction | 16 | 7.8 | 9 |
| point | 16 | 0.9 | 16 |

### Same prompt across frames (does the opening bin depend on the scene?)

| prompt | first bins per frame |
|---|---|
| do_not_touch | apple_ep000000_f0=142, apple_ep000037_f0=142, apple_ep000074_f0=142, apple_ep000111_f0=142, grapes_ep000000_f0=142, grapes_ep000037_f0=68, grapes_ep000074_f0=142, grapes_ep000111_f0=142, pear_ep000000_f0=142, pear_ep000037_f0=142, pear_ep000074_f0=142, pear_ep000111_f0=238, starfruit_ep000000_f0=142, starfruit_ep000037_f0=142, starfruit_ep000074_f0=142, starfruit_ep000111_f0=142 |
| future_region | apple_ep000000_f0=97, apple_ep000037_f0=97, apple_ep000074_f0=79, apple_ep000111_f0=171, grapes_ep000000_f0=75, grapes_ep000037_f0=63, grapes_ep000074_f0=190, grapes_ep000111_f0=50, pear_ep000000_f0=63, pear_ep000037_f0=171, pear_ep000074_f0=63, pear_ep000111_f0=63, starfruit_ep000000_f0=63, starfruit_ep000037_f0=50, starfruit_ep000074_f0=63, starfruit_ep000111_f0=145 |
| match_instruction | apple_ep000000_f0=68, apple_ep000037_f0=68, apple_ep000074_f0=255, apple_ep000111_f0=255, grapes_ep000000_f0=16, grapes_ep000037_f0=68, grapes_ep000074_f0=89, grapes_ep000111_f0=237, pear_ep000000_f0=142, pear_ep000037_f0=68, pear_ep000074_f0=68, pear_ep000111_f0=68, starfruit_ep000000_f0=237, starfruit_ep000037_f0=68, starfruit_ep000074_f0=68, starfruit_ep000111_f0=237 |
| mismatch_instruction | apple_ep000000_f0=68, apple_ep000037_f0=242, apple_ep000074_f0=89, apple_ep000111_f0=68, grapes_ep000000_f0=237, grapes_ep000037_f0=74, grapes_ep000074_f0=89, grapes_ep000111_f0=89, pear_ep000000_f0=142, pear_ep000037_f0=237, pear_ep000074_f0=68, pear_ep000111_f0=68, starfruit_ep000000_f0=39, starfruit_ep000037_f0=142, starfruit_ep000074_f0=142, starfruit_ep000111_f0=142 |
| point | apple_ep000000_f0=142, apple_ep000037_f0=142, apple_ep000074_f0=142, apple_ep000111_f0=142, grapes_ep000000_f0=142, grapes_ep000037_f0=238, grapes_ep000074_f0=142, grapes_ep000111_f0=142, pear_ep000000_f0=142, pear_ep000037_f0=142, pear_ep000074_f0=142, pear_ep000111_f0=142, starfruit_ep000000_f0=-, starfruit_ep000037_f0=142, starfruit_ep000074_f0=142, starfruit_ep000111_f0=142 |

## erflow / group `POS`

### Same frame: matched vs mismatched instruction

| frame | match bins | mismatch bins | common prefix | jaccard | match stopped at | mismatch stopped at |
|---|---:|---:|---:|---:|---:|---:|
| apple_ep000000_f0 | 32 | 32 | 2 | 0.27 | 32 | 32 |
| apple_ep000037_f0 | 32 | 32 | 4 | 0.33 | 32 | 32 |
| apple_ep000074_f0 | 32 | 32 | 0 | 0.13 | 32 | 32 |
| apple_ep000111_f0 | 32 | 32 | 1 | 0.19 | 32 | 32 |
| grapes_ep000000_f0 | 32 | 32 | 0 | 0.12 | 32 | 32 |
| grapes_ep000037_f0 | 32 | 32 | 2 | 0.18 | 32 | 32 |
| grapes_ep000074_f0 | 1 | 32 | 0 | 0.00 | 2 | 32 |
| grapes_ep000111_f0 | 32 | 32 | 0 | 0.16 | 32 | 32 |
| pear_ep000000_f0 | 32 | 32 | 0 | 0.14 | 32 | 32 |
| pear_ep000037_f0 | 32 | 32 | 4 | 0.33 | 32 | 32 |
| pear_ep000074_f0 | 1 | 32 | 0 | 0.00 | 2 | 32 |
| pear_ep000111_f0 | 32 | 32 | 0 | 0.09 | 32 | 32 |
| starfruit_ep000000_f0 | 1 | 1 | 0 | 0.00 | 2 | 2 |
| starfruit_ep000037_f0 | 32 | 0 | 0 | 0.00 | 32 | 1 |
| starfruit_ep000074_f0 | 32 | 32 | 0 | 0.18 | 32 | 32 |
| starfruit_ep000111_f0 | 32 | 32 | 1 | 0.27 | 32 | 32 |

### Stream length per prompt (a short stream means the model stopped early)

| prompt | runs | mean bins | runs with <=1 bin |
|---|---:|---:|---:|
| do_not_touch | 16 | 12.6 | 10 |
| future_region | 16 | 2.8 | 15 |
| match_instruction | 16 | 26.2 | 3 |
| mismatch_instruction | 16 | 28.1 | 2 |
| point | 16 | 6.1 | 13 |

### Same prompt across frames (does the opening bin depend on the scene?)

| prompt | first bins per frame |
|---|---|
| do_not_touch | apple_ep000000_f0=169, apple_ep000037_f0=169, apple_ep000074_f0=169, apple_ep000111_f0=169, grapes_ep000000_f0=169, grapes_ep000037_f0=169, grapes_ep000074_f0=39, grapes_ep000111_f0=169, pear_ep000000_f0=169, pear_ep000037_f0=169, pear_ep000074_f0=169, pear_ep000111_f0=169, starfruit_ep000000_f0=169, starfruit_ep000037_f0=169, starfruit_ep000074_f0=169, starfruit_ep000111_f0=169 |
| future_region | apple_ep000000_f0=2, apple_ep000037_f0=27, apple_ep000074_f0=184, apple_ep000111_f0=27, grapes_ep000000_f0=27, grapes_ep000037_f0=27, grapes_ep000074_f0=169, grapes_ep000111_f0=27, pear_ep000000_f0=2, pear_ep000037_f0=27, pear_ep000074_f0=27, pear_ep000111_f0=27, starfruit_ep000000_f0=169, starfruit_ep000037_f0=186, starfruit_ep000074_f0=-, starfruit_ep000111_f0=100 |
| match_instruction | apple_ep000000_f0=169, apple_ep000037_f0=169, apple_ep000074_f0=168, apple_ep000111_f0=168, grapes_ep000000_f0=233, grapes_ep000037_f0=169, grapes_ep000074_f0=228, grapes_ep000111_f0=168, pear_ep000000_f0=168, pear_ep000037_f0=169, pear_ep000074_f0=228, pear_ep000111_f0=168, starfruit_ep000000_f0=169, starfruit_ep000037_f0=169, starfruit_ep000074_f0=168, starfruit_ep000111_f0=169 |
| mismatch_instruction | apple_ep000000_f0=169, apple_ep000037_f0=169, apple_ep000074_f0=169, apple_ep000111_f0=168, grapes_ep000000_f0=168, grapes_ep000037_f0=169, grapes_ep000074_f0=169, grapes_ep000111_f0=169, pear_ep000000_f0=169, pear_ep000037_f0=169, pear_ep000074_f0=169, pear_ep000111_f0=169, starfruit_ep000000_f0=228, starfruit_ep000037_f0=-, starfruit_ep000074_f0=169, starfruit_ep000111_f0=169 |
| point | apple_ep000000_f0=166, apple_ep000037_f0=166, apple_ep000074_f0=3, apple_ep000111_f0=3, grapes_ep000000_f0=-, grapes_ep000037_f0=-, grapes_ep000074_f0=-, grapes_ep000111_f0=-, pear_ep000000_f0=3, pear_ep000037_f0=-, pear_ep000074_f0=-, pear_ep000111_f0=166, starfruit_ep000000_f0=-, starfruit_ep000037_f0=-, starfruit_ep000074_f0=-, starfruit_ep000111_f0=- |

## erflow / group `seg`

### Same frame: matched vs mismatched instruction

| frame | match bins | mismatch bins | common prefix | jaccard | match stopped at | mismatch stopped at |
|---|---:|---:|---:|---:|---:|---:|
| apple_ep000000_f0 | 32 | 32 | 0 | 0.41 | 32 | 32 |
| apple_ep000037_f0 | 32 | 32 | 0 | 0.38 | 32 | 32 |
| apple_ep000074_f0 | 1 | 32 | 1 | 0.06 | 2 | 32 |
| apple_ep000111_f0 | 32 | 32 | 0 | 0.53 | 32 | 32 |
| grapes_ep000000_f0 | 1 | 32 | 0 | 0.06 | 2 | 32 |
| grapes_ep000037_f0 | 32 | 32 | 0 | 0.35 | 32 | 32 |
| grapes_ep000074_f0 | 32 | 32 | 0 | 0.23 | 32 | 32 |
| grapes_ep000111_f0 | 32 | 32 | 0 | 0.35 | 32 | 32 |
| pear_ep000000_f0 | 32 | 32 | 0 | 0.29 | 32 | 32 |
| pear_ep000037_f0 | 32 | 32 | 0 | 0.25 | 32 | 32 |
| pear_ep000074_f0 | 32 | 32 | 0 | 0.38 | 32 | 32 |
| pear_ep000111_f0 | 1 | 1 | 1 | 1.00 | 2 | 2 |
| starfruit_ep000000_f0 | 32 | 32 | 0 | 0.32 | 32 | 32 |
| starfruit_ep000037_f0 | 32 | 32 | 0 | 0.20 | 32 | 32 |
| starfruit_ep000074_f0 | 32 | 1 | 1 | 0.05 | 32 | 2 |
| starfruit_ep000111_f0 | 32 | 1 | 0 | 0.00 | 32 | 2 |

### Stream length per prompt (a short stream means the model stopped early)

| prompt | runs | mean bins | runs with <=1 bin |
|---|---:|---:|---:|
| do_not_touch | 16 | 24.2 | 4 |
| future_region | 16 | 4.9 | 14 |
| match_instruction | 16 | 26.2 | 3 |
| mismatch_instruction | 16 | 26.2 | 3 |
| point | 16 | 8.0 | 12 |

### Same prompt across frames (does the opening bin depend on the scene?)

| prompt | first bins per frame |
|---|---|
| do_not_touch | apple_ep000000_f0=25, apple_ep000037_f0=25, apple_ep000074_f0=30, apple_ep000111_f0=5, grapes_ep000000_f0=25, grapes_ep000037_f0=30, grapes_ep000074_f0=25, grapes_ep000111_f0=25, pear_ep000000_f0=25, pear_ep000037_f0=25, pear_ep000074_f0=35, pear_ep000111_f0=25, starfruit_ep000000_f0=25, starfruit_ep000037_f0=25, starfruit_ep000074_f0=25, starfruit_ep000111_f0=188 |
| future_region | apple_ep000000_f0=27, apple_ep000037_f0=32, apple_ep000074_f0=32, apple_ep000111_f0=32, grapes_ep000000_f0=17, grapes_ep000037_f0=17, grapes_ep000074_f0=7, grapes_ep000111_f0=7, pear_ep000000_f0=17, pear_ep000037_f0=17, pear_ep000074_f0=17, pear_ep000111_f0=188, starfruit_ep000000_f0=32, starfruit_ep000037_f0=17, starfruit_ep000074_f0=32, starfruit_ep000111_f0=17 |
| match_instruction | apple_ep000000_f0=188, apple_ep000037_f0=188, apple_ep000074_f0=188, apple_ep000111_f0=1, grapes_ep000000_f0=24, grapes_ep000037_f0=30, grapes_ep000074_f0=188, grapes_ep000111_f0=5, pear_ep000000_f0=9, pear_ep000037_f0=27, pear_ep000074_f0=27, pear_ep000111_f0=5, starfruit_ep000000_f0=21, starfruit_ep000037_f0=5, starfruit_ep000074_f0=5, starfruit_ep000111_f0=21 |
| mismatch_instruction | apple_ep000000_f0=27, apple_ep000037_f0=5, apple_ep000074_f0=188, apple_ep000111_f0=27, grapes_ep000000_f0=30, grapes_ep000037_f0=188, grapes_ep000074_f0=5, grapes_ep000111_f0=188, pear_ep000000_f0=3, pear_ep000037_f0=188, pear_ep000074_f0=5, pear_ep000111_f0=5, starfruit_ep000000_f0=188, starfruit_ep000037_f0=30, starfruit_ep000074_f0=5, starfruit_ep000111_f0=5 |
| point | apple_ep000000_f0=27, apple_ep000037_f0=-, apple_ep000074_f0=193, apple_ep000111_f0=27, grapes_ep000000_f0=-, grapes_ep000037_f0=-, grapes_ep000074_f0=-, grapes_ep000111_f0=-, pear_ep000000_f0=-, pear_ep000037_f0=-, pear_ep000074_f0=-, pear_ep000111_f0=-, starfruit_ep000000_f0=-, starfruit_ep000037_f0=-, starfruit_ep000074_f0=-, starfruit_ep000111_f0=27 |

