#!/usr/bin/env python3
"""Compare structured token streams produced under vocabulary restriction.

    python3 analyze_structured.py --dir .../restricted_full --out report.md

Free generation never emits the structured vocabulary, so the streams here come
from generation restricted to one group plus EOS. Two questions are answered:

  1. Same image, different instruction: how far do the streams share a prefix,
     and where do they diverge? A shared prefix with a divergent tail means the
     scene fixes the start of the motion and the instruction changes the rest.
  2. Does the stream length carry meaning? An instruction naming an object that
     is absent should stop early if the model checks object presence.
"""

import argparse
import collections
import glob
import json
import pathlib
import re

GROUP_RE = re.compile(r"<\|(POS|ROT|EEF|LOW|HAND)_(\d+)\|>|<seg(\d+)>")


def bins(row):
    """Structured bin indices in the generated stream, EOS excluded."""
    text = row["text"].split("<|im_end|>")[0]
    return [int(m.group(2) or m.group(3)) for m in GROUP_RE.finditer(text)]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / (len(sa | sb) or 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(f"{args.dir}/*_results.jsonl")):
        tag = pathlib.Path(f).name.replace("_results.jsonl", "")
        for line in open(f):
            r = json.loads(line)
            r["tag"] = tag
            r["bins"] = bins(r)
            rows.append(r)

    by_group = collections.defaultdict(list)
    for r in rows:
        model, group = r["tag"].split("_")[:2]
        r["model"] = model
        by_group[(model, group)].append(r)

    lines = ["# Structured token streams under vocabulary restriction", "",
             "`n_bins` counts structured tokens before EOS; `stopped at` is the total number of "
             "generated tokens (1-2 means the model emitted EOS almost immediately).", ""]
    for (model, group), group_rows in sorted(by_group.items()):
        lines += [f"## {model} / group `{group}`", ""]

        # 1. Same frame, matched vs mismatched instruction
        lines += ["### Same frame: matched vs mismatched instruction", "",
                  "| frame | match bins | mismatch bins | common prefix | jaccard | match stopped at | mismatch stopped at |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        pairs = collections.defaultdict(dict)
        for r in group_rows:
            if r["prompt_id"] in ("match_instruction", "mismatch_instruction"):
                pairs[r["sample"]][r["prompt_id"]] = r
        for sample, pair in sorted(pairs.items()):
            a, b = pair.get("match_instruction"), pair.get("mismatch_instruction")
            if not a or not b:
                continue
            lines.append(
                f"| {sample} | {len(a['bins'])} | {len(b['bins'])} | {common_prefix(a['bins'], b['bins'])} "
                f"| {jaccard(a['bins'], b['bins']):.2f} | {a['n_new_tokens']} | {b['n_new_tokens']} |"
            )

        # 2. Stream length per prompt
        lines += ["", "### Stream length per prompt (a short stream means the model stopped early)", "",
                  "| prompt | runs | mean bins | runs with <=1 bin |", "|---|---:|---:|---:|"]
        by_prompt = collections.defaultdict(list)
        for r in group_rows:
            by_prompt[r["prompt_id"]].append(r)
        for pid, rs in sorted(by_prompt.items()):
            lens = [len(r["bins"]) for r in rs]
            short = sum(1 for n in lens if n <= 1)
            lines.append(f"| {pid} | {len(rs)} | {sum(lens) / len(lens):.1f} | {short} |")

        # 3. Same prompt across frames: is the opening bin scene-dependent?
        lines += ["", "### Same prompt across frames (does the opening bin depend on the scene?)", "",
                  "| prompt | first bins per frame |", "|---|---|"]
        for pid, rs in sorted(by_prompt.items()):
            firsts = ", ".join(f"{r['sample']}={r['bins'][0] if r['bins'] else '-'}" for r in sorted(rs, key=lambda x: x["sample"]))
            lines.append(f"| {pid} | {firsts} |")
        lines.append("")

    pathlib.Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} from {len(rows)} rows")


if __name__ == "__main__":
    main()
