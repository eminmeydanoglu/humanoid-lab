#!/usr/bin/env python3
"""Turn probe results into the comparison tables the experiment needs.

    python3 analyze_probe.py --results <tag>_results.jsonl --out report.md

Answers two questions:
  1. Elicitation: which prompts make the model emit Unitree's extended
     vocabulary (POS/ROT/EEF/LOW/HAND/seg) at all?
  2. Prompt sensitivity: on the *same* image, does swapping the target object in
     the instruction change those token streams? Matched and mismatched
     instructions are compared per frame, so image content is held constant.
"""

import argparse
import collections
import json
import pathlib
import re

GROUPS = ["POS", "ROT", "EEF", "LOW", "HAND", "seg", "SEP_VQ", "robot_state", "seg_begin", "seg_end"]
PATTERNS = [
    (re.compile(r"^<\|POS_(\d+)\|>$"), "POS"),
    (re.compile(r"^<\|ROT_(\d+)\|>$"), "ROT"),
    (re.compile(r"^<\|EEF_(\d+)\|>$"), "EEF"),
    (re.compile(r"^<\|LOW_(\d+)\|>$"), "LOW"),
    (re.compile(r"^<\|HAND_(\d+)\|>$"), "HAND"),
    (re.compile(r"^<seg(\d+)>$"), "seg"),
    (re.compile(r"^<\|SEP_VQ\|>$"), "SEP_VQ"),
    (re.compile(r"^<\|robot_state\|>$"), "robot_state"),
    (re.compile(r"^<seg_begin>$"), "seg_begin"),
    (re.compile(r"^<seg_end>$"), "seg_end"),
]


def group_indices(token_ids, tokenizer_names):
    """Map group name -> list of bin indices emitted (e.g. POS -> [184, 20])."""
    out = collections.defaultdict(list)
    for i in token_ids:
        name = tokenizer_names.get(i)
        if name is None:
            continue
        for pattern, group in PATTERNS:
            m = pattern.match(name)
            if m:
                out[group].append(int(m.group(1)) if m.groups() else -1)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results)]
    lines = [f"# Probe report: {pathlib.Path(args.results).name}", ""]

    # 1. Elicitation per prompt
    elic = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        key = r["prompt_id"]
        elic[key]["runs"] += 1
        if r["special_group_counts"]:
            elic[key]["with_special_tokens"] += 1
        for g in r["special_group_counts"]:
            elic[key][g] += 1
        if r["hit_max_new_tokens"]:
            elic[key]["truncated_at_max"] += 1
    lines += ["## 1. Which prompts elicit extended-vocabulary tokens", "",
              "| prompt | family | runs | with special tokens | " + " | ".join(GROUPS) + " | truncated |",
              "|---|---|---:|---:|" + "---:|" * len(GROUPS) + "---:|"]
    families = {r["prompt_id"]: r["prompt_family"] for r in rows}
    for pid, c in sorted(elic.items(), key=lambda kv: -kv[1]["with_special_tokens"]):
        cols = " | ".join(str(c[g]) if c[g] else "" for g in GROUPS)
        lines.append(f"| {pid} | {families[pid]} | {c['runs']} | {c['with_special_tokens']} | {cols} | {c['truncated_at_max']} |")

    # 2. Same image, matched vs mismatched instruction
    lines += ["", "## 2. Same image, matched vs mismatched instruction", ""]
    by_sample = collections.defaultdict(dict)
    for r in rows:
        if r["prompt_id"] in ("match_instruction", "mismatch_instruction"):
            by_sample[r["sample"]][r["prompt_id"]] = r
    lines += ["| frame | prompt | n tokens | groups | text (first 200 chars) |", "|---|---|---:|---|---|"]
    for sample, pair in sorted(by_sample.items()):
        for pid in ("match_instruction", "mismatch_instruction"):
            r = pair.get(pid)
            if not r:
                continue
            text = r["text"].replace("|", "\\|").replace("\n", " ")[:200]
            lines.append(f"| {sample} | {r['prompt']} | {r['n_new_tokens']} | {r['special_group_counts'] or '-'} | {text} |")

    # 3. Token-stream divergence when only the object word changes
    if any("token_ids" in r for r in rows):
        lines += ["", "## 3. Token-stream divergence (matched vs mismatched, same image)", "",
                  "`common_prefix` is how many leading generated tokens are identical; "
                  "`shared` is the Jaccard overlap of emitted token ids.", "",
                  "| frame | common prefix | jaccard | match token count | mismatch token count |", "|---|---:|---:|---:|---:|"]
        for sample, pair in sorted(by_sample.items()):
            a, b = pair.get("match_instruction"), pair.get("mismatch_instruction")
            if not (a and b) or "token_ids" not in a or "token_ids" not in b:
                continue
            ta, tb = a["token_ids"], b["token_ids"]
            prefix = 0
            for x, y in zip(ta, tb):
                if x != y:
                    break
                prefix += 1
            inter = len(set(ta) & set(tb))
            union = len(set(ta) | set(tb)) or 1
            lines.append(f"| {sample} | {prefix} | {inter / union:.2f} | {len(ta)} | {len(tb)} |")

    pathlib.Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
