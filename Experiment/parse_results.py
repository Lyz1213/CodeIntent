#!/usr/bin/env python3
"""
parse_completions.py

Strict format-check + parse of vLLM/HF completions.jsonl.

Usage:
  python parse_completions.py \
    --input  completions.jsonl \
    --output parsed.jsonl
"""
import argparse, json, re
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple, Optional
import os

# Define templates: list of tags (must match exactly, in order)


Format = {
    "reasoning":{"sequence":["</reasoning>","<intention>", "</intention>"],"position":["<intention>", "</intention>"]},
    "reasoning_full":{"sequence":["<reasoning>","</reasoning>","<intention>","</intention>"], "position":["<intention>", "</intention>"]},
    "code":{"sequence":["<code>"],"position":["<code>",None]},
    "all":{"sequence":["</reasoning>","<intention>", "</intention>", "<code>"],"position":["<code>", None]},
    "all_full":{"sequence":["<reasoning>","</reasoning>","<intention>", "</intention>", "<code>"],"position":["<code>", None]}
}



def find_all_tokens(s: str, tokens: List[str]) -> List[str]:
    """
    Return the sequence of all tokens from `tokens` found in `s`, in
    left-to-right order (allowing duplicates).
    """
    occ = []
    for pattern in tokens:
        # we scan with finditer to catch duplicates
        for m in re.finditer(re.escape(pattern), s):
            occ.append((m.start(), pattern))
    occ.sort(key=lambda x: x[0])
    return [tok for _, tok in occ]

def check_format(s: str, tokens: List[str]) -> Tuple[bool, str]:
    """
    Check that `s` contains exactly the sequence `tokens`, no extras.
    Returns (ok, reason). Reason is "" if ok, else one of:
      - "missing": fewer than len(tokens)
      - "extra":   correct first len(tokens) but more afterwards
      - "order":   same count but wrong sequence
    """
    occ = find_all_tokens(s, tokens)
    if occ == tokens:
        return True, ""
    if len(occ) < len(tokens):
        return False, "missing"
    if occ[:len(tokens)] == tokens and len(occ) > len(tokens):
        return False, "extra"
    return False, "order"

def parse_between(s, start, end) -> str:
    """
    Extract substring of `s` after `start` up to `end`.
    If `end` is None or not found, returns to end of `s`.
    """
    if start is None:
        i = 0
    else:
        i = s.find(start)
        if i < 0:
            return ""
        i += len(start)

    if end is None:
        return s[i:].strip()
    j = s.find(end, i)
    return (s[i:j].strip() if j>=0 else s[i:].strip())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True,
                   help="Path to completions.jsonl")
    p.add_argument("--output", required=True,
                   help="Path for parsed.jsonl")
    p.add_argument("--format", choices=["all","all_full","code","reasoning","reasoning_full"],default="all")
    p.add_argument("--saveformat", default="jsonl")
    args = p.parse_args()

    stats = {"total":0, "success":0,
             "missing":0, "extra":0, "order":0, "nomatch":0}

    # out_path = Path(args.output)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.saveformat=="jsonl":
        with open(os.path.join(args.input, "completion.jsonl"), "r", encoding="utf-8") as fin, \
                open(os.path.join(args.output, "parsed_results.jsonl"), "w", encoding="utf-8") as fout:

            for line in tqdm(fin, desc="Parsing"):
                stats["total"] += 1
                js = json.loads(line)
                # take first completion
                raw = ""
                try:
                    raw = js["completions"][0]
                    head, sep, tail = raw.partition("</code>")
                    if sep:
                        raw = head + sep
                except Exception:
                    stats["nomatch"] += 1
                    continue

                # try each template
                sequence = Format[args.format]["sequence"]
                position = Format[args.format]["position"]
                ok, reason = check_format(raw, sequence)
                if not ok:
                    stats[reason] += 1
                    continue

                # success
                stats["success"] += 1

                sections = parse_between(raw, position[0], position[1])

                # rewrite JSON object
                js["ori_completion"] = raw
                js["completions"] = sections
                fout.write(json.dumps(js, ensure_ascii=False) + "\n")
    elif args.saveformat=="json":
        datas = []
        with open(os.path.join(args.input, "completion.jsonl"), "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc="Parsing"):
                stats["total"] += 1
                js = json.loads(line)
                # take first completion
                raw = ""
                try:
                    raw = js["completions"][0]
                    head, sep, tail = raw.partition("</code>")
                    if sep:
                        raw = head + sep
                except Exception:
                    stats["nomatch"] += 1
                    continue

                # try each template
                sequence = Format[args.format]["sequence"]
                position = Format[args.format]["position"]
                position = ["<intention>", "</intention>"]

                ok, reason = check_format(raw, sequence)
                if not ok:
                    stats[reason] += 1
                    continue

                # success
                stats["success"] += 1

                sections = parse_between(raw, position[0], position[1])

                # rewrite JSON object
                js["ori_completion"] = raw
                js["completions"] = sections
                datas.append(js)
        with open(args.output, "w", encoding="utf-8") as fout:
            json.dump(datas, fout, ensure_ascii=False, indent=4)
    else:
        print('invalid saveformat')


    # print stats
    print("=== Parsing Summary ===")
    for k in ["total","success","missing","extra","order","nomatch"]:
        print(f"{k:8} : {stats[k]}")
    print(f"Output written to {args.output}")

if __name__ == "__main__":
    main()
