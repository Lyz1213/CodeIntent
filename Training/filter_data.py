#!/usr/bin/env python3
"""
filter_dataset.py  –  scan, print stats, filter by tail length
--------------------------------------------------------------
Outputs
  <out_dir>/filtered.json   (list of kept records)
  <out_dir>/stats.txt       (same table printed to console)
"""

from __future__ import annotations
import argparse, json, textwrap, numpy as np, os
from pathlib import Path
from transformers import AutoTokenizer
from tqdm.auto import tqdm

SECTION_START = "<reasoning>"
MAX_IMPORT_LINES = 50
ELLIPSIS_TEXT = "\n#...\n"

def split_import_block(src: str) -> tuple[str, str]:
    lines = src.splitlines(keepends=True)
    block, i = [], 0
    while i < len(lines) and len(block) < MAX_IMPORT_LINES:
        s = lines[i].lstrip()
        if not s.strip() or s.startswith(("import ", "from ",
                                          "#", "/*", '"""', "'''")):
            block.append(lines[i]); i += 1; continue
        break
    return "".join(block), "".join(lines[i:])

def build_prompt(left_ctx, sig, reasoning, intention, solution):
    return (f"{left_ctx}\n{sig}\n"
            f"<reasoning> {reasoning} </reasoning>\n"
            f"<intention> {intention} </intention>\n"
            f"<code> {solution} </code>")

def pretty(name, arr):
    return (f"{name:7}  min {arr.min():>6}  max {arr.max():>6}  "
            f"mean {arr.mean():>7.1f}  median {np.median(arr):>6.0f}  "
            f"p90 {np.percentile(arr,90):>6.0f}  "
            f"p95 {np.percentile(arr,95):>6.0f}  "
            f"p99 {np.percentile(arr,99):>6.0f}")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--json", required=True)
    pa.add_argument("--tokenizer", required=True)
    pa.add_argument("--out_dir", required=True)
    pa.add_argument("--max_len", type=int, default=4096)
    pa.add_argument("--margin", type=int, default=200,
                    help="safety margin below max_len")
    args = pa.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    start_r = tok.convert_tokens_to_ids(SECTION_START)

    raw = json.loads(Path(args.json).read_text())

    keep, tot, hd, tl = [], [], [], []
    for r in tqdm(raw, desc="scan"):
        ids_all = tok(build_prompt(r["left_context"], r["function_signature"],
                                   r["reasoning"], r["intention"], r["solution"]),
                       truncation=False)["input_ids"]

        tail_ids = tok(build_prompt("", r["function_signature"],
                                    r["reasoning"], r["intention"], r["solution"]),
                       truncation=False)["input_ids"]
        if start_r not in tail_ids:
            print('start r not tail_ids , the original data is ',build_prompt("", r["function_signature"],
                                    r["reasoning"], r["intention"], r["solution"]))
            continue
        if len(tail_ids) > (args.max_len - args.margin):
            continue  # drop monster sample

        cut = ids_all.index(start_r)
        keep.append(r)
        tot.append(len(ids_all)); hd.append(cut); tl.append(len(ids_all) - cut)

    stats_txt = (
        "\nToken length statistics (before truncation)\n"
        "-------------------------------------------\n" +
        pretty("TOTAL", np.array(tot)) + "\n" +
        pretty("HEAD",  np.array(hd)) + "\n" +
        pretty("TAIL",  np.array(tl)) + "\n"
    )
    print(stats_txt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(keep, ensure_ascii=False))
    (out_dir / "stats.txt").write_text(stats_txt)
    print(f"Kept {len(keep)} / {len(raw)} examples → {out_dir/'filtered.json'}")

if __name__ == "__main__":
    main()
