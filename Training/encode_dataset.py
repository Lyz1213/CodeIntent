#!/usr/bin/env python3
"""
encode_dataset.py  –  truncate, mask, save Arrow + 15-sample preview
--------------------------------------------------------------------
Input   --json <filtered.json>   (produced by filter_dataset.py)
Output  <encode_dir>/arrow_dataset
        <encode_dir>/sample_check.txt
"""

from __future__ import annotations
import argparse, json, random, textwrap
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

SECTION_START = "<reasoning>"
SECTION_END = "</code>"
ELLIPSIS_TEXT = "\n#...\n"
MAX_IMPORT_LINES = 50
SAMPLE_K = 15

def split_import_block(src):
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
            f"<reasoning>\n{reasoning}\n</reasoning>\n"
            f"<intention>\n{intention}\n</intention>\n"
            f"<code>\n{solution}\n</code>")



def encode_factory(tok, max_len, start_r, end_c, ellipsis_ids):
    def encode(rec):
        bos_id, eos_id = tok.bos_token_id, tok.eos_token_id
        imp_blk, rest_ctx = split_import_block(rec["left_context"])
        tail_ids = tok(build_prompt("", rec["function_signature"],
                                    rec["reasoning"], rec["intention"],
                                    rec["solution"]), truncation=False, add_special_tokens=False)["input_ids"]
        remain = max_len - len(tail_ids) - 2
        assert(remain > (len(ellipsis_ids) + 20))
        imp_ids = tok(imp_blk, truncation=False,add_special_tokens=False)["input_ids"][:(remain - len(ellipsis_ids) - 20)]
        remain -= len(imp_ids)

        rest_ids = tok(rest_ctx, truncation=False,add_special_tokens=False)["input_ids"]
        if len(rest_ids) > remain and remain > len(ellipsis_ids):
            head_ids = imp_ids + ellipsis_ids + rest_ids[-(remain - len(ellipsis_ids)):]
        else:
            head_ids = imp_ids + rest_ids[-remain:]
        assert(end_c in tail_ids)
        ids = (head_ids + tail_ids)[:max_len - 2]
        ids = [bos_id] + ids + [eos_id]
        if (len(ids) > max_len):
            print('too lang with length ', len(head_ids), len(tail_ids))
            ids = ids[-max_len:]
        if start_r not in ids:
            print(f"<reasoning> is not in ids\n\n{ids[-100:]}\n\n{tok.decode(tail_ids[-100:], skip_special_tokens=False)}")
        assert(end_c in ids)
        labels = ids.copy()
        labels[: ids.index(start_r)] = [-100] * ids.index(start_r)
        return {"input_ids": ids, "labels": labels}
    return encode

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--json", required=True)
    pa.add_argument("--tokenizer", required=True)
    pa.add_argument("--encode_dir", required=True)
    pa.add_argument("--max_len", type=int, default=4096)
    args = pa.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    start_r = tok.convert_tokens_to_ids(SECTION_START)
    end_c = tok.convert_tokens_to_ids(SECTION_END)
    ellipsis_ids = tok(ELLIPSIS_TEXT, add_special_tokens=False)["input_ids"]

    data = json.loads(Path(args.json).read_text())
    ds = Dataset.from_list(data)
    ds = ds.map(
        encode_factory(tok, args.max_len, start_r, end_c,ellipsis_ids),
        remove_columns=ds.column_names, num_proc=4)

    enc_dir = Path(args.encode_dir)
    enc_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(enc_dir / "arrow_dataset")
    print(f"Arrow dataset saved → {enc_dir/'arrow_dataset'}  ({len(ds)} rows)")

    # -------- sample preview -------------------------------------------
    idxs = random.sample(range(len(ds)), min(SAMPLE_K, len(ds)))
    with (enc_dir / "sample_check.txt").open("w", encoding="utf-8") as fh:
        for idx in idxs:
            rec   = ds[idx]
            ids   = rec["input_ids"]
            masks = rec["labels"]
            r_pos = ids.index(start_r)
            slc   = ids[max(0, r_pos - 50):]
            txt   = tok.decode(slc, skip_special_tokens=False)
            fh.write(f"### SAMPLE idx={idx}\nids  : {slc}\nmask : {masks[max(0,r_pos-50):]}\ntext :\n{txt}\n\n")
    print(f"Wrote {len(idxs)} samples → {enc_dir/'sample_check.txt'}")

if __name__ == "__main__":
    main()
