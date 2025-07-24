#!/usr/bin/env python3
"""
prepare_tokenizer.py
────────────────────
Add six section tags to an existing tokenizer *and*
guarantee that a padding token exists.

Usage
-----
python prepare_tokenizer.py \
    --model  codellama/CodeLlama-7b-hf \
    --out_dir tokenizer_reasoning_code
"""
import argparse
from transformers import AutoTokenizer

SECTION_TAGS = [
    "<reasoning>", "</reasoning>",
    "<intention>", "</intention>",
    "<code>",      "</code>",
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="base tokenizer to load, e.g. codellama/CodeLlama-7b-hf")
    ap.add_argument("--out_dir", required=True,
                    help="directory to save the augmented tokenizer")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 1) add section tags as additional_special_tokens
    added = tok.add_special_tokens({"additional_special_tokens": SECTION_TAGS})
    print(f"Added {added} special tokens")

    # 2) make sure a pad token exists (reuse eos if absent)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"Set pad_token to eos_token: {tok.pad_token!r}")

    tok.save_pretrained(args.out_dir)
    print(f"Tokenizer with tags + pad saved to {args.out_dir}")

if __name__ == "__main__":
    main()
