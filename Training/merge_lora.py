#!/usr/bin/env python3
# coding: utf-8
"""
Merge a LoRA adapter into a base model *and* verify that the four
special-token rows are copied correctly.

Example
-------
python merge_lora.py \
    --base  codellama/CodeLlama-13b-Python-hf \
    --lora  ./checkpoint_final            \
    --tokenizer ./checkpoint_final        \
    --output_dir ./merged_debug
"""
import argparse, os, torch, json, difflib
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file as load_st

SPECIAL_TOKENS = ["<intention>", "</intention>", "<code>", "</code>", "</reasoning>", "<reasoning>"]

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--base",      required=True)
ap.add_argument("--lora",      required=True, help="Adapter *folder* (has adapter_model.safetensors)")
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--output_dir",required=True)
ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
args = ap.parse_args()
dtype = {"float16":torch.float16, "bfloat16":torch.bfloat16, "float32":torch.float32}[args.dtype]

# ---------- load tokenizer ----------
print(f"\n→ Loading tokenizer   : {args.tokenizer}")
tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
missing = [t for t in SPECIAL_TOKENS if tok.convert_tokens_to_ids(t) is None]
if missing:
    raise ValueError(f"Tokenizer does not know these tokens: {missing}")

ids = tok.convert_tokens_to_ids(SPECIAL_TOKENS)
print("   token → id mapping :", dict(zip(SPECIAL_TOKENS, ids)))

# ---------- base weights BEFORE merge ----------
print(f"\n→ Loading base model  : {args.base}")
base = AutoModelForCausalLM.from_pretrained(args.base,
                                            torch_dtype=dtype,
                                            trust_remote_code=True)
base.resize_token_embeddings(len(tok))

with torch.no_grad():
    emb_base = base.get_input_embeddings().weight[ids].clone()

print("   ‖emb_row‖ before    :", emb_base.norm(dim=1).tolist())

# ---------- mount adapter ----------
print(f"\n→ Loading adapter     : {args.lora}")
model = PeftModel.from_pretrained(base, args.lora)

# ---------- check after adapter (still un-merged) ----------
with torch.no_grad():
    emb_peft = model.get_input_embeddings().weight[ids].clone()
print("   ‖emb_row‖ after PEFT:", emb_peft.norm(dim=1).tolist())

# quick diff
delta_peft = (emb_peft - emb_base).abs().max().item()
print(f"   max |Δ| vs. base    : {delta_peft:.6f}")

if delta_peft < 1e-6:
    print("   ⚠  rows unchanged → adapter did NOT inject trained embeddings!")

# ---------- merge ----------
print("\n→ Merging LoRA into base …")
merged = model.merge_and_unload()

with torch.no_grad():
    emb_final = merged.get_input_embeddings().weight[ids].clone()
print("   ‖emb_row‖ after merge:", emb_final.norm(dim=1).tolist())
delta_merge = (emb_final - emb_peft).abs().max().item()
print(f"   max |Δ| vs. PEFT    : {delta_merge:.6f}")

# ---------- save ----------
os.makedirs(args.output_dir, exist_ok=True)
merged.save_pretrained(args.output_dir)
tok.save_pretrained(args.output_dir)
print(f"\n✓ merged model saved to {args.output_dir}")
