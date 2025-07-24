#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference with a trained code model (local or HF hub).

Example
-------
python eval_finetune.py \
  --model /path/to/merged_model  \
  --output_dir results/run1      \
  --moda greedy                  \
  --max_tokens 1024
"""
from pathlib import Path
from argparse import ArgumentParser
import os, json, sys
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# ────────────────────── arg parser ─────────────────────────
def parse_args():
    p = ArgumentParser()
    p.add_argument("--setting", type=str, default="local-completion")
    p.add_argument("--model",   type=str, required=True,
                   help="Either an HF model name or *path to your saved model*")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--moda",    type=str, choices=["greedy","diverse","sampling"],
                   default="greedy")
    p.add_argument("--max_tokens", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--level",   type=str, default="first-level")
    p.add_argument("--lang",    type=str, default="python")
    p.add_argument("--stop", type=str, default="</code>")
    p.add_argument("--cceval", action="store_true")
    return p.parse_args()

# ────────────────────── model loader ───────────────────────
def load_model(model_name_or_path: str):
    # local dir?
    if Path(model_name_or_path).is_dir():
        model_dir = model_name_or_path
        print(f"Loading local model at {model_dir}")
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        max_model_len = getattr(cfg, "max_position_embeddings",
                                getattr(cfg, "n_ctx", 4096))
        if max_model_len > 16384:
            max_model_len = 16384
        print('max_len is ', max_model_len)
    else:
        # original hub selection logic
        if model_name_or_path.startswith("deepseek-lite"):
            model_dir = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
            max_model_len = 18384
        elif model_name_or_path.startswith("deepseek-base"):
            model_dir = "deepseek-ai/DeepSeek-Coder-V2-Instruct"
            max_model_len = 15384
        elif model_name_or_path.startswith("CodeLlama-7b"):
            model_dir = "codellama/CodeLlama-7b-Python-hf"
            max_model_len = 15384
        elif model_name_or_path.startswith("CodeLlama-13b"):
            model_dir = "codellama/CodeLlama-13b-Python-hf"
            max_model_len = 15384
        elif model_name_or_path.startswith("starcoder2-7b"):
            model_dir = "bigcode/starcoder2-7b"
            max_model_len = 15384
        elif model_name_or_path.startswith("starcoder2-15b"):
            model_dir = "bigcode/starcoder2-15b-instruct-v0.1"
            max_model_len = 15384
        elif model_name_or_path == "gemma-7b":
            model_dir = "google/gemma-7b"
            max_model_len = 8192
        elif model_name_or_path == "qwen1.5-7b":
            model_dir = "Qwen/Qwen1.5-7B"
            max_model_len = 32768
        else:
            raise ValueError(f"Unknown model key {model_name_or_path}")

    model = LLM(model=model_dir,
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
                tensor_parallel_size=1,
                max_model_len=max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # ─ DEBUG ─ print special-token ids & embedding norms (if HF weights exist)
    special = ["<intention>", "</intention>", "<code>", "</code>","<reasoning>","</reasoning>"]
    ids = [tokenizer.convert_tokens_to_ids(t) for t in special]
    print("\n[debug] special-token ids :", dict(zip(special, ids)))
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype="auto")
        emb_norms = hf_model.get_input_embeddings().weight[ids].norm(dim=1).tolist()
        #print("[debug] embedding ‖row‖  :", emb_norms)
        del hf_model
    except Exception as e:
        print("[debug] could not load HF weights for norm check –", str(e))
    # ─ DEBUG end ─

    return model, tokenizer

# ────────────────────── helper fns ─────────────────────────
def retrieve_context_length(model_name: str, tokenizer):
    if Path(model_name).is_dir():
        return min(tokenizer.model_max_length, 16384)
    if model_name.startswith(("deepseek", "CodeLlama")):
        return 16384
    if model_name.startswith("starcoder2"):
        return 15384
    if model_name.startswith("gemma-7b"):
        return 8192
    if model_name.startswith("qwen1.5-7b"):
        return 32768
    return 4096

def retrieve_special_ids(model_name: str, tokenizer):
    # local model branch
    if Path(model_name).is_dir():
        return tokenizer.bos_token_id, None, None, None

    if model_name.startswith("qwen1.5-7b"):
        bos_id = 151643
    else:
        bos_id = tokenizer.bos_token_id

    if model_name.startswith("deepseek"):
        prefix_id = tokenizer.convert_tokens_to_ids("<｜fim▁begin｜>")
        middle_id = tokenizer.convert_tokens_to_ids("<｜fim▁hole｜>")
        suffix_id = tokenizer.convert_tokens_to_ids("<｜fim▁end｜>")
    elif model_name.startswith("codellama"):
        prefix_id = tokenizer.prefix_id
        middle_id = tokenizer.middle_id
        suffix_id = tokenizer.suffix_id
    elif model_name.startswith("starcoder2"):
        prefix_id = tokenizer.convert_tokens_to_ids("<fim_prefix>")
        middle_id = tokenizer.convert_tokens_to_ids("<fim_middle>")
        suffix_id = tokenizer.convert_tokens_to_ids("<fim_suffix>")
    else:
        prefix_id = middle_id = suffix_id = None

    return bos_id, prefix_id, middle_id, suffix_id

def produce_prompt(args, js, tokenizer):
    context_window = retrieve_context_length(args.model, tokenizer)
    input_code = js['infer_prompt']
    input_ids  = tokenizer(input_code)["input_ids"]
    bos_id, *_ = retrieve_special_ids(args.model, tokenizer)
    prompt_ids = [bos_id] + input_ids
    if len(prompt_ids) > context_window:
        #print(f"[warn] prompt {js['name']} truncated ({len(prompt_ids)} > {context_window})")
        prompt_ids = prompt_ids[-context_window:]

    # ─ DEBUG ─ show last part of prompt
    tail_txt = tokenizer.decode(prompt_ids[-50:], skip_special_tokens=False).replace("\n", "\\n")
    #print(f"[debug] prompt tail : …{tail_txt}")
    #print(f"[debug] prompt tail ids       : {prompt_ids[-20:]}")
    # ─ DEBUG end ─
    return prompt_ids

# ─────────────────── evaluation loop ───────────────────────
def inference(args, model, tokenizer, prompt_file, output_dir, sampling_params):
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "completion.jsonl"
    with open(prompt_file) as f_in, open(out_path, "w") as f_out:
        for line in tqdm(f_in, desc="generating"):
            js = json.loads(line)
            prompt_ids = produce_prompt(args, js, tokenizer)

            results = model.generate(
                prompt_token_ids=[prompt_ids],
                sampling_params=sampling_params,
                use_tqdm=False)

            completions = []
            #out = results[0].outputs[0]
            #print("[debug] top-25 at first step:")
            # for tok, lp in zip(out.logprob_top_tokens[0], out.logprob_top_probs[0]):
            #     print(f"{tok!r:<20} {lp:>8.3f}")
            for result in results:
                for out in result.outputs:
                    # ─ DEBUG ─ first 30 token IDs & first 120 chars
                    #print(f"[debug] gen ids   : {out.token_ids[:30]}")
                    # print(f"[debug] gen text  : {out.text[:120].replace(chr(10),'\n')}")
                    # ─ DEBUG end ─
                    completions.append(out.text)

            js["completions"] = completions
            f_out.write(json.dumps(js, ensure_ascii=False) + "\n")
            f_out.flush()

# ───────────────────────── main ────────────────────────────
def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)
    print("Model & tokenizer loaded.")

    if args.stop == "</s>":
        stoplist = ["</s>"]
    elif args.stop == "</intention>":
        stoplist = ["</intention>", "</s>"]
    else:
        stoplist = ["</s>", "</code>"]
    print("stoplist is ", stoplist)

    if args.moda == 'greedy':
        sp = SamplingParams(
            temperature=0.2,
            repetition_penalty=1.05,  # small nudge to let rare tokens through
            max_tokens=args.max_tokens,
            stop=stoplist,
            n=1,
            skip_special_tokens=False
        )
    elif args.moda == 'diverse':
        sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_tokens, n=5)
    elif args.moda == 'sampling':
        sp = SamplingParams(temperature=0.4, top_p=0.95,
                            repetition_penalty=1.05,
                            stop=stoplist, max_tokens=args.max_tokens,
                            n=5, skip_special_tokens=False)
    else:
        raise ValueError("Invalid --moda")
    if args.lang == "python":
        if args.cceval:
            prompt_file = f'./cceval_prompts/{args.level}_prompts.jsonl'
        else:
            prompt_file = f'./prompts/{args.level}_prompts.jsonl' \

    else:
        f'./javaprompts/{args.level}_prompts.jsonl'

    inference(args, model, tokenizer, prompt_file, args.output_dir, sp)

if __name__ == "__main__":
    main()
