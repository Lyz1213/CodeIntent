#!/usr/bin/env python3
# train_reasoning_code_lora.py - Fixed for multi-GPU training (with batch-dropping to avoid timeouts)

from __future__ import annotations
import argparse, logging, os, sys, datasets, torch, time, numpy as np
from collections.abc import Sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, DataLoader
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    __version__ as HF_VER,
)
from peft import LoraConfig, get_peft_model

try:
    from bitsandbytes.optim import AdamW8bit as AdamW8Bit
except ImportError:
    AdamW8Bit = None

import traceback
from itertools import chain

# ─ helpers ─────────────────────────────────────────────────────────────────────
def supports_attn_kw() -> bool:
    major, minor = map(int, HF_VER.split(".")[:2])
    return (major, minor) >= (4, 39)

def setup_logging(path):
    os.makedirs(path, exist_ok=True)
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(path, "training.log"), "a", "utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    log.addHandler(fh)
    return log

class DistributedTokenBatchSampler(Sampler[Sequence[int]]):
    def __init__(self, lens, max_tok, num_replicas=None, rank=None):
        # Get distributed info if not provided
        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()
        self.max_tok = max_tok

        # Global indices and lengths
        global_indices = np.arange(len(lens))
        global_lens = np.array(lens)

        # Sort by length for better load balancing
        sorted_idx = np.argsort(global_lens)[::-1]  # longest first
        self.sorted_lens = global_lens[sorted_idx]
        self.sorted_indices = global_indices[sorted_idx]

        # Create all batches first
        self.total_batches = self._create_batches()

        # ── Drop extra buckets so that total_batches is divisible by num_replicas ──
        n_total = len(self.total_batches)
        print('original total is ', self.total_batches)
        drop_cnt = n_total % self.num_replicas
        print('drop_cnt is ', drop_cnt)
        if drop_cnt != 0:
            self.total_batches = self.total_batches[: n_total - drop_cnt]

        # Distribute batches across GPUs
        self.batches = self.total_batches[self.rank :: self.num_replicas]
        self.num_batches = len(self.batches)
        # right after computing self.batches and self.num_batches:
        print(f"[Rank {self.rank}] After drop: total_batches={len(self.total_batches)}, "
              f"num_batches_per_rank={self.num_batches}")

        # Log distribution stats (using only “real” batches)
        if self.rank == 0:
            avg_batch_size = sum(len(b) for b in self.total_batches) / len(self.total_batches)

            logging.info(
                f"Total batches: {len(self.total_batches)}, "
                f"Per GPU batches: {self.num_batches}, "
                f"Avg batch size: {avg_batch_size:.1f}"
            )
        avg_batch_size = sum(len(b) for b in self.total_batches) / len(self.total_batches)

        print(
                f"Total batches: {len(self.total_batches)}, "
                f"Per GPU batches: {self.num_batches}, "
                f"Avg batch size: {avg_batch_size:.1f}"
            )

    def _create_batches(self):
        batches = []
        current_batch = []
        current_tokens = 0

        for idx, seq_len in zip(self.sorted_indices, self.sorted_lens):
            # Handle sequences longer than max_tok
            if seq_len > self.max_tok:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([int(idx)])
                continue

            # Start new batch if adding this sequence would exceed max_tok
            if current_tokens + seq_len > self.max_tok and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            # Add sequence to current batch
            current_batch.append(int(idx))
            current_tokens += seq_len

        # Add last batch if not empty
        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return self.num_batches


def pad_batch(feats, pad_id, lbl_pad=-100):
    for f in feats:
        if len(f["input_ids"]) != len(f["labels"]):
            raise ValueError("id/label len mismatch")
    ids = [torch.tensor(f["input_ids"]) for f in feats]
    lab = [torch.tensor(f["labels"]) for f in feats]
    ids = pad_sequence(ids, True, pad_id)
    lab = pad_sequence(lab, True, lbl_pad)
    attn = ids.ne(pad_id).long()
    return {"input_ids": ids, "attention_mask": attn, "labels": lab}


# ─ main ────────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_name", required=True)
    pa.add_argument("--tokenizer_dir", required=True)
    pa.add_argument("--train_dir", required=True)
    pa.add_argument("--output_dir", required=True)
    pa.add_argument("--lora_r", type=int, default=32)
    pa.add_argument("--lora_alpha", type=int, default=64)
    pa.add_argument("--lora_dropout", type=float, default=0.05)
    pa.add_argument("--batch_tokens", type=int, default=2048)
    pa.add_argument("--epochs", type=int, default=2)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--warmup_ratio", type=float, default=0.05)
    pa.add_argument("--flash_attn", action="store_true")
    pa.add_argument("--adam8bit", action="store_true")
    args = pa.parse_args()

    accelerator = Accelerator(log_with="tensorboard",
                              project_dir=args.output_dir)
    accelerator.even_batches = False

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logger = setup_logging(args.output_dir)
        logger.info(f"Run args: {vars(args)}")
    else:
        logger = logging.getLogger("train")

    try:
        # ─── Load tokenizer and model ───────────────────────────────────────
        tok = AutoTokenizer.from_pretrained(args.tokenizer_dir,
                                            trust_remote_code=True)
        for t in ["<intention>", "</intention>", "<code>", "</code>"]:
            assert tok.convert_tokens_to_ids(t) is not None, f"{t} missing!"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        use_flash = False
        attn_kw = {"attn_implementation": "flash_attention_2"} if use_flash else {}

        base = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **attn_kw
        )
        base.resize_token_embeddings(len(tok))

        special_ids = tok.convert_tokens_to_ids(
            ["<reasoning>", "</reasoning>",
             "<intention>", "</intention>", "<code>", "</code>"])
        with torch.no_grad():
            mean_vec = base.get_input_embeddings().weight[:-4].mean(dim=0, keepdim=True)
            base.get_input_embeddings().weight[special_ids] = mean_vec

        for p in base.get_input_embeddings().parameters():
            p.requires_grad = True

        # ### ADDED ①: tie lm_head to embeddings so they share storage ###
        base.lm_head.weight = base.get_input_embeddings().weight
        base.tie_weights()
        # ------------------------------------------------------------------

        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        # ─── Setup LoRA ───────────────────────────────────────────────────
        target = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        all_names = {n for n, _ in chain(base.named_modules(),
                                         base.named_parameters())}
        present = [t for t in target if any(n.endswith(t) for n in all_names)]
        missing = [t for t in target if t not in present]
        print("present :", present)
        print("missing :", missing if missing else "None ✓")

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target,
            modules_to_save=["embed_tokens", "lm_head"],  # ### ADDED ② ###
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        if accelerator.is_main_process:
            logger.info(model.print_trainable_parameters())

        # (dataset loading, training loop, saving … remain unchanged)
        # -----------------------------------------------------------------
        ds = datasets.load_from_disk(args.train_dir)
        lens = [len(r["input_ids"]) for r in ds]
        sampler = DistributedTokenBatchSampler(
            lens, max_tok=args.batch_tokens,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index
        )
        loader = DataLoader(
            ds, batch_sampler=sampler,
            collate_fn=lambda f: pad_batch(f, tok.pad_token_id),
            num_workers=4, pin_memory=True, shuffle=False
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        optim = (AdamW8Bit(trainable, lr=args.lr, betas=(0.9, 0.95))
                 if args.adam8bit and AdamW8Bit else
                 torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95)))
        tot_steps  = args.epochs * len(loader)
        warm_steps = max(1, int(tot_steps * args.warmup_ratio))
        sched = get_cosine_schedule_with_warmup(
            optim, num_warmup_steps=warm_steps, num_training_steps=tot_steps
        )

        model, optim, sched = accelerator.prepare(model, optim, sched)
        model.train(); global_step = 0

        for ep in range(args.epochs):
            if accelerator.is_main_process:
                logger.info(f"Starting epoch {ep+1}/{args.epochs}")
            for step, batch in enumerate(loader):
                with accelerator.accumulate(model):
                    loss = model(**batch).loss
                    accelerator.backward(loss)
                    optim.step(); sched.step(); optim.zero_grad()
                global_step += 1
                if step % 10 == 0 and accelerator.is_main_process:
                    logger.info(f"E{ep} S{step}/{len(loader)} Loss {loss.item():.4f}")

            accelerator.wait_for_everyone()
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{ep}"))
            if accelerator.is_main_process:
                logger.info(f"Epoch {ep} checkpoint saved")

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(args.output_dir)
            tok.save_pretrained(args.output_dir)
            logger.info("✓ Training complete and model saved.")

    except Exception:
        if accelerator.is_main_process:
            logger.exception("Training crashed")
        raise

if __name__ == "__main__":
    main()
