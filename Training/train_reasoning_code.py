#!/usr/bin/env python3
"""
train_reasoning_code.py
────────────────────────────────────────────────────────────
• TokenBatchSampler (exact __len__)
• Padded-label collator
• Optional Flash-Attention 2  (--flash_attn)
• Optional AdamW8bit          (--adam8bit)
• Rank-0 logging:
      training.log   -> INFO + tracebacks
      stdout.txt     -> redirected stdout
      stderr.txt     -> redirected stderr
"""
from __future__ import annotations
import argparse, logging, os, sys, datasets, torch
from collections.abc import Sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, DataLoader
from accelerate import Accelerator
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup, __version__ as HF_VER,
)

try:
    from bitsandbytes.optim import AdamW8bit as AdamW8Bit
except ImportError:
    AdamW8Bit = None  # fallback later

# ───────────────────────── utilities ─────────────────────────
def supports_attn_kw() -> bool:
    major, minor = map(int, HF_VER.split(".")[:2])
    return (major, minor) >= (4, 39)

def setup_logging(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training.log")
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt); fh.flush = fh.stream.flush
    logger.addHandler(fh)
    return logger

def redirect_stdio(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = open(os.path.join(save_dir, "stdout.txt"), "a", buffering=1)
    sys.stderr = open(os.path.join(save_dir, "stderr.txt"), "a", buffering=1)
def freeze_bottom_pct(model, pct: float = 0.30, freeze_embed: bool = True):
    """
    Args
    ----
    model : AutoModelForCausalLM
    pct   : fraction (0-1) of lowest layers to freeze
    freeze_embed : also freeze the token + positional embeddings
    """
    # 1) locate the stack of transformer blocks
    try:                                  # Llama / CodeLlama / Mistral
        blocks = model.model.layers
    except AttributeError:                # Falcon / GPT-NeoX style
        blocks = model.transformer.h

    n_total   = len(blocks)
    n_freeze  = int(n_total * pct + 0.5)          # round to nearest
    print(f"Freezing {n_freeze}/{n_total} layers ({pct:.0%})")

    for i, layer in enumerate(blocks):
        if i < n_freeze:                          # bottom = early indices
            layer.requires_grad_(False)           # in-place recursion

    if freeze_embed:
        if hasattr(model, "embed_tokens"):        # Llama-family
            model.embed_tokens.requires_grad_(False)
        elif hasattr(model, "transformer"):       # GPT-NeoX style
            model.transformer.wte.requires_grad_(False)

    # 2) sanity log
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f} M / {total/1e6:.1f} M "
          f"({trainable/total:.1%})")
# ─────────────────────  data helpers  ────────────────────────
# class TokenBatchSampler(Sampler[Sequence[int]]):
#     def __init__(self, lengths: list[int], max_tokens: int):
#         self.batches: list[list[int]] = []
#         batch, cur = [], 0
#         for idx, L in enumerate(lengths):
#             if L > max_tokens:
#                 if batch:
#                     self.batches.append(batch); batch, cur = [], 0
#                 self.batches.append([idx]); continue
#             if cur + L > max_tokens and batch:
#                 self.batches.append(batch); batch, cur = [], 0
#             batch.append(idx); cur += L
#         if batch:
#             self.batches.append(batch)
#     def __iter__(self): return iter(self.batches)
#     def __len__(self):  return len(self.batches)


# ─────────────────────  data helpers  ────────────────────────
class TokenBatchSampler(Sampler[Sequence[int]]):
    def __init__(
        self,
        lengths: list[int],
        max_tokens: int,
        world_size: int = 1,       # ← NEW
    ):
        self.batches: list[list[int]] = []
        batch, cur = [], 0
        for idx, L in enumerate(lengths):
            if L > max_tokens:
                if batch:
                    self.batches.append(batch); batch, cur = [], 0
                self.batches.append([idx]); continue
            if cur + L > max_tokens and batch:
                self.batches.append(batch); batch, cur = [], 0
            batch.append(idx); cur += L
        if batch:
            self.batches.append(batch)
        # ─── Ensure divisibility by world_size ──────────────────
        rem = len(self.batches) % world_size
        if rem:                         # drop the last `rem` batches
            self.batches = self.batches[:-rem]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def pad_batch(features, pad_id, label_pad_id=-100):
    for f in features:
        if len(f["input_ids"]) != len(f["labels"]):
            raise ValueError("input_ids / labels length mismatch")
    ids  = [torch.tensor(f["input_ids"]) for f in features]
    labs = [torch.tensor(f["labels"])    for f in features]
    input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    labels    = pad_sequence(labs,batch_first=True,padding_value=label_pad_id)
    attn      = input_ids.ne(pad_id).long()
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

# ─────────────────────────── main ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_tokens", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--flash_attn", action="store_true")
    ap.add_argument("--adam8bit",  action="store_true")
    args = ap.parse_args()

    accelerator = Accelerator(log_with="tensorboard")
    accelerator.even_batches = False

    # rank-0 logging & std redirection
    if accelerator.is_main_process:
        #redirect_stdio(args.output_dir)
        logger = setup_logging(args.output_dir)
        logger.info(f"Run args: {vars(args)}")
    else:
        logger = logging.getLogger("train")

    try:
        # ─ tokenizer & model ──────────────────────────────────────────
        tok = AutoTokenizer.from_pretrained(args.tokenizer_dir,
                                            trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        extra = {}
        if args.flash_attn:
            key = "attn_implementation" if supports_attn_kw() else "use_flash_attention_2"
            extra[key] = "flash_attention_2" if supports_attn_kw() else True
            accelerator.print("⚡ Flash-Attention 2 enabled")
            if accelerator.is_main_process:
                logger.info("Flash-Attention 2 enabled")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **extra)
        model.to(accelerator.device)
        model.resize_token_embeddings(len(tok))
        model.gradient_checkpointing_enable()
        model.config.use_cache = False


        # ─ dataloader ────────────────────────────────────────────────
        ds = datasets.load_from_disk(args.train_dir)
        sampler = TokenBatchSampler([len(r["input_ids"]) for r in ds],
                                    args.batch_tokens, world_size=4)
        loader  = DataLoader(
            ds, batch_sampler=sampler,
            collate_fn=lambda feats: pad_batch(feats, tok.pad_token_id),
            num_workers=2, pin_memory=True)

        # ─ optimizer ────────────────────────────────────────────────
        if args.adam8bit:
            if AdamW8Bit is None:
                raise RuntimeError("bitsandbytes not installed")
            optim = AdamW8Bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
            if accelerator.is_main_process:
                logger.info("Using AdamW8bit optimizer")
        else:
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.95))

        total_steps = args.epochs * len(loader)
        sched = get_cosine_schedule_with_warmup(
            optim, int(total_steps * args.warmup_ratio), total_steps)
        print('before prepare ', len(loader))
        model, optim, loader, sched = accelerator.prepare(
            model, optim, loader, sched)
        print('after prepare ', len(loader))
        # ─ early check: ensure save_pretrained exists ───────────────
        if accelerator.is_main_process:
            if not hasattr(accelerator.unwrap_model(model), "save_pretrained"):
                logger.error("Underlying model lacks save_pretrained; aborting.")
                raise AttributeError("save_pretrained not found after unwrap_model")

        # ─ training loop ───────────────────────────────────────────
        # model.train()
        # for ep in range(args.epochs):
        #     for step, batch in enumerate(loader):
        #         loss = model(**batch).loss
        #         accelerator.backward(loss)
        #         optim.step(); sched.step(); optim.zero_grad()
        #
        #         if step % 100 == 0:
        #             accelerator.print(f"ep {ep} | {step:>5}/{len(loader)} "
        #                               f"| loss {loss:.4f}")
        #             if accelerator.is_main_process:
        #                 logger.info(f"epoch {ep} step {step}/{len(loader)} "
        #                             f"loss {loss:.4f}")
        #
        #     accelerator.save_state(args.output_dir)
        #     if accelerator.is_main_process:
        #         logger.info(f"Epoch {ep} finished, checkpoint saved")

        model.train()
        for ep in range(args.epochs):
            for step, batch in enumerate(loader):
                loss = model(**batch).loss
                accelerator.backward(loss)
                optim.step();
                sched.step();
                optim.zero_grad()

                if step % 100 == 0:
                    accelerator.print(f"ep {ep} | {step:>5}/{len(loader)} "
                                      f"| loss {loss:.4f}")
                    if accelerator.is_main_process:
                        logger.info(f"epoch {ep} step {step}/{len(loader)} "
                                    f"loss {loss:.4f}")

            # ─ checkpoint at the end of every epoch ────────────────
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint{ep}")
            if accelerator.is_main_process:
                os.makedirs(ckpt_dir, exist_ok=True)
                logger.info(f"Epoch {ep} finished – saving checkpoint to {ckpt_dir}")

            # 1️⃣ Save the full (model-optim-sched) state – *all* ranks must call this
            accelerator.save_state(ckpt_dir)

            # 2️⃣ Save just the model & tokenizer – only rank-0 needs to do this
            if accelerator.is_main_process:
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                tok.save_pretrained(ckpt_dir)

        # ─ final save ──────────────────────────────────────────────
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(args.output_dir)
            tok.save_pretrained(args.output_dir)
            logger.info("Training complete — model & tokenizer saved")
            accelerator.print(f"✓ model saved to {args.output_dir}")

    except Exception:                         # log any crash
        if accelerator.is_main_process:
            logger.exception("Training crashed with an exception")
        raise

# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
