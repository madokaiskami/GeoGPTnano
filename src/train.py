# src/train.py
# Reverse-nanoGPT style trainer with:
# - SentencePiece packing -> memmap tokens
# - AMP (torch.cuda.amp), AdamW, cosine LR + warmup
# - Gradient Accumulation
# - Gradient Checkpointing (per-block, toggle in GPTConfig)
# - SDPA (FlashAttention2 path) via PyTorch 2.x SDPA
# - SWA（Stochastic Weight Averaging，惰性初始化）
# - Resume (--resume / --auto_resume)
#
# Run:
#   python -m src.train --config configs/train_small.yaml

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

from .models import GPT, GPTConfig


@dataclass
class TrainConfig:
    # I/O
    sp_model: str = "tokenizer/spm_unigram_24000.model"
    train_txt: str = "data/splits/train.txt"
    val_txt: str = "data/splits/val.txt"
    bin_dir: str = "data/splits/bin"
    ckpt_dir: str = "checkpoints"

    # Model
    vocab_size: int = 24000
    n_layer: int = 12
    n_head: int = 8
    d_model: int = 512
    n_ctx: int = 512
    ffn_mult: int = 4
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_weights: bool = True
    bias: bool = False

    # Resume-safe toggles
    use_sdpa: bool = True
    grad_checkpoint: bool = False

    # Train
    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 1.5e-3
    min_lr: float = 1.5e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    dropout_train: float = 0.0

    warmup_steps: int = 2000
    max_steps: int = 20000
    eval_every: int = 500
    save_every: int = 2000
    eval_batches: int = 100
    seed: int = 1337
    amp: bool = True
    compile: bool = False

    # SWA
    use_swa: bool = False
    swa_start: int = 10000
    swa_freq: int = 500
    swa_lr: float = 7.5e-4  # MUST be numeric in YAML (not a quoted string)

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = ("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                  else "float16")


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------
# Data packing
# -------------------------
def _strip_literal_eos(s: str) -> str:
    s = s.rstrip()
    if s.endswith("<eos>"):
        s = s[: -len("<eos>")].rstrip()
    return s


def build_memmaps(cfg: TrainConfig):
    Path(cfg.bin_dir).mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=cfg.sp_model)
    eos_id = sp.eos_id()
    assert eos_id >= 0, "SPM model must define eos_id."

    def tok_file(txt_path: str, out_path: str):
        ids = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                x = sp.encode(_strip_literal_eos(line), out_type=int, add_bos=False, add_eos=True)
                ids.extend(x)
        arr = np.array(ids, dtype=np.int32)
        if cfg.vocab_size <= 65535:
            arr = arr.astype(np.uint16)
        np.save(out_path, arr)
        return int(arr.shape[0])

    n_train = tok_file(cfg.train_txt, os.path.join(cfg.bin_dir, "train_ids.npy"))
    n_val = tok_file(cfg.val_txt, os.path.join(cfg.bin_dir, "val_ids.npy"))
    meta = {"train_tokens": n_train, "val_tokens": n_val, "vocab_size": cfg.vocab_size,
            "dtype": "uint16" if cfg.vocab_size <= 65535 else "int32"}
    with open(os.path.join(cfg.bin_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[pack] ", meta)


class TokenMemmap:
    def __init__(self, path: str):
        self.arr = np.load(path, mmap_mode="r")
        self.n = int(self.arr.shape[0])

    def get_batch(self, batch_size: int, block_size: int, device: str, split: str):
        assert self.n > block_size + 1, "Not enough tokens for one crop."
        if split == "train":
            ix = np.random.randint(0, self.n - (block_size + 1), size=(batch_size,))
        else:
            stride = max(1, (self.n - (block_size + 1)) // max(1, batch_size))
            ix = np.arange(0, batch_size * stride, stride) % (self.n - (block_size + 1))
        x = np.stack([self.arr[i: i + block_size] for i in ix])
        y = np.stack([self.arr[i + 1: i + 1 + block_size] for i in ix])
        x = torch.from_numpy(x.astype(np.int64)).to(device)
        y = torch.from_numpy(y.astype(np.int64)).to(device)
        return x, y


# -------------------------
# Model / Optimizer
# -------------------------
def configure_model(cfg: TrainConfig) -> GPT:
    mcfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_model=cfg.d_model,
        n_ctx=cfg.n_ctx,
        ffn_mult=cfg.ffn_mult,
        dropout=cfg.dropout,
        rope_theta=cfg.rope_theta,
        tie_weights=cfg.tie_weights,
        bias=cfg.bias,
        use_sdpa=cfg.use_sdpa,
        grad_checkpoint=cfg.grad_checkpoint,
    )
    return GPT(mcfg)


def configure_optimizers(model: nn.Module, weight_decay: float, lr: float, betas=(0.9, 0.95)):
    decay_params, no_decay_params = [], []

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            full = f"{mn}.{pn}" if mn else pn
            lname = full.lower()
            if lname.endswith("bias"):
                no_decay_params.append(p)
            elif "ln" in lname or "norm" in lname:
                no_decay_params.append(p)
            elif "tok_emb" in lname or "embedding" in lname or "lm_head" in lname:
                no_decay_params.append(p)
            elif isinstance(m, nn.Linear) and pn == "weight":
                decay_params.append(p)
            else:
                no_decay_params.append(p)

    def uniq(params):
        seen, out = set(), []
        for p in params:
            i = id(p)
            if i not in seen:
                out.append(p); seen.add(i)
        return out

    decay_params = uniq(decay_params)
    no_decay_params = uniq(no_decay_params)
    nd_ids = {id(p) for p in no_decay_params}
    decay_params = [p for p in decay_params if id(p) not in nd_ids]

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


# -------------------------
# Schedules / Eval
# -------------------------
def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, (cfg.max_steps - cfg.warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, val_mem: TokenMemmap, cfg: TrainConfig) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = val_mem.get_batch(cfg.batch_size, cfg.n_ctx, cfg.device, split="val")
        with torch.cuda.amp.autocast(
            enabled=cfg.amp and cfg.device.startswith("cuda"),
            dtype=(torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16)
        ):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# -------------------------
# Checkpoint utils
# -------------------------
def latest_checkpoint(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    cks = sorted(p.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return str(cks[0]) if cks else None


def save_checkpoint(path: str, model, optimizer, scaler, global_step: int, best_val: float,
                    swa_model: Optional[AveragedModel] = None):
    state = {
        "model": model.state_dict(),
        "optimizer": (optimizer.state_dict() if optimizer is not None else None),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "config": model.config.__dict__,
        "global_step": int(global_step),
        "best_val": float(best_val),
        "swa_model": (swa_model.state_dict() if swa_model is not None else None),
    }
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scaler=None, device: str = "cpu",
                    swa_model: Optional[AveragedModel] = None):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    global_step = int(state.get("global_step", state.get("step", 0)))
    best_val = float(state.get("best_val", float("inf")))
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if swa_model is not None and state.get("swa_model") is not None:
        swa_model.load_state_dict(state["swa_model"])
    model.to(device)
    if swa_model is not None:
        swa_model.to(device)
    return global_step, best_val


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--prepare_only", action="store_true")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--auto_resume", action="store_true")
    args = ap.parse_args()

    # Load config
    if args.config:
        raw = load_yaml(args.config)
        raw = raw.get("train", raw)
        cfg = TrainConfig(**raw)
    else:
        cfg = TrainConfig()

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.bin_dir, exist_ok=True)
    set_seed(cfg.seed)

    # Pack tokens -> memmaps (idempotent)
    train_ids = os.path.join(cfg.bin_dir, "train_ids.npy")
    val_ids = os.path.join(cfg.bin_dir, "val_ids.npy")
    if not (os.path.exists(train_ids) and os.path.exists(val_ids)):
        print("[build] token memmaps not found; building...")
        build_memmaps(cfg)
    else:
        print("[build] using existing memmaps.")

    if args.prepare_only:
        return

    # Data
    train_mem = TokenMemmap(train_ids)
    val_mem = TokenMemmap(val_ids)

    # Model
    model = configure_model(cfg).to(cfg.device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Optimizer + AMP
    optimizer = configure_optimizers(model, cfg.weight_decay, cfg.lr, cfg.betas)
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.amp and cfg.device.startswith("cuda") and cfg.dtype == "float16"
    )

    # Adjust dropout at train time
    for m in model.modules():
        if isinstance(m, nn.Dropout) and cfg.dropout_train is not None:
            m.p = cfg.dropout_train

    # SWA states (lazy init)
    swa_model: Optional[AveragedModel] = None
    swa_scheduler: Optional[SWALR] = None

    # Resume
    global_step = 0
    best_val = float("inf")
    resume_path = args.resume or (latest_checkpoint(cfg.ckpt_dir) if args.auto_resume else None)
    if resume_path:
        if cfg.use_swa and swa_model is None:
            swa_model = AveragedModel(model, avg_fn=None)
        global_step, best_val = load_checkpoint(
            resume_path, model, optimizer, scaler, device=cfg.device, swa_model=swa_model
        )

    def maybe_activate_swa():
        nonlocal swa_model, swa_scheduler
        if not cfg.use_swa:
            return
        if global_step >= cfg.swa_start and swa_scheduler is None:
            if swa_model is None:
                swa_model = AveragedModel(model, avg_fn=None).to(cfg.device)
            swa_scheduler = SWALR(optimizer, swa_lr=float(cfg.swa_lr))
            print(f"[swa] activated at step {global_step}, swa_lr={float(cfg.swa_lr)}")

    maybe_activate_swa()

    # Train loop
    model.train()
    while global_step < cfg.max_steps:
        # SWA 
        if cfg.use_swa and swa_scheduler is None and global_step + 1 >= cfg.swa_start:
            if swa_model is None:
                swa_model = AveragedModel(model, avg_fn=None).to(cfg.device)
            swa_scheduler = SWALR(optimizer, swa_lr=float(cfg.swa_lr))
            print(f"[swa] activated at step {global_step+1}, swa_lr={float(cfg.swa_lr)}")

        optimizer.zero_grad(set_to_none=True)
        last_loss_full = None

        # Gradient Accumulation
        for _ in range(cfg.grad_accum_steps):
            x, y = train_mem.get_batch(cfg.batch_size, cfg.n_ctx, cfg.device, split="train")
            with torch.cuda.amp.autocast(
                enabled=cfg.amp and cfg.device.startswith("cuda"),
                dtype=(torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16)
            ):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps

            if cfg.dtype == "float16" and cfg.amp and cfg.device.startswith("cuda"):
                scaler.scale(loss).backward()
            else:
                loss.backward()

            last_loss_full = float(loss.item()) * cfg.grad_accum_steps

        # LR 
        if cfg.use_swa and swa_scheduler is not None:
            swa_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = get_lr(global_step + 1, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Clip & Step
        if cfg.grad_clip and cfg.grad_clip > 0:
            if cfg.dtype == "float16" and cfg.amp and cfg.device.startswith("cuda"):
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if cfg.dtype == "float16" and cfg.amp and cfg.device.startswith("cuda"):
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        # SWA 
        if cfg.use_swa and swa_model is not None and swa_scheduler is not None:
            if (global_step - cfg.swa_start) >= 0 and ((global_step - cfg.swa_start) % max(1, cfg.swa_freq) == 0):
                swa_model.update_parameters(model)

        # Logging
        if global_step % 50 == 0 or global_step == 1:
            print(f"step {global_step:5d} | lr {lr:.3e} | train_loss {last_loss_full:.4f}")

        # Eval
        if global_step % cfg.eval_every == 0:
            eval_target = (swa_model.module if (cfg.use_swa and swa_model is not None and swa_scheduler is not None)
                           else model)
            val_loss = evaluate(eval_target, val_mem, cfg)
            print(f"[eval] step {global_step} | val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(cfg.ckpt_dir, f"best_step{global_step}_val{val_loss:.3f}.pt")
                save_checkpoint(ckpt_path, model, optimizer, scaler, global_step, best_val, swa_model=swa_model)
                print(f"[ckpt] saved {ckpt_path}")

        # Periodic save
        if global_step % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"step{global_step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, global_step, best_val, swa_model=swa_model)
            print(f"[ckpt] saved {ckpt_path}")


if __name__ == "__main__":
    main()

