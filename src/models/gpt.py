# src/models/gpt.py
# Decoder-only Transformer (RMSNorm + RoPE + GELU, weight tying)
# Extras:
#   - grad checkpointing (per-block)
#   - SDPA path (Flash-Attn 2 when available via PyTorch 2.x)
#
# This file is shape-compatible with older checkpoints if you keep
# vocab_size, n_layer, n_head, d_model, ffn_mult unchanged.

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 12
    n_head: int = 8
    d_model: int = 512
    n_ctx: int = 512
    ffn_mult: int = 4
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_weights: bool = True
    bias: bool = False

    # New toggles (all resume-safe, I believe))
    use_sdpa: bool = True            # try to use scaled_dot_product_attention (FlashAttn path)
    grad_checkpoint: bool = False    # use torch.utils.checkpoint over each block


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no bias)."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x


class RotaryEmbedding(nn.Module):
    """RoPE tables (cos/sin) with auto-extend."""
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE."
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    @torch.no_grad()
    def _build_cache(self, T: int):
        t = torch.arange(T, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)    # (T, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, hd)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.max_seq_len = T

    @torch.no_grad()
    def _maybe_extend(self, needed_T: int):
        if needed_T > self.max_seq_len:
            self._build_cache(needed_T)

    def forward(self, x: torch.Tensor, seq_start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        T = x.size(2)
        self._maybe_extend(seq_start + T)
        cos = self.cos_cached[:, :, seq_start: seq_start + T, :]
        sin = self.sin_cached[:, :, seq_start: seq_start + T, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout
        self.use_sdpa = bool(config.use_sdpa)

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_drop = nn.Dropout(self.dropout)
        self.resid_drop = nn.Dropout(self.dropout)

        # Fallback causal mask for non-SDPA path
        mask = torch.tril(torch.ones(config.n_ctx, config.n_ctx))
        self.register_buffer("mask", mask.view(1, 1, config.n_ctx, config.n_ctx), persistent=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.n_ctx, theta=config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        # (B, H, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self.rope(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.use_sdpa and q.is_cuda:
            # PyTorch 2.x SDPA selects Flash/MemEff/Math kernels automatically.
            # We rely on is_causal=True instead of explicit mask.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )  # (B,H,T,hd)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.ffn_mult * config.d_model
        self.fc = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.proj = nn.Linear(hidden, config.d_model, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.grad_checkpoint and self.training:
            # Per-block checkpointing (PyTorch>=1.10 supports non-reentrant flag;
            # here we keep default for broad compatibility).
            for blk in self.blocks:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            return x
        else:
            for blk in self.blocks:
                x = blk(x)
            return x

    def forward(self, idx: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, T = idx.size()
        x = self.tok_emb(idx)
        x = self.drop(x)
        x = self._forward_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_p: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.n_ctx:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf <= top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = True
                filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)
                next_id_sorted = torch.multinomial(filtered, num_samples=1)
                next_id = sorted_idx.gather(-1, next_id_sorted)
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
