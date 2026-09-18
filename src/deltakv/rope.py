"""Rotary positional embeddings. RoPE is linear, so ΔK_rope = RoPE(ΔK)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Llama-style rotate_half: split the last dim in two contiguous halves."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to a packed tensor.

    ``x`` is ``[..., n_heads, head_dim]`` (or ``[..., head_dim]``).
    ``cos``/``sin`` broadcast over heads: ``[..., head_dim]``.
    """
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables, Llama-compatible."""

    def __init__(self, head_dim: int, max_seq: int = 8192, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.max_seq = max_seq
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def tables(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if seq_len > self.max_seq:
            raise ValueError(f"seq_len {seq_len} exceeds RoPE cache {self.max_seq}")
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        """Rotate ``x`` of shape ``[seq, n_heads, head_dim]`` or ``[seq, head_dim]``."""
        seq = x.shape[0]
        cos, sin = self.tables(seq if positions is None else int(positions.max().item()) + 1, x.device, x.dtype)
        if positions is None:
            c, s = cos[:seq], sin[:seq]
        else:
            c, s = cos[positions], sin[positions]
        return apply_rope(x, c, s)


def reshape_for_heads(x: Tensor, n_heads: int, head_dim: int) -> Tensor:
    """``[seq, n_heads * head_dim]`` → ``[seq, n_heads, head_dim]``."""
    return x.view(x.shape[0], n_heads, head_dim)


def merge_heads(x: Tensor) -> Tensor:
    """``[seq, n_heads, head_dim]`` → ``[seq, n_heads * head_dim]``."""
    return x.reshape(x.shape[0], -1)
