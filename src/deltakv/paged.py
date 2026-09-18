"""Scatter a dense ΔKV into vLLM-style paged blocks."""

from __future__ import annotations

from torch import Tensor


def scatter_into_paged(
    delta: Tensor,
    cache: Tensor,
    block_table: list[int],
    block_size: int,
    *,
    kv_dim: int = 0,
) -> None:
    """Write ``delta`` ``[seq, n_kv_heads, head_dim]`` into a paged cache.

    Supported layouts (vLLM V1 and common variants):

    * ``[num_blocks, block_size, n_kv_heads, head_dim]``
    * ``[2, num_blocks, block_size, n_kv_heads, head_dim]`` with ``kv_dim=0``
      selecting K (0) or V (1)
    * ``[num_blocks, 2, block_size, n_kv_heads, head_dim]`` with ``kv_dim=1``
    """
    seq = delta.shape[0]
    n_blocks_needed = (seq + block_size - 1) // block_size
    if len(block_table) < n_blocks_needed:
        raise ValueError("block_table shorter than prefix")

    if cache.ndim == 4:
        pages = cache
    elif cache.ndim == 5 and cache.shape[0] == 2:
        pages = cache[kv_dim]
    elif cache.ndim == 5 and cache.shape[1] == 2:
        pages = cache[:, kv_dim]
    else:
        raise ValueError(f"unsupported paged KV layout {tuple(cache.shape)}")

    for i, bid in enumerate(block_table[:n_blocks_needed]):
        start = i * block_size
        end = min(start + block_size, seq)
        span = end - start
        pages[bid, :span] = pages[bid, :span] + delta[start:end]


def gather_from_paged(
    cache: Tensor,
    block_table: list[int],
    seq_len: int,
    block_size: int,
    *,
    kv_dim: int = 0,
) -> Tensor:
    """Inverse of :func:`scatter_into_paged`."""
    if cache.ndim == 4:
        pages = cache
    elif cache.ndim == 5 and cache.shape[0] == 2:
        pages = cache[kv_dim]
    elif cache.ndim == 5 and cache.shape[1] == 2:
        pages = cache[:, kv_dim]
    else:
        raise ValueError(f"unsupported paged KV layout {tuple(cache.shape)}")
    n_kv, hd = pages.shape[-2], pages.shape[-1]
    out = pages.new_zeros(seq_len, n_kv, hd)
    n_blocks_needed = (seq_len + block_size - 1) // block_size
    for i, bid in enumerate(block_table[:n_blocks_needed]):
        start = i * block_size
        end = min(start + block_size, seq_len)
        out[start:end] = pages[bid, : end - start]
    return out
