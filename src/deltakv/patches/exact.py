"""Exact projection-layer patches: ΔK = RoPE(X ΔW_K), ΔV = X ΔW_V."""

from __future__ import annotations

import torch
from torch import Tensor

from deltakv.deltas.descriptor import LayerDelta, WeightDelta
from deltakv.deltas.factors import LowRankFactors
from deltakv.patches.tensors import LowRankKVPatch
from deltakv.rope import RotaryEmbedding, merge_heads, reshape_for_heads


def exact_kv_patch(
    x_normed: Tensor,
    layer: LayerDelta,
    *,
    n_kv_heads: int,
    head_dim: int,
    rope: RotaryEmbedding | None = None,
    positions: Tensor | None = None,
) -> LowRankKVPatch:
    """Exact ΔK/ΔV given the *true* input activations of this layer's k/v proj.

    RoPE is a linear operator, so ``RoPE(K+ΔK) = RoPE(K)+RoPE(ΔK)``. When the
    adapter is rank-r, ΔK before RoPE is exactly rank-r; after RoPE we store a
    dense [seq, n_kv, d] increment because rotation mixes the basis. Callers
    that skip RoPE (no positional keys, or RoPE applied later) keep the
    compressed form.
    """
    k_fac = layer.get("k_proj")
    v_fac = layer.get("v_proj")
    seq = x_normed.shape[0]

    k_codes = k_basis = v_codes = v_basis = None
    k_dense = v_dense = None

    if k_fac is not None:
        k_codes = k_fac.codes(x_normed)
        k_basis = k_fac.B
        if rope is not None:
            dk = reshape_for_heads(k_fac.apply(x_normed), n_kv_heads, head_dim)
            dk = rope(dk, positions)
            k_dense = dk
            k_codes = k_basis = None
    else:
        k_dense = x_normed.new_zeros(seq, n_kv_heads, head_dim)

    if v_fac is not None:
        v_codes = v_fac.codes(x_normed)
        v_basis = v_fac.B
    else:
        v_dense = x_normed.new_zeros(seq, n_kv_heads, head_dim)

    return LowRankKVPatch(
        layer_idx=layer.layer_idx,
        k_codes=k_codes,
        k_basis=k_basis,
        v_codes=v_codes,
        v_basis=v_basis,
        k_dense=k_dense,
        v_dense=v_dense,
    )


def zeroth_order_patches(
    hidden_normed: dict[int, Tensor],
    delta: WeightDelta,
    *,
    n_kv_heads: int,
    head_dim: int,
    rope: RotaryEmbedding | None = None,
    positions: Tensor | None = None,
) -> dict[int, LowRankKVPatch]:
    """Adapter path on *cached* (stale) hidden states at every modified layer.

    Exact at any layer whose input X did not move; approximate above the first
    modified layer. This is the cheapest deep-layer route and the baseline the
    probe/analytic routes correct.
    """
    out: dict[int, LowRankKVPatch] = {}
    for idx, layer in delta.layers.items():
        x = hidden_normed.get(idx)
        if x is None:
            continue
        out[idx] = exact_kv_patch(
            x,
            layer,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope=rope,
            positions=positions,
        )
    return out


def apply_linear_delta(x: Tensor, fac: LowRankFactors | None) -> Tensor:
    if fac is None:
        return x.new_zeros(*x.shape[:-1], 0) if x.ndim else x
    return fac.apply(x)


def packed_delta(x: Tensor, fac: LowRankFactors | None, n_heads: int, head_dim: int) -> Tensor:
    """Return ``[seq, n_heads, head_dim]``, zeros if this proj has no ΔW."""
    seq = x.shape[0]
    if fac is None:
        return x.new_zeros(seq, n_heads, head_dim)
    return reshape_for_heads(fac.apply(x), n_heads, head_dim)


def lora_activation_scores(x_normed: Tensor, fac: LowRankFactors | None) -> Tensor:
    """Per-token score ``||x A^T||`` — tokens that fire the adapter the hardest."""
    if fac is None or int(x_normed.shape[-1]) != fac.in_features:
        return x_normed.new_zeros(x_normed.shape[0])
    return fac.codes(x_normed).norm(dim=-1)
