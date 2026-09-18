"""Probe-anchored KV offset correction (AgentKVShift on the weight axis)."""

from __future__ import annotations

import torch
from torch import Tensor

from deltakv.deltas.descriptor import WeightDelta
from deltakv.patches.exact import lora_activation_scores
from deltakv.patches.subspace import bases_for_layer, subspace_correct
from deltakv.patches.tensors import KVCache, LowRankKVPatch


def select_probes(
    scores: Tensor,
    ratio: float,
    min_tokens: int = 4,
    max_ratio: float = 0.5,
) -> Tensor:
    """Return indices of the top-b tokens. Always includes 0 (BOS/anchor)."""
    n = int(scores.numel())
    b = int(max(min_tokens, round(ratio * n)))
    b = min(b, max(1, int(max_ratio * n)), n)
    top = torch.topk(scores, k=b, largest=True).indices
    if 0 not in top.tolist():
        top = torch.cat([scores.new_zeros(1, dtype=torch.long), top[:-1]])
    return torch.unique(top.sort().values)


def weight_axis_scores(
    x_normed: Tensor,
    delta: WeightDelta,
    layer_idx: int = 0,
) -> Tensor:
    """Tokens that fire LoRA/edit factors the hardest at the first modified layer."""
    layer = delta.layers.get(layer_idx) or delta.layers.get(delta.first_modified_layer)
    if layer is None:
        return x_normed.new_ones(x_normed.shape[0])
    acc = x_normed.new_zeros(x_normed.shape[0])
    in_dim = int(x_normed.shape[-1])
    for fac in layer.projections.values():
        if fac.in_features != in_dim:
            continue
        acc = acc + lora_activation_scores(x_normed, fac)
    if float(acc.max()) <= 0:
        return x_normed.new_ones(x_normed.shape[0])
    return acc


def probe_offset_correct(
    reused_k: Tensor,
    reused_v: Tensor,
    fresh_k: Tensor,
    fresh_v: Tensor,
    probe_index: Tensor,
    *,
    weights: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """AgentKVShift mean-shift: estimate chunk offset from probes, add to the rest.

    ``reused_*`` / ``fresh_*`` are ``[seq, n_kv_heads, head_dim]``.
    Probes are replaced with fresh values; others get ``w_j * μ``.
    Returns ``(k_corr, v_corr, mu_k, mu_v)``.
    """
    seq = reused_k.shape[0]
    idx = probe_index.long()
    mu_k = (fresh_k[idx] - reused_k[idx]).mean(dim=0)
    mu_v = (fresh_v[idx] - reused_v[idx]).mean(dim=0)
    if weights is None:
        w = reused_k.new_ones(seq)
        # Down-weight tokens whose layer-1 residual is already tiny.
        d = (fresh_k - reused_k).flatten(1).norm(dim=-1)
        scale = d.mean().clamp_min(1e-8)
        w = (d / scale).clamp(max=1.0)
        # Probes always take the fresh value; keep w for the others.
    else:
        w = weights
    k_corr = reused_k + w.view(seq, 1, 1) * mu_k
    v_corr = reused_v + w.view(seq, 1, 1) * mu_v
    k_corr[idx] = fresh_k[idx]
    v_corr[idx] = fresh_v[idx]
    return k_corr, v_corr, mu_k, mu_v


def residual_gate(
    reused: Tensor,
    corrected: Tensor,
    fresh_probe: Tensor,
    probe_index: Tensor,
    threshold: float,
    max_ratio: float,
) -> Tensor:
    """CacheBlend-style: extra recompute for tokens whose residual still looks large.

    We don't have fresh KV for non-probes, so we *estimate* per-token residual
    as distance from the probe-derived offset field, and take the top outliers
    above ``threshold`` relative to probe residual scale.
    """
    seq = reused.shape[0]
    est = (corrected - reused).flatten(1).norm(dim=-1)
    probe_scale = (fresh_probe[probe_index] - reused[probe_index]).flatten(1).norm(dim=-1)
    scale = probe_scale.mean().clamp_min(1e-8)
    relative = est / scale
    mask = relative > threshold
    extra = mask.nonzero(as_tuple=False).view(-1)
    cap = max(0, int(max_ratio * seq) - int(probe_index.numel()))
    if extra.numel() > cap:
        extra = extra[relative[extra].topk(cap).indices]
    return extra


def apply_probe_to_cache(
    kv: KVCache,
    fresh_layers: dict[int, tuple[Tensor, Tensor]],
    probe_index: Tensor,
    blend_threshold: float = 0.15,
    blend_max_ratio: float = 0.40,
    *,
    layer_deltas: dict | None = None,
    subspace_rank: int = 8,
    probe_layers: set[int] | None = None,
    skip_layers: set[int] | None = None,
) -> tuple[KVCache, dict[int, LowRankKVPatch], Tensor]:
    """Correct cached KV using a rank-r subspace shift + residual μ.

    ``layer_deltas`` maps layer index → ``LayerDelta`` so V can use LoRA ``B``
    as the exact column-space basis. K is RoPE'd, so its basis is the SVD of
    the probe residual. Layers in ``probe_layers`` also run the CacheBlend
    residual gate; ``None`` gates every layer (legacy probe route).
    ``skip_layers`` keep the incoming KV (exact projection patches).
    """
    out = kv.clone()
    patches: dict[int, LowRankKVPatch] = {}
    extra_all: list[Tensor] = []
    for l, (fk, fv) in fresh_layers.items():
        if skip_layers is not None and l in skip_layers:
            continue
        rk, rv = out.layer(l)
        layer = None if layer_deltas is None else layer_deltas.get(l)
        k_b, v_b = bases_for_layer(layer)
        k_rank = int(k_b.shape[1]) if k_b is not None else subspace_rank
        k_c, _, _ = subspace_correct(rk, fk, probe_index, basis=None, rank=k_rank)
        v_c, _, _ = subspace_correct(rv, fv, probe_index, basis=v_b, rank=subspace_rank)
        gated = probe_layers is None or l in probe_layers
        if gated:
            extra = residual_gate(rk, k_c, fk, probe_index, blend_threshold, blend_max_ratio)
            if extra.numel():
                k_c[extra] = fk[extra]
                v_c[extra] = fv[extra]
                extra_all.append(extra)
        out.k[l] = k_c
        out.v[l] = v_c
        patches[l] = LowRankKVPatch(
            layer_idx=l,
            k_codes=None,
            k_basis=None,
            v_codes=None,
            v_basis=None,
            k_dense=k_c - rk,
            v_dense=v_c - rv,
        )
    extra_idx = (
        torch.unique(torch.cat(extra_all)) if extra_all else probe_index.new_empty(0, dtype=torch.long)
    )
    return out, patches, extra_idx
