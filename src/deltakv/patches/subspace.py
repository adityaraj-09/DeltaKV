"""Rank-r LoRA-subspace KV correction. Generalizes AgentKVShift's rank-1 μ."""

from __future__ import annotations

import torch
from torch import Tensor

from deltakv.deltas.descriptor import LayerDelta
from deltakv.deltas.factors import LowRankFactors


def flatten_kv(t: Tensor) -> Tensor:
    """``[seq, n_kv_heads, head_dim] → [seq, d_kv]``."""
    return t.reshape(t.shape[0], -1)


def unflatten_kv(t: Tensor, n_kv_heads: int, head_dim: int) -> Tensor:
    return t.view(t.shape[0], n_kv_heads, head_dim)


def _solve_codes(residual: Tensor, basis: Tensor) -> Tensor:
    """Least-squares codes so ``residual ≈ codes @ basis.T``.

    ``residual`` is ``[n, d]``, ``basis`` is ``[d, r]``. Returns ``[n, r]``.
    """
    # (Bᵀ B) Cᵀ = Bᵀ Rᵀ  →  C = R B (Bᵀ B)⁻¹
    gram = basis.T @ basis
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    rhs = residual @ basis
    try:
        return torch.linalg.solve(gram + 1e-6 * eye, rhs.T).T
    except RuntimeError:
        pinv = torch.linalg.pinv(gram + 1e-6 * eye)
        return rhs @ pinv


def svd_basis(residual: Tensor, rank: int) -> Tensor:
    """Leading-r right singular vectors of ``residual [n, d]`` as ``[d, r]``."""
    r = max(1, min(int(rank), residual.shape[0], residual.shape[1]))
    # svd on float32 for CPU stability
    _, _, vh = torch.linalg.svd(residual.float(), full_matrices=False)
    return vh[:r].T.to(residual.dtype).contiguous()


def subspace_correct(
    reused: Tensor,
    fresh: Tensor,
    probe_index: Tensor,
    *,
    basis: Tensor | None = None,
    rank: int = 8,
    token_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Correct ``reused`` toward ``fresh`` in a rank-r subspace plus residual μ.

    When ``basis`` is LoRA ``B [d, r]``, per-token codes live in ``col(B)``
    — the exact column space of a projection-layer ΔV — and only the
    orthogonal leftover is mean-shifted (AgentKVShift's μ). A rank-1
    mean-shift is the special case ``rank=1`` with no ``basis``.

    Returns ``(corrected, subspace_offset[d], orth_mu[d])``.
    """
    seq, n_kv, hd = reused.shape
    idx = probe_index.long()
    rf = flatten_kv(reused)
    ff = flatten_kv(fresh)
    residual = ff - rf
    rp = residual[idx]
    d = rf.shape[-1]

    if basis is None:
        r_eff = min(rank, max(1, int(idx.numel()) - 1), d)
        basis = svd_basis(rp, r_eff) if rp.shape[0] >= 2 else None

    if basis is not None and basis.numel() and int(idx.numel()) >= 1:
        if basis.shape[0] != d:
            raise ValueError(f"basis d={basis.shape[0]} != kv d={d}")
        # Per-token codes on the full residual estimate (probes are exact;
        # non-probes use mixed-forward KV). Shared μ is the leftover
        # orthogonal to col(B), estimated from probes only.
        codes = _solve_codes(residual, basis)
        fitted = codes @ basis.T
        subspace = codes.mean(dim=0) @ basis.T
        orth_mu = (rp - fitted[idx]).mean(dim=0)
        structured = True
        field = fitted  # per-token col(B); μ applied separately
    else:
        subspace = residual.new_zeros(d)
        orth_mu = rp.mean(dim=0) if rp.numel() else residual.new_zeros(d)
        structured = False
        field = residual.new_zeros(seq, d)

    if token_weights is None:
        dist = residual.norm(dim=-1)
        scale = dist.mean().clamp_min(1e-8)
        w = (dist / scale).clamp(max=1.0)
    else:
        w = token_weights

    # Structured (LoRA column space) is applied in full; only the unstructured
    # residual μ is CacheBlend-weighted. Probes always take the fresh value.
    corr = rf + field + w.unsqueeze(-1) * orth_mu.unsqueeze(0)
    if not structured:
        corr = rf + w.unsqueeze(-1) * orth_mu.unsqueeze(0)
    corr[idx] = ff[idx]
    return unflatten_kv(corr, n_kv, hd), subspace, orth_mu


def bases_for_layer(
    layer: LayerDelta | None,
) -> tuple[Tensor | None, Tensor | None]:
    """LoRA ``B`` for k_proj / v_proj, or None if that map has no ΔW."""
    if layer is None:
        return None, None
    k_fac: LowRankFactors | None = layer.get("k_proj")
    v_fac: LowRankFactors | None = layer.get("v_proj")
    k_b = k_fac.B if k_fac is not None else None
    v_b = v_fac.B if v_fac is not None else None
    return k_b, v_b
