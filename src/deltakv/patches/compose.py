"""Patch composition, application onto a KV tensor, and error accounting."""

from __future__ import annotations

import torch
from torch import Tensor

from deltakv.config import DeltaKVConfig
from deltakv.deltas.descriptor import WeightDelta
from deltakv.patches.tensors import AppliedPatch, KVCache, LowRankKVPatch
from deltakv.types import ErrorEstimate, PatchRoute, WeightVersion


def materialize_layer_patch(
    patch: LowRankKVPatch,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if patch.k_dense is not None or patch.k_codes is not None:
        dk, dv = patch.materialize(n_kv_heads, head_dim)
        return dk, dv
    z = torch.zeros(seq_len, n_kv_heads, head_dim, device=device, dtype=dtype)
    return z, z.clone()


def apply_layer_patches(kv: KVCache, patches: dict[int, LowRankKVPatch]) -> KVCache:
    out = kv.clone()
    for idx, p in patches.items():
        dk, dv = p.materialize(out.n_kv_heads, out.head_dim)
        out.add_layer_delta(idx, dk, dv)
    return out


def dense_delta_patches(base: KVCache, patched: KVCache) -> dict[int, LowRankKVPatch]:
    """Store the full patched−base increment so chain replay does not drop stages."""
    out: dict[int, LowRankKVPatch] = {}
    for l in range(patched.n_layers):
        pk, pv = patched.layer(l)
        bk, bv = base.layer(l)
        dk, dv = pk - bk, pv - bv
        if float(dk.abs().max()) == 0.0 and float(dv.abs().max()) == 0.0:
            continue
        out[l] = LowRankKVPatch(
            layer_idx=l,
            k_codes=None,
            k_basis=None,
            v_codes=None,
            v_basis=None,
            k_dense=dk,
            v_dense=dv,
        )
    return out


def compose_applied(
    first: AppliedPatch,
    second: AppliedPatch,
    n_kv_heads: int,
    head_dim: int,
) -> AppliedPatch:
    """patch(patch(KV, ΔW1), ΔW2) with additive error tracking."""
    if first.target.id != second.source.id:
        raise ValueError("applied-patch lineage break")
    merged: dict[int, LowRankKVPatch] = {}
    ids = set(first.layer_patches) | set(second.layer_patches)
    for idx in ids:
        a = first.layer_patches.get(idx)
        b = second.layer_patches.get(idx)
        if a is None:
            merged[idx] = b  # type: ignore[assignment]
            continue
        if b is None:
            merged[idx] = a
            continue
        da_k, da_v = a.materialize(n_kv_heads, head_dim)
        db_k, db_v = b.materialize(n_kv_heads, head_dim)
        merged[idx] = LowRankKVPatch(
            layer_idx=idx,
            k_codes=None,
            k_basis=None,
            v_codes=None,
            v_basis=None,
            k_dense=da_k + db_k,
            v_dense=da_v + db_v,
        )
    err = ErrorEstimate(
        relative_kv_l2=first.error.relative_kv_l2 + second.error.relative_kv_l2,
        per_layer=_sum_layers(first.error.per_layer, second.error.per_layer),
        route=second.route,
        measured=first.error.measured and second.error.measured,
        notes="composed",
    )
    return AppliedPatch(
        source=first.source,
        target=second.target,
        route=second.route,
        layer_patches=merged,
        error=err,
    )


def _sum_layers(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    if not a:
        return b
    if not b:
        return a
    n = max(len(a), len(b))
    out = []
    for i in range(n):
        out.append((a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0))
    return tuple(out)


def zeroth_order_error(
    delta: WeightDelta,
    n_layers: int,
    lipschitz: float,
) -> ErrorEstimate:
    """Cheap a-priori bound: hidden-state drift compounds above the first ΔW."""
    first = delta.first_modified_layer
    mag = max(delta.relative_magnitude(), 1e-8)
    # Normalize loosely: random_lora_factors uses scale ~ 0.02 and unit A/B,
    # so mag is already small. Treat mag as a relative proxy.
    rel = mag
    per = []
    drift = 0.0
    for l in range(n_layers):
        local = rel if l in delta.layers else 0.0
        layer_err = drift + local
        per.append(float(layer_err))
        drift = drift * lipschitz + local
        if l < first:
            drift = 0.0
            per[-1] = local
    return ErrorEstimate(
        relative_kv_l2=float(max(per) if per else rel),
        per_layer=tuple(per),
        route=PatchRoute.ZEROTH,
        measured=False,
        notes="lipschitz-compounded zeroth-order",
    )


def measured_error(patched: KVCache, fresh: KVCache) -> ErrorEstimate:
    per = []
    rels = []
    for l in range(patched.n_layers):
        pk, pv = patched.layer(l)
        fk, fv = fresh.layer(l)
        num = (pk - fk).norm() + (pv - fv).norm()
        den = fk.norm() + fv.norm() + 1e-8
        r = float(num / den)
        per.append(r)
        rels.append(r)
    return ErrorEstimate(
        relative_kv_l2=float(max(rels) if rels else 0.0),
        per_layer=tuple(per),
        route=PatchRoute.EXACT,
        measured=True,
    )


def should_rebase(error: ErrorEstimate, config: DeltaKVConfig) -> bool:
    return config.error_budget.needs_rebase(error.relative_kv_l2)


def within_budget(error: ErrorEstimate, config: DeltaKVConfig) -> bool:
    return config.error_budget.allows(error.relative_kv_l2, error.next_token_kl)
