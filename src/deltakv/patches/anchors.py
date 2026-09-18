"""LoRC-style condition numbers, progressive probe budgets, hybrid layer plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from deltakv.config import DeltaKVConfig
from deltakv.deltas.descriptor import WeightDelta
from deltakv.types import LayerStrategy


def spectral_condition(weight: Tensor, eps: float = 1e-6) -> float:
    """κ(W) = σ_max / σ_min for a linear map ``[out, in]``."""
    s = torch.linalg.svdvals(weight.float())
    smax = float(s[0].clamp_min(eps))
    kept = s[s > eps * smax]
    smin = float(kept[-1]) if kept.numel() else eps
    return smax / max(smin, eps)


def layer_kappas(k_weights: list[Tensor], v_weights: list[Tensor]) -> list[float]:
    """Per-layer κ(W_k) · κ(W_v)."""
    if len(k_weights) != len(v_weights):
        raise ValueError("k/v weight lists must match")
    return [
        spectral_condition(k) * spectral_condition(v) for k, v in zip(k_weights, v_weights)
    ]


def cumulative_sensitivity(per_layer_kappa: list[float]) -> list[float]:
    """κ̃_ℓ = ∏_{j=ℓ}^{L-1} κ_j  (LoRC: shallow errors are amplified more)."""
    n = len(per_layer_kappa)
    out = [1.0] * n
    acc = 1.0
    for j in range(n - 1, -1, -1):
        acc *= max(per_layer_kappa[j], 1e-8)
        out[j] = acc
    return out


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    t = torch.tensor(values, dtype=torch.float64)
    t = torch.log(t.clamp_min(1e-12))
    lo, hi = float(t.min()), float(t.max())
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [float((x - lo) / (hi - lo)) for x in t.tolist()]


def probe_ratio_schedule(
    cum_kappa: list[float],
    *,
    shallow_ratio: float = 0.20,
    deep_ratio: float = 0.05,
) -> list[float]:
    """More probes in the first third and in high-κ̃ layers."""
    n = len(cum_kappa)
    if n == 0:
        return []
    third = max(1, n // 3)
    norm = normalize_scores(cum_kappa)
    out = []
    for l, s in enumerate(norm):
        cap = shallow_ratio if l < third else deep_ratio
        out.append(deep_ratio + s * (cap - deep_ratio))
    return out


@dataclass
class LayerPlan:
    strategy: LayerStrategy
    probe_ratio: float
    kappa: float
    cum_kappa: float


def plan_layers(
    delta: WeightDelta,
    n_layers: int,
    cum_kappa: list[float],
    *,
    has_hidden: bool,
    config: DeltaKVConfig,
) -> dict[int, LayerPlan]:
    """Per-layer strategy: skip / exact / subspace / probe.

    - No ΔW and no upstream ΔW → skip
    - No local ΔW, upstream ΔX exists → subspace (correct ΔX-induced KV drift)
    - Layer 0 (and every ``hidden_stride`` boundary) with stored X and k/v ΔW → exact
    - High cumulative κ → probe (subspace + residual gate)
    - Otherwise → subspace (rank-r shift, no extra blend)
    """
    per = []
    if cum_kappa:
        # invert product to per-layer by ratio of neighbors
        per = [cum_kappa[i] / cum_kappa[i + 1] if i + 1 < n_layers else cum_kappa[i] for i in range(n_layers)]
    else:
        per = [1.0] * n_layers
        cum_kappa = [1.0] * n_layers
    ratios = probe_ratio_schedule(
        cum_kappa,
        shallow_ratio=config.probe_shallow_ratio,
        deep_ratio=config.probe_deep_ratio,
    )
    norm = normalize_scores(cum_kappa)
    stride = max(1, int(config.hidden_stride))
    plans: dict[int, LayerPlan] = {}
    for l in range(n_layers):
        layer = delta.layers.get(l)
        ratio = ratios[l] if l < len(ratios) else config.probe_ratio
        k = per[l] if l < len(per) else 1.0
        ck = cum_kappa[l] if l < len(cum_kappa) else 1.0
        if layer is None or not layer.projections:
            # No local ΔW, but upstream ΔX still moves this layer's KV.
            upstream = any(i in delta.layers for i in range(l))
            if upstream:
                plans[l] = LayerPlan(LayerStrategy.SUBSPACE, ratio, k, ck)
            else:
                plans[l] = LayerPlan(LayerStrategy.SKIP, 0.0, k, ck)
            continue
        is_boundary = l == 0 or (has_hidden and l % stride == 0)
        # Stored X + k/v ΔW is exact for this layer's KV even if q/o/mlp also move.
        if has_hidden and layer.has_kv() and is_boundary:
            plans[l] = LayerPlan(LayerStrategy.EXACT, ratio, k, ck)
            continue
        score = norm[l] if l < len(norm) else 0.0
        if score >= config.kappa_probe_threshold:
            plans[l] = LayerPlan(LayerStrategy.PROBE, max(ratio, config.probe_ratio), k, ck)
        else:
            plans[l] = LayerPlan(LayerStrategy.SUBSPACE, ratio, k, ck)
    return plans


def global_probe_ratio(plans: dict[int, LayerPlan], fallback: float) -> float:
    active = [p.probe_ratio for p in plans.values() if p.strategy is not LayerStrategy.SKIP]
    if not active:
        return fallback
    return float(max(active))
