"""FLOP accounting for the cost-based patch / recompute optimizer."""

from __future__ import annotations

from dataclasses import dataclass

from deltakv.deltas.descriptor import WeightDelta


@dataclass
class ModelCostDims:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    seq_len: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def prefill_flops(dims: ModelCostDims) -> float:
    """Approximate dense prefill FLOPs (matmul-dominated)."""
    n, d, L, ff = dims.seq_len, dims.d_model, dims.n_layers, dims.d_ff
    d_kv = dims.n_kv_heads * dims.head_dim
    attn_proj = n * d * (d + 2 * d_kv + d)  # q, k, v, o  (2 FLOPs folded as 1 MAC≈2; we count MACs)
    attn_scores = dims.n_heads * n * n * dims.head_dim
    mlp = n * (2 * d * ff + ff * d)  # gate, up, down
    return float(L * (attn_proj + attn_scores + mlp))


def exact_patch_flops(dims: ModelCostDims, delta: WeightDelta) -> float:
    """O(n · d · r) adapter path on k/v (and any other stored projections)."""
    n, d = dims.seq_len, dims.d_model
    total = 0.0
    for layer in delta.layers.values():
        for fac in layer.projections.values():
            # codes: n * in * r, materialize: n * r * out
            total += n * fac.in_features * fac.rank + n * fac.rank * fac.out_features
    return float(total)


def analytic_flops(dims: ModelCostDims, rank: int) -> float:
    """Low-rank ΔS / ΔV plus linearized MLP sketch."""
    n, d, L, h = dims.seq_len, dims.d_model, dims.n_layers, dims.n_heads
    attn = L * h * (n * d * rank + n * n * rank)
    mlp = L * n * d * rank
    return float(attn + mlp)


def probe_flops(dims: ModelCostDims, probe_ratio: float) -> float:
    return float(probe_ratio * prefill_flops(dims) + dims.n_layers * dims.seq_len * dims.d_model)


def patch_storage_values(n_tokens: int, d_kv: int, rank: int) -> tuple[int, int, float]:
    """Return ``(compressed, full_kv_layer, compression_ratio)`` for one K (or V) tensor."""
    compressed = n_tokens * rank + rank * d_kv
    full = n_tokens * d_kv
    ratio = full / max(compressed, 1)
    return compressed, full, ratio
