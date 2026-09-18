"""Weight-delta descriptors shipped alongside a version bump."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from deltakv.deltas.factors import (
    LowRankFactors,
    difference,
    from_dense,
    from_outer,
    random_lora_factors,
)
from deltakv.types import DeltaKind, WeightVersion

# Canonical linear names inside one decoder block.
PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class LayerDelta:
    """Per-block collection of low-rank (or SVD-compressed) ΔW factors."""

    layer_idx: int
    projections: dict[str, LowRankFactors] = field(default_factory=dict)

    def get(self, name: str) -> LowRankFactors | None:
        return self.projections.get(name)

    @property
    def max_rank(self) -> int:
        if not self.projections:
            return 0
        return max(f.rank for f in self.projections.values())

    def frobenius(self) -> float:
        if not self.projections:
            return 0.0
        return float(torch.stack([f.frobenius_norm() for f in self.projections.values()]).sum())

    def has_kv(self) -> bool:
        return "k_proj" in self.projections or "v_proj" in self.projections


@dataclass
class WeightDelta:
    """Compact descriptor for W → W+ΔW. This is what a serving node broadcasts.

    For LoRA the payload is the adapter factors themselves (kilobytes).
    For ROME it is a rank-1 pair. For quant/RL it is an optional SVD sketch.
    """

    kind: DeltaKind
    source: WeightVersion
    target: WeightVersion
    layers: dict[int, LayerDelta] = field(default_factory=dict)
    dense_scale: float = 0.0  # ||ΔW|| / ||W|| proxy when dense
    notes: str = ""

    @property
    def n_layers(self) -> int:
        return (max(self.layers) + 1) if self.layers else 0

    @property
    def max_rank(self) -> int:
        if not self.layers:
            return 0
        return max(ld.max_rank for ld in self.layers.values())

    @property
    def first_modified_layer(self) -> int:
        if not self.layers:
            return 0
        return min(self.layers)

    def kv_layers(self) -> list[int]:
        return sorted(i for i, ld in self.layers.items() if ld.has_kv())

    def relative_magnitude(self) -> float:
        """Proxy for ‖ΔW‖ / ‖W‖. Uses an explicit scale when the caller
        knows it (quant / RL); otherwise the largest single-map LoRA-style
        Frobenius (not a sum across q/k/v, which would triple-count)."""
        if self.dense_scale > 0:
            return self.dense_scale
        if not self.layers:
            return 0.0
        m = 0.0
        for ld in self.layers.values():
            for f in ld.projections.values():
                m = max(m, float(f.frobenius_norm()))
        return m

    def payload_values(self) -> int:
        """Number of stored factor values (the ~64× compression claim)."""
        total = 0
        for ld in self.layers.values():
            for f in ld.projections.values():
                total += f.A.numel() + f.B.numel()
        return total


def empty_layer(idx: int) -> LayerDelta:
    return LayerDelta(layer_idx=idx)


def lora_layer(
    idx: int,
    *,
    k: LowRankFactors | None = None,
    v: LowRankFactors | None = None,
    q: LowRankFactors | None = None,
    o: LowRankFactors | None = None,
    gate: LowRankFactors | None = None,
    up: LowRankFactors | None = None,
    down: LowRankFactors | None = None,
) -> LayerDelta:
    proj: dict[str, LowRankFactors] = {}
    mapping = {
        "k_proj": k,
        "v_proj": v,
        "q_proj": q,
        "o_proj": o,
        "gate_proj": gate,
        "up_proj": up,
        "down_proj": down,
    }
    for name, fac in mapping.items():
        if fac is not None:
            fac.name = fac.name or name
            proj[name] = fac
    return LayerDelta(layer_idx=idx, projections=proj)


def compose_deltas(first: WeightDelta, second: WeightDelta) -> WeightDelta:
    """Rebase: apply ``second`` after ``first``. Ranks add; errors tracked elsewhere."""
    if first.target.id != second.source.id:
        raise ValueError(
            f"delta lineage break: {first.target.id} != {second.source.id}"
        )
    layers: dict[int, LayerDelta] = {}
    for idx in set(first.layers) | set(second.layers):
        a = first.layers.get(idx)
        b = second.layers.get(idx)
        if a is None:
            layers[idx] = b  # type: ignore[assignment]
            continue
        if b is None:
            layers[idx] = a
            continue
        from deltakv.deltas.factors import concat_factors

        proj: dict[str, LowRankFactors] = {}
        names = set(a.projections) | set(b.projections)
        for name in names:
            fa, fb = a.projections.get(name), b.projections.get(name)
            if fa is None:
                proj[name] = fb  # type: ignore[assignment]
            elif fb is None:
                proj[name] = fa
            else:
                proj[name] = concat_factors(fa, fb, sign_right=1.0)
        layers[idx] = LayerDelta(layer_idx=idx, projections=proj)
    kind = second.kind if first.kind == second.kind else DeltaKind.DENSE
    return WeightDelta(
        kind=kind,
        source=first.source,
        target=second.target,
        layers=layers,
        dense_scale=max(first.dense_scale, second.dense_scale),
        notes=f"compose({first.target.id},{second.target.id})",
    )


# Re-export constructors used by connectors / tests.
__all__ = [
    "LayerDelta",
    "PROJ_NAMES",
    "WeightDelta",
    "compose_deltas",
    "difference",
    "empty_layer",
    "from_dense",
    "from_outer",
    "lora_layer",
    "random_lora_factors",
]
