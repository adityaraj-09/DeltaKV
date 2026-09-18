"""Quantization-tier ΔW = W_qnew − W_qold, optionally SVD-compressed."""

from __future__ import annotations

from torch import Tensor

from deltakv.deltas.descriptor import LayerDelta, WeightDelta
from deltakv.deltas.factors import from_dense
from deltakv.types import DeltaKind, WeightVersion

_NAME_MAP = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def quant_delta(
    *,
    source: WeightVersion,
    target: WeightVersion,
    dense_deltas: dict[tuple[int, str], Tensor],
    rank: int = 64,
    relative_scale: float = 0.0,
    notes: str = "quant",
) -> WeightDelta:
    """``dense_deltas`` maps ``(layer_idx, proj_name)`` → dense ``[out, in]`` ΔW."""
    layers: dict[int, LayerDelta] = {}
    for (idx, name), dw in dense_deltas.items():
        canon = _NAME_MAP.get(name, name)
        fac = from_dense(dw, rank=rank, name=canon)
        layers.setdefault(idx, LayerDelta(layer_idx=idx)).projections[canon] = fac
    return WeightDelta(
        kind=DeltaKind.QUANT,
        source=source,
        target=target,
        layers=layers,
        dense_scale=relative_scale,
        notes=notes,
    )
