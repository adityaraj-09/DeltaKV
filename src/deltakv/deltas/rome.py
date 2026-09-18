"""ROME / MEMIT rank-one (or low-rank) model edits."""

from __future__ import annotations

from torch import Tensor

from deltakv.deltas.descriptor import LayerDelta, WeightDelta
from deltakv.deltas.factors import from_outer
from deltakv.types import DeltaKind, WeightVersion


def rome_delta(
    *,
    source: WeightVersion,
    target: WeightVersion,
    layer_idx: int,
    projection: str,
    u: Tensor,
    v: Tensor,
    scale: float = 1.0,
    notes: str = "rome",
) -> WeightDelta:
    """Closed-form patch case: a single rank-1 write into one linear map.

    ROME typically writes ``down_proj`` (or ``mlp.down_proj``) of one mid
    layer. ΔKV then only needs that layer's factors plus probe correction
    for the layers above.
    """
    fac = from_outer(u, v, scale=scale, name=projection)
    layer = LayerDelta(layer_idx=layer_idx, projections={projection: fac})
    return WeightDelta(
        kind=DeltaKind.ROME,
        source=source,
        target=target,
        layers={layer_idx: layer},
        notes=notes,
    )
