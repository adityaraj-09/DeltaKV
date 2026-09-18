"""KL-bounded RL policy step: dense ΔW, SVD-sketched for the patch path."""

from __future__ import annotations

from torch import Tensor

from deltakv.deltas.descriptor import LayerDelta, WeightDelta
from deltakv.deltas.factors import from_dense
from deltakv.types import DeltaKind, WeightVersion


def rl_step_delta(
    *,
    source: WeightVersion,
    target: WeightVersion,
    dense_deltas: dict[tuple[int, str], Tensor],
    rank: int = 64,
    relative_scale: float = 0.0,
    notes: str = "rl_step",
) -> WeightDelta:
    """Broadcast payload after an RL / small-LR SFT step.

    The serving node never needs the full dense ΔW — a rank-``r`` sketch is
    enough for the exact projection patch plus analytic/probe propagation.
    """
    layers: dict[int, LayerDelta] = {}
    for (idx, name), dw in dense_deltas.items():
        fac = from_dense(dw, rank=rank, name=name)
        layers.setdefault(idx, LayerDelta(layer_idx=idx)).projections[name] = fac
    return WeightDelta(
        kind=DeltaKind.RL_STEP,
        source=source,
        target=target,
        layers=layers,
        dense_scale=relative_scale,
        notes=notes,
    )
