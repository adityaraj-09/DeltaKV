"""LoRA / adapter-swap deltas. Matches PEFT: y = Wx + (α/r) B(Ax)."""

from __future__ import annotations

from typing import Any, Iterable

from deltakv.deltas.descriptor import LayerDelta, WeightDelta, lora_layer
from deltakv.deltas.factors import LowRankFactors, difference
from deltakv.types import DeltaKind, WeightVersion


def lora_delta(
    *,
    source: WeightVersion,
    target: WeightVersion,
    layers: dict[int, LayerDelta],
    notes: str = "",
) -> WeightDelta:
    return WeightDelta(
        kind=DeltaKind.LORA,
        source=source,
        target=target,
        layers=layers,
        notes=notes or "lora",
    )


def adapter_swap(
    old: WeightDelta | None,
    new: WeightDelta,
) -> WeightDelta:
    """Physical shared-prefix cache: patch from adapter A to adapter B.

    ΔW = ΔW_B − ΔW_A (rank ≤ r_A + r_B). ``old=None`` is base → B.
    """
    if old is None:
        return new
    if old.source.id != new.source.id:
        raise ValueError("adapter swap requires a shared base version")
    layers: dict[int, LayerDelta] = {}
    for idx in set(old.layers) | set(new.layers):
        a = old.layers.get(idx)
        b = new.layers.get(idx)
        if a is None:
            layers[idx] = b  # type: ignore[assignment]
            continue
        if b is None:
            # Removing an adapter: negate old factors.
            proj = {
                name: LowRankFactors(A=f.A, B=f.B, scale=-f.scale, name=f.name)
                for name, f in a.projections.items()
            }
            layers[idx] = LayerDelta(layer_idx=idx, projections=proj)
            continue
        proj = {}
        names = set(a.projections) | set(b.projections)
        for name in names:
            fa, fb = a.projections.get(name), b.projections.get(name)
            if fb is None:
                proj[name] = LowRankFactors(A=fa.A, B=fa.B, scale=-fa.scale, name=fa.name)  # type: ignore[union-attr]
            elif fa is None:
                proj[name] = fb
            else:
                proj[name] = difference(fb, fa)
        layers[idx] = LayerDelta(layer_idx=idx, projections=proj)
    return WeightDelta(
        kind=DeltaKind.LORA,
        source=old.target,
        target=new.target,
        layers=layers,
        notes=f"swap {old.target.id} -> {new.target.id}",
    )


def from_peft_state_dict(
    state: dict[str, Any],
    *,
    source: WeightVersion,
    target: WeightVersion,
    alpha: float | None = None,
    rank: int | None = None,
    layer_prefix: str = "model.layers",
) -> WeightDelta:
    """Build a WeightDelta from a PEFT / HuggingFace LoRA state dict.

    Keys look like ``base_model.model.model.layers.{i}.self_attn.k_proj.lora_A.weight``.
    We accept any key that contains ``layers.{i}`` and a projection name plus
    ``lora_A`` / ``lora_B``. Scaling is ``alpha / rank`` when both are given,
    otherwise 1.0 (caller should pass the PEFT values).
    """
    scale = 1.0
    if alpha is not None and rank:
        scale = float(alpha) / float(rank)

    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    proj_aliases = {
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "query": "q_proj",
        "key": "k_proj",
        "value": "v_proj",
        "dense": "o_proj",
        "Wqkv": "q_proj",
    }
    for key, tensor in state.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
        layer_idx = _parse_layer_index(key)
        if layer_idx is None:
            continue
        proj = None
        for raw, canon in proj_aliases.items():
            if f".{raw}." in key or key.endswith(raw) or f"{raw}.lora" in key:
                proj = canon
                break
        if proj is None:
            continue
        slot = grouped.setdefault((layer_idx, proj), {})
        if "lora_A" in key:
            slot["A"] = tensor
        else:
            slot["B"] = tensor

    layers: dict[int, LayerDelta] = {}
    by_layer: dict[int, dict[str, LowRankFactors]] = {}
    for (idx, proj), mats in grouped.items():
        if "A" not in mats or "B" not in mats:
            continue
        fac = LowRankFactors(A=mats["A"], B=mats["B"], scale=scale, name=proj)
        by_layer.setdefault(idx, {})[proj] = fac
    for idx, proj in by_layer.items():
        layers[idx] = LayerDelta(layer_idx=idx, projections=proj)
    return lora_delta(source=source, target=target, layers=layers, notes="peft")


def _parse_layer_index(key: str) -> int | None:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p in {"layers", "h", "blocks"} and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def stack_layer_deltas(items: Iterable[tuple[int, LayerDelta]]) -> dict[int, LayerDelta]:
    return {i: d for i, d in items}


__all__ = [
    "adapter_swap",
    "from_peft_state_dict",
    "lora_delta",
    "lora_layer",
    "stack_layer_deltas",
]
