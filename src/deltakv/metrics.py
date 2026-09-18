"""Quality metrics: per-layer KV error, cosine, next-token KL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from deltakv.patches.tensors import KVCache


@dataclass
class LayerKVError:
    layer: int
    relative_l2_k: float
    relative_l2_v: float
    cosine_k: float
    cosine_v: float


@dataclass
class CompareReport:
    layers: list[LayerKVError]
    max_relative_l2: float
    mean_relative_l2: float
    min_cosine: float
    next_token_kl: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "max_relative_l2": self.max_relative_l2,
            "mean_relative_l2": self.mean_relative_l2,
            "min_cosine": self.min_cosine,
            "next_token_kl": self.next_token_kl,
        }


def _rel_l2(a: Tensor, b: Tensor) -> float:
    return float((a - b).norm() / (b.norm() + 1e-8))


def _cosine(a: Tensor, b: Tensor) -> float:
    x = a.flatten().float()
    y = b.flatten().float()
    return float(F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item())


def compare_kv(patched: KVCache, fresh: KVCache) -> CompareReport:
    layers: list[LayerKVError] = []
    for l in range(patched.n_layers):
        pk, pv = patched.layer(l)
        fk, fv = fresh.layer(l)
        layers.append(
            LayerKVError(
                layer=l,
                relative_l2_k=_rel_l2(pk, fk),
                relative_l2_v=_rel_l2(pv, fv),
                cosine_k=_cosine(pk, fk),
                cosine_v=_cosine(pv, fv),
            )
        )
    rels = [max(x.relative_l2_k, x.relative_l2_v) for x in layers]
    cos = [min(x.cosine_k, x.cosine_v) for x in layers]
    return CompareReport(
        layers=layers,
        max_relative_l2=max(rels) if rels else 0.0,
        mean_relative_l2=float(sum(rels) / max(len(rels), 1)),
        min_cosine=min(cos) if cos else 1.0,
    )


def next_token_kl(logits_p: Tensor, logits_q: Tensor) -> float:
    """KL(P || Q) in nats for the last-position next-token distribution.

    ``logits_*`` may be ``[vocab]`` or ``[seq, vocab]`` (last row used).
    """

    def _last(t: Tensor) -> Tensor:
        while t.ndim > 1:
            t = t[-1]
        if t.ndim != 1:
            raise ValueError(f"expected a vocab vector, got {tuple(t.shape)}")
        return t

    log_p = F.log_softmax(_last(logits_p).float(), dim=-1)
    log_q = F.log_softmax(_last(logits_q).float(), dim=-1)
    p = log_p.exp()
    kl = torch.sum(p * (log_p - log_q))
    return float(kl.clamp_min(0.0).item())


def attach_kl(report: CompareReport, patched_logits: Tensor, fresh_logits: Tensor) -> CompareReport:
    report.next_token_kl = next_token_kl(fresh_logits, patched_logits)
    return report
