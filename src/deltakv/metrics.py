"""Quality metrics: KV error, next-token KL, logprob L1, KIVI-4bit floor."""

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
    logprob_mae: float | None = None
    logprob_max: float | None = None
    logit_l2: float | None = None
    kivi4_floor: float | None = None
    within_kivi4: bool | None = None

    def as_dict(self) -> dict[str, float | bool | None]:
        return {
            "max_relative_l2": self.max_relative_l2,
            "mean_relative_l2": self.mean_relative_l2,
            "min_cosine": self.min_cosine,
            "next_token_kl": self.next_token_kl,
            "logprob_mae": self.logprob_mae,
            "logprob_max": self.logprob_max,
            "logit_l2": self.logit_l2,
            "kivi4_floor": self.kivi4_floor,
            "within_kivi4": self.within_kivi4,
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


def _last_logits(t: Tensor) -> Tensor:
    while t.ndim > 1:
        t = t[-1]
    if t.ndim != 1:
        raise ValueError(f"expected a vocab vector, got {tuple(t.shape)}")
    return t


def next_token_kl(logits_p: Tensor, logits_q: Tensor) -> float:
    """KL(P || Q) in nats for the last-position next-token distribution."""
    log_p = F.log_softmax(_last_logits(logits_p).float(), dim=-1)
    log_q = F.log_softmax(_last_logits(logits_q).float(), dim=-1)
    p = log_p.exp()
    kl = torch.sum(p * (log_p - log_q))
    return float(kl.clamp_min(0.0).item())


def logprob_error(logits_fresh: Tensor, logits_patched: Tensor) -> tuple[float, float]:
    """MAE and max of |log π_fresh − log π_patched| (RL / TIS scale)."""
    log_f = F.log_softmax(_last_logits(logits_fresh).float(), dim=-1)
    log_p = F.log_softmax(_last_logits(logits_patched).float(), dim=-1)
    delta = (log_f - log_p).abs()
    return float(delta.mean().item()), float(delta.max().item())


def logit_l2(logits_fresh: Tensor, logits_patched: Tensor) -> float:
    a = _last_logits(logits_fresh).float()
    b = _last_logits(logits_patched).float()
    return float((a - b).norm() / (a.norm() + 1e-8))


def _minmax_quantize(t: Tensor, bits: int, group_size: int) -> Tensor:
    """Per-group min-max fake-quant along the last dim (KIVI-style floor)."""
    orig = t.shape
    d = orig[-1]
    flat = t.reshape(-1, d)
    g = min(group_size, d)
    pad = (g - d % g) % g
    if pad:
        flat = F.pad(flat, (0, pad))
    grouped = flat.view(flat.shape[0], -1, g)
    lo = grouped.amin(dim=-1, keepdim=True)
    hi = grouped.amax(dim=-1, keepdim=True)
    levels = max(1, (1 << bits) - 1)
    scale = (hi - lo).clamp_min(1e-8) / levels
    q = torch.round((grouped - lo) / scale).clamp(0, levels)
    deq = q * scale + lo
    deq = deq.view(flat.shape[0], -1)[:, :d]
    return deq.view(*orig)


def kivi_noise_floor(fresh: KVCache, bits: int = 4, group_size: int = 32) -> float:
    """Relative L2 between fresh KV and a 4-bit group-quantized copy."""
    rels = []
    for l in range(fresh.n_layers):
        k, v = fresh.layer(l)
        kq, vq = _minmax_quantize(k, bits, group_size), _minmax_quantize(v, bits, group_size)
        rels.append(max(_rel_l2(kq, k), _rel_l2(vq, v)))
    return float(max(rels) if rels else 0.0)


def attach_quality(
    report: CompareReport,
    patched_logits: Tensor,
    fresh_logits: Tensor,
    fresh_kv: KVCache | None = None,
) -> CompareReport:
    """Fill serving/RL metrics: KL, logprob L1, logit L2, KIVI-4bit floor."""
    report.next_token_kl = next_token_kl(fresh_logits, patched_logits)
    mae, mx = logprob_error(fresh_logits, patched_logits)
    report.logprob_mae = mae
    report.logprob_max = mx
    report.logit_l2 = logit_l2(fresh_logits, patched_logits)
    if fresh_kv is not None:
        report.kivi4_floor = kivi_noise_floor(fresh_kv, bits=4)
        report.within_kivi4 = report.max_relative_l2 <= report.kivi4_floor
    return report


def attach_kl(report: CompareReport, patched_logits: Tensor, fresh_logits: Tensor) -> CompareReport:
    return attach_quality(report, patched_logits, fresh_logits)
