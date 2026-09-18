"""Low-rank ΔW factors using the PEFT convention: ΔW = scale · B @ A."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LowRankFactors:
    """PEFT-style LoRA factors for a single linear map.

    ``A`` is ``[rank, in_features]``, ``B`` is ``[out_features, rank]``.
    HuggingFace Linear computes ``y = x @ W.T``, so the adapter path is::

        Δy = scale * (x @ A.T) @ B.T

    Every per-token Δy therefore lives in the column space of ``B``
    (dimension ``rank``), which is the compression the rest of ΔKV exploits.
    """

    A: Tensor
    B: Tensor
    scale: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or self.B.ndim != 2:
            raise ValueError("A and B must be rank-2")
        if self.A.shape[0] != self.B.shape[1]:
            raise ValueError(
                f"rank mismatch: A {tuple(self.A.shape)} vs B {tuple(self.B.shape)}"
            )

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.A.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.B.shape[0])

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> LowRankFactors:
        return LowRankFactors(
            A=self.A.to(device=device, dtype=dtype),
            B=self.B.to(device=device, dtype=dtype),
            scale=self.scale,
            name=self.name,
        )

    def delta_w(self) -> Tensor:
        """Materialize dense ΔW of shape ``[out, in]``."""
        return self.scale * (self.B @ self.A)

    def codes(self, x: Tensor) -> Tensor:
        """``[..., in] → [..., rank]``. Cheap token-wise coefficients."""
        return self.scale * (x @ self.A.transpose(-1, -2))

    def apply(self, x: Tensor) -> Tensor:
        """``[..., in] → [..., out]``. Exact adapter path."""
        return self.codes(x) @ self.B.transpose(-1, -2)

    def frobenius_norm(self) -> Tensor:
        # ||s B A||_F^2 = s^2 tr(A A^T B^T B) but we just materialize for small r
        # Cheap bound: s * ||B||_F * ||A||_F
        return self.scale * self.B.norm() * self.A.norm()

    def exact_frobenius_norm(self) -> Tensor:
        return self.delta_w().norm()


def concat_factors(left: LowRankFactors, right: LowRankFactors, sign_right: float = 1.0) -> LowRankFactors:
    """ΔW_left + sign_right · ΔW_right as a single factor pair (rank adds)."""
    if left.in_features != right.in_features or left.out_features != right.out_features:
        raise ValueError("factor shapes do not broadcast")
    A = torch.cat([left.A, right.A], dim=0)
    B = torch.cat([left.scale * left.B, (sign_right * right.scale) * right.B], dim=1)
    return LowRankFactors(A=A, B=B, scale=1.0, name=f"{left.name}+{right.name}")


def difference(new: LowRankFactors, old: LowRankFactors | None) -> LowRankFactors:
    """Adapter swap: ΔW = ΔW_new − ΔW_old. ``old=None`` means base → new."""
    if old is None:
        return new
    return concat_factors(new, old, sign_right=-1.0)


def from_dense(delta_w: Tensor, rank: int, name: str = "") -> LowRankFactors:
    """Truncated SVD of a dense ΔW into PEFT-shaped factors."""
    if delta_w.ndim != 2:
        raise ValueError("delta_w must be [out, in]")
    # torch.linalg.svd is float32-stable on CPU
    dw = delta_w.float()
    u, s, vh = torch.linalg.svd(dw, full_matrices=False)
    r = max(1, min(int(rank), int(s.numel())))
    A = torch.diag(s[:r]) @ vh[:r]
    B = u[:, :r]
    return LowRankFactors(A=A.to(delta_w.dtype), B=B.to(delta_w.dtype), scale=1.0, name=name)


def random_lora_factors(
    in_features: int,
    out_features: int,
    rank: int,
    scale: float = 0.02,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    name: str = "",
) -> LowRankFactors:
    """Small random adapter used by tests and the bench harness."""
    A = torch.randn(rank, in_features, device=device, dtype=dtype, generator=generator)
    B = torch.randn(out_features, rank, device=device, dtype=dtype, generator=generator)
    B = B / (B.norm() + 1e-8)
    A = A / (A.norm() + 1e-8)
    return LowRankFactors(A=A, B=B, scale=scale, name=name)


def from_outer(u: Tensor, v: Tensor, scale: float = 1.0, name: str = "") -> LowRankFactors:
    """Rank-1 (ROME/MEMIT) update ΔW = scale · u v^T.

    ``u`` is ``[out]`` or ``[out, 1]``, ``v`` is ``[in]`` or ``[in, 1]``.
    """
    u = u.reshape(-1, 1)
    v = v.reshape(1, -1)
    return LowRankFactors(A=v, B=u, scale=scale, name=name)
