"""KV tensors, low-rank patches, and cache entries."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from deltakv.types import ErrorEstimate, PatchRoute, WeightVersion


@dataclass
class KVCache:
    """Full KV for a prefix. Layout: ``[n_layers, seq, n_kv_heads, head_dim]``."""

    k: Tensor
    v: Tensor
    version: WeightVersion
    token_ids: Tensor | None = None

    def __post_init__(self) -> None:
        if self.k.shape != self.v.shape:
            raise ValueError(f"K/V shape mismatch {tuple(self.k.shape)} vs {tuple(self.v.shape)}")

    @property
    def n_layers(self) -> int:
        return int(self.k.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.k.shape[1])

    @property
    def n_kv_heads(self) -> int:
        return int(self.k.shape[2])

    @property
    def head_dim(self) -> int:
        return int(self.k.shape[3])

    def layer(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.k[idx], self.v[idx]

    def clone(self) -> KVCache:
        return KVCache(
            k=self.k.clone(),
            v=self.v.clone(),
            version=self.version,
            token_ids=None if self.token_ids is None else self.token_ids.clone(),
        )

    def add_layer_delta(self, idx: int, dk: Tensor, dv: Tensor) -> None:
        self.k[idx] = self.k[idx] + dk
        self.v[idx] = self.v[idx] + dv


@dataclass
class HiddenSnapshot:
    """Optional per-layer residual-stream activations for analytic patches.

    ``h[l]`` is the residual stream *entering* layer ``l`` (pre-attn), shape
    ``[seq, d_model]``. ``h_mid[l]`` is the residual *after* attention, before
    the MLP (needed for accurate SwiGLU first-order). ``embed`` is token
    embeddings. Stored in ``dtype`` (fp16/fp8).
    """

    h: tuple[Tensor, ...]
    embed: Tensor
    dtype: torch.dtype = torch.float16
    h_mid: tuple[Tensor, ...] | None = None

    def layer_in(self, idx: int) -> Tensor:
        return self.h[idx]

    def layer_mid(self, idx: int) -> Tensor:
        if self.h_mid is None:
            return self.h[idx]
        return self.h_mid[idx]

    @property
    def n_layers(self) -> int:
        return len(self.h)

    @property
    def seq_len(self) -> int:
        return int(self.embed.shape[0])


@dataclass
class LowRankKVPatch:
    """Compressed ΔKV at one layer: ΔK = codes_k @ B_k.T (then maybe RoPE already applied).

    Storage is ``n·r + r·d`` instead of ``n·d``. Exact at projection layers.
    """

    layer_idx: int
    k_codes: Tensor | None  # [seq, r_k]
    k_basis: Tensor | None  # [n_kv_heads * head_dim, r_k]  (B)
    v_codes: Tensor | None
    v_basis: Tensor | None
    k_dense: Tensor | None = None  # fallback when not low-rank (after RoPE+probe)
    v_dense: Tensor | None = None

    def materialize(self, n_kv_heads: int, head_dim: int) -> tuple[Tensor, Tensor]:
        dk = self._mat(self.k_dense, self.k_codes, self.k_basis, n_kv_heads, head_dim)
        dv = self._mat(self.v_dense, self.v_codes, self.v_basis, n_kv_heads, head_dim)
        return dk, dv

    @staticmethod
    def _mat(
        dense: Tensor | None,
        codes: Tensor | None,
        basis: Tensor | None,
        n_kv_heads: int,
        head_dim: int,
    ) -> Tensor:
        if dense is not None:
            return dense
        if codes is None or basis is None:
            raise ValueError("patch is empty")
        flat = codes @ basis.transpose(0, 1)  # [seq, n_kv * d]
        return flat.view(codes.shape[0], n_kv_heads, head_dim)

    def nbytes(self) -> int:
        n = 0
        for t in (self.k_codes, self.k_basis, self.v_codes, self.v_basis, self.k_dense, self.v_dense):
            if t is not None:
                n += t.numel() * t.element_size()
        return n


@dataclass
class AppliedPatch:
    source: WeightVersion
    target: WeightVersion
    route: PatchRoute
    layer_patches: dict[int, LowRankKVPatch]
    error: ErrorEstimate
    probe_index: Tensor | None = None
    recompute_index: Tensor | None = None


@dataclass
class CacheEntry:
    """Materialized-view row: ``(base_kv, patch_chain, error_estimate)``."""

    key_hash: str
    token_ids: Tensor
    base: KVCache
    hidden: HiddenSnapshot | None
    chain: list[AppliedPatch] = field(default_factory=list)
    hits: int = 0

    @property
    def current_version(self) -> WeightVersion:
        if self.chain:
            return self.chain[-1].target
        return self.base.version

    @property
    def accumulated_error(self) -> ErrorEstimate:
        if not self.chain:
            return ErrorEstimate(relative_kv_l2=0.0, route=PatchRoute.EXACT, measured=True)
        # Conservative: sum relative errors (over-estimate, never under).
        rel = sum(p.error.relative_kv_l2 for p in self.chain)
        layers: list[float] = []
        for p in self.chain:
            if p.error.per_layer:
                if not layers:
                    layers = list(p.error.per_layer)
                else:
                    layers = [a + b for a, b in zip(layers, p.error.per_layer, strict=False)]
        return ErrorEstimate(
            relative_kv_l2=rel,
            per_layer=tuple(layers),
            route=self.chain[-1].route,
            measured=all(p.error.measured for p in self.chain),
        )

    def nbytes(self) -> int:
        n = self.base.k.numel() * self.base.k.element_size() * 2
        if self.hidden is not None:
            n += self.hidden.embed.numel() * self.hidden.embed.element_size()
            n += sum(h.numel() * h.element_size() for h in self.hidden.h)
        for p in self.chain:
            n += sum(lp.nbytes() for lp in p.layer_patches.values())
        return n
