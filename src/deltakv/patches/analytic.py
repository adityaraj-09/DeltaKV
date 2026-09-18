"""First-order / low-rank hidden-state propagation (Route A)."""

from __future__ import annotations

import torch
from torch import Tensor

from deltakv.deltas.descriptor import LayerDelta, WeightDelta
from deltakv.patches.exact import packed_delta
from deltakv.patches.tensors import LowRankKVPatch
from deltakv.rope import RotaryEmbedding, merge_heads, reshape_for_heads


def rank_truncate(delta: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Δ ≈ codes @ basis.T with ``codes [n, r]``, ``basis [d, r]``."""
    flat = delta.reshape(delta.shape[0], -1).float()
    # economy SVD on [n, d]
    u, s, vh = torch.linalg.svd(flat, full_matrices=False)
    r = max(1, min(rank, int(s.numel())))
    codes = u[:, :r] * s[:r]
    basis = vh[:r].transpose(0, 1).contiguous()
    return codes.to(delta.dtype), basis.to(delta.dtype)


def softmax_jacobian_product(attn: Tensor, d_scores: Tensor) -> Tensor:
    """``ΔA = J_softmax(A)[ΔS]`` for a causal attention matrix.

    ``attn`` and ``d_scores`` are ``[seq, seq]`` (or ``[heads, seq, seq]``).
    Row-wise: ``a ⊙ (δ − ⟨a, δ⟩)``.
    """
    # attn: [..., q, k]
    centered = d_scores - (attn * d_scores).sum(dim=-1, keepdim=True)
    return attn * centered


def attention_first_order(
    x: Tensor,
    dx: Tensor,
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor,
    layer: LayerDelta | None,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope: RotaryEmbedding,
    causal_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(Δh_attn, ΔK, ΔV, attn_probs)`` for one block.

    Uses the *original* Q,K,V (from ``x`` and base weights) plus first-order
    ΔQ,ΔK,ΔV. Softmax Jacobian is exact-linear; the ΔX ΔW product is dropped
    (second order). GQA: K/V heads are repeated to match Q heads.
    """
    seq = x.shape[0]
    scale = head_dim ** -0.5
    q = reshape_for_heads(x @ q_w.t(), n_heads, head_dim)
    k = reshape_for_heads(x @ k_w.t(), n_kv_heads, head_dim)
    v = reshape_for_heads(x @ v_w.t(), n_kv_heads, head_dim)
    q = rope(q)
    k = rope(k)

    dq = reshape_for_heads(dx @ q_w.t(), n_heads, head_dim)
    dk = reshape_for_heads(dx @ k_w.t(), n_kv_heads, head_dim)
    dv = reshape_for_heads(dx @ v_w.t(), n_kv_heads, head_dim)
    if layer is not None:
        dq = dq + packed_delta(x, layer.get("q_proj"), n_heads, head_dim)
        dk = dk + packed_delta(x, layer.get("k_proj"), n_kv_heads, head_dim)
        dv = dv + packed_delta(x, layer.get("v_proj"), n_kv_heads, head_dim)
    dq = rope(dq)
    dk = rope(dk)

    k_full, dk_full, v_full, dv_full = _repeat_kv(k, dk, v, dv, n_heads, n_kv_heads)

    # scores: [heads, seq, seq]
    scores = torch.einsum("qhd,khd->hqk", q, k_full) * scale
    scores = scores + causal_mask
    attn = torch.softmax(scores, dim=-1)

    d_scores = (
        torch.einsum("qhd,khd->hqk", dq, k_full) + torch.einsum("qhd,khd->hqk", q, dk_full)
    ) * scale
    d_attn = softmax_jacobian_product(attn, d_scores)

    ctx = torch.einsum("hqk,khd->qhd", attn, v_full)
    d_ctx = torch.einsum("hqk,khd->qhd", d_attn, v_full) + torch.einsum("hqk,khd->qhd", attn, dv_full)
    dh = merge_heads(d_ctx) @ o_w.t()
    if layer is not None and layer.get("o_proj") is not None:
        dh = dh + layer.get("o_proj").apply(merge_heads(ctx))  # type: ignore[union-attr]
    return dh, dk, dv, attn


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def silu_grad(x: Tensor) -> Tensor:
    s = torch.sigmoid(x)
    return s + x * s * (1.0 - s)


def mlp_first_order(
    x: Tensor,
    dx: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    layer: LayerDelta | None,
) -> Tensor:
    """First-order SwiGLU: ``silu(x Wg) ⊙ (x Wu)`` then ``Wdown``."""
    g = x @ gate_w.t()
    u = x @ up_w.t()
    dg = dx @ gate_w.t()
    du = dx @ up_w.t()
    if layer is not None:
        if layer.get("gate_proj") is not None:
            dg = dg + layer.get("gate_proj").apply(x)  # type: ignore[union-attr]
        if layer.get("up_proj") is not None:
            du = du + layer.get("up_proj").apply(x)  # type: ignore[union-attr]
    h = silu(g) * u
    dh = silu_grad(g) * dg * u + silu(g) * du
    y = dh @ down_w.t()
    if layer is not None and layer.get("down_proj") is not None:
        y = y + layer.get("down_proj").apply(h)  # type: ignore[union-attr]
    return y


def _repeat_kv(
    k: Tensor, dk: Tensor, v: Tensor, dv: Tensor, n_heads: int, n_kv_heads: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if n_heads == n_kv_heads:
        return k, dk, v, dv
    rep = n_heads // n_kv_heads
    def exp(t: Tensor) -> Tensor:
        # [seq, kv, d] -> [seq, h, d]
        return t.repeat_interleave(rep, dim=1)
    return exp(k), exp(dk), exp(v), exp(dv)


def analytic_kv_patches(
    hidden_in: dict[int, Tensor],
    delta: WeightDelta,
    weights: "BlockWeights",
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope: RotaryEmbedding,
    rms_eps: float = 1e-6,
    propagator_rank: int = 64,
) -> tuple[dict[int, LowRankKVPatch], dict[int, Tensor]]:
    """Propagate ΔX through the stack; emit per-layer ΔKV.

    ``hidden_in[l]`` is the residual stream entering layer ``l`` under the
    *old* weights (LMCache HiddenStateStore / local snapshot). ΔX starts at 0
    (embeddings usually have no LoRA) and accumulates.
    """
    seq = next(iter(hidden_in.values())).shape[0]
    device = next(iter(hidden_in.values())).device
    causal = torch.triu(hidden_in[0].new_full((seq, seq), float("-inf")), diagonal=1)
    causal = causal.unsqueeze(0)  # [1, seq, seq] broadcasts over heads

    dx = hidden_in[0].new_zeros(seq, hidden_in[0].shape[-1])
    patches: dict[int, LowRankKVPatch] = {}
    dx_layers: dict[int, Tensor] = {}
    n_layers = weights.n_layers

    for l in range(n_layers):
        x = hidden_in[l]
        dx_layers[l] = dx
        x_n, dx_n = rmsnorm_first_order(x, dx, weights.attn_scale[l], rms_eps)
        layer = delta.layers.get(l)
        dh, dk, dv, _ = attention_first_order(
            x_n,
            dx_n,
            weights.q[l],
            weights.k[l],
            weights.v[l],
            weights.o[l],
            layer,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope=rope,
            causal_mask=causal,
        )
        patches[l] = LowRankKVPatch(
            layer_idx=l,
            k_codes=None,
            k_basis=None,
            v_codes=None,
            v_basis=None,
            k_dense=dk,
            v_dense=dv,
        )
        x2 = x + weights.attn_out_placeholder(x)  # residual already in x_next
        # residual stream after attn is hidden_in[l+1] minus mlp... we don't have
        # that split. Use stored next residual and attribute Δ to attn+mlp.
        # Practical: dx := dx + dh, then MLP on post-attn residual estimate.
        dx = dx + dh
        # Post-attn residual ≈ stored next-layer input before MLP? Toy engine
        # stores residual-into-layer only. Approximate post-attn as x + attn(x)
        # without re-running attn: use x (stale) + dh as the MLP input base.
        x_post = x + (hidden_in[l + 1] - x) * 0.0 + x  # keep x; first-order MLP on x
        # Better approximation: MLP sees RMSNorm(x + attn(x)). We don't cache
        # attn(x). Use RMSNorm(x) as a cheap proxy plus dx.
        x_m, dx_m = rmsnorm_first_order(x, dx, weights.mlp_scale[l], rms_eps)
        d_mlp = mlp_first_order(
            x_m, dx_m, weights.gate[l], weights.up[l], weights.down[l], layer
        )
        dx = dx + d_mlp
        if propagator_rank and dx.numel() > 0:
            codes, basis = rank_truncate(dx, propagator_rank)
            dx = codes @ basis.t()
    return patches, dx_layers


def rmsnorm_first_order(
    x: Tensor, dx: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    """RMSNorm(x) and its first-order image under Δx.

    ``y = x / rms · w``. The Jacobian is ``(I - x x^T / ||x||^2) / rms * w``
    applied row-wise (plus the usual eps).
    """
    # x, dx: [seq, d]
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(ms + eps)
    y = x / rms * weight
    d_ms = 2.0 * (x * dx).mean(dim=-1, keepdim=True)
    d_rms = 0.5 * d_ms / rms
    dy = (dx * rms - x * d_rms) / (rms * rms) * weight
    return y, dy


class BlockWeights:
    """Base-weight views the analytic propagator needs. Filled by the toy / HF connector."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.q: list[Tensor] = []
        self.k: list[Tensor] = []
        self.v: list[Tensor] = []
        self.o: list[Tensor] = []
        self.gate: list[Tensor] = []
        self.up: list[Tensor] = []
        self.down: list[Tensor] = []
        self.attn_scale: list[Tensor] = []
        self.mlp_scale: list[Tensor] = []

    def attn_out_placeholder(self, x: Tensor) -> Tensor:
        return x.new_zeros(x.shape)
