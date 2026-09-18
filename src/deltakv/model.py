"""Tiny decoder-only transformer used as the product reference implementation.

It is a real Llama-style block stack (RMSNorm, RoPE, GQA, SwiGLU) small enough
to run on CPU in tests, and the ΔKV engine uses it end-to-end: prefill, inject
a LoRA/ROME/quant/RL delta, patch KV, generate, compare against a fresh prefill.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from deltakv.deltas.descriptor import WeightDelta
from deltakv.deltas.factors import LowRankFactors
from deltakv.patches.analytic import BlockWeights, mlp_first_order, rmsnorm_first_order, silu
from deltakv.patches.tensors import HiddenSnapshot, KVCache
from deltakv.rope import RotaryEmbedding, merge_heads, reshape_for_heads
from deltakv.types import WeightVersion


@dataclass
class ToyConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 128
    max_seq: int = 2048
    rms_eps: float = 1e-6
    rope_base: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must divide by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.weight


class ToyBlock(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        d, hd = cfg.d_model, cfg.head_dim
        self.cfg = cfg
        self.attn_norm = RMSNorm(d, cfg.rms_eps)
        self.q_proj = nn.Linear(d, cfg.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * hd, d, bias=False)
        self.mlp_norm = RMSNorm(d, cfg.rms_eps)
        self.gate_proj = nn.Linear(d, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(d, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, d, bias=False)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        if self.cfg.n_kv_heads == self.cfg.n_heads:
            return x
        rep = self.cfg.n_heads // self.cfg.n_kv_heads
        return x.repeat_interleave(rep, dim=1)

    def project_kv(self, x_norm: Tensor, rope: RotaryEmbedding) -> tuple[Tensor, Tensor, Tensor]:
        hd, nh, nkv = self.cfg.head_dim, self.cfg.n_heads, self.cfg.n_kv_heads
        q = rope(reshape_for_heads(self.q_proj(x_norm), nh, hd))
        k = rope(reshape_for_heads(self.k_proj(x_norm), nkv, hd))
        v = reshape_for_heads(self.v_proj(x_norm), nkv, hd)
        return q, k, v

    def attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal_mask: Tensor,
    ) -> Tensor:
        k_f, v_f = self._repeat_kv(k), self._repeat_kv(v)
        scale = self.cfg.head_dim ** -0.5
        scores = torch.einsum("qhd,khd->hqk", q, k_f) * scale + causal_mask
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("hqk,khd->qhd", attn, v_f)
        return self.o_proj(merge_heads(ctx))

    def mlp(self, x: Tensor) -> Tensor:
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        causal_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return ``(h_out, k, v, h_in, h_mid)``."""
        h_in = x
        xn = self.attn_norm(x)
        q, k, v = self.project_kv(xn, rope)
        x = x + self.attention(q, k, v, causal_mask)
        h_mid = x
        x = x + self.mlp(self.mlp_norm(x))
        return x, k, v, h_in, h_mid


class ToyTransformer(nn.Module):
    def __init__(self, cfg: ToyConfig | None = None, version: WeightVersion | None = None):
        super().__init__()
        self.cfg = cfg or ToyConfig()
        c = self.cfg
        self.version = version or WeightVersion(id="base", kind="base")
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([ToyBlock(c) for _ in range(c.n_layers)])
        self.final_norm = RMSNorm(c.d_model, c.rms_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.rope = RotaryEmbedding(c.head_dim, max_seq=c.max_seq, base=c.rope_base)
        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=std)

    def _causal(self, seq: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.triu(torch.full((1, seq, seq), float("-inf"), device=device, dtype=dtype), diagonal=1)

    def prefill(self, token_ids: Tensor) -> tuple[Tensor, KVCache, HiddenSnapshot]:
        """Full prefill. ``token_ids`` is ``[seq]`` int64."""
        if token_ids.ndim != 1:
            token_ids = token_ids.view(-1)
        seq = token_ids.shape[0]
        x = self.embed(token_ids)
        embed = x.detach()
        mask = self._causal(seq, x.device, x.dtype)
        ks, vs, hs, mids = [], [], [], []
        for block in self.blocks:
            x, k, v, h_in, h_mid = block(x, self.rope, mask)
            ks.append(k)
            vs.append(v)
            hs.append(h_in.detach())
            mids.append(h_mid.detach())
        hidden = HiddenSnapshot(
            h=tuple(hs),
            embed=embed,
            dtype=x.dtype,
            h_mid=tuple(mids),
        )
        kv = KVCache(
            k=torch.stack(ks, dim=0),
            v=torch.stack(vs, dim=0),
            version=self.version,
            token_ids=token_ids.detach(),
        )
        logits = self.lm_head(self.final_norm(x))
        return logits, kv, hidden

    def decode_one(self, token_id: Tensor, past: KVCache) -> tuple[Tensor, KVCache]:
        """Append one token to ``past`` (already under ``self.version``)."""
        x = self.embed(token_id.view(-1))
        pos = past.seq_len
        # Rebuild a growing causal mask per layer using cached K/V + new row.
        new_ks, new_vs = [], []
        for i, block in enumerate(self.blocks):
            xn = block.attn_norm(x)
            q = self.rope(
                reshape_for_heads(block.q_proj(xn), self.cfg.n_heads, self.cfg.head_dim),
                positions=torch.tensor([pos], device=x.device),
            )
            k = self.rope(
                reshape_for_heads(block.k_proj(xn), self.cfg.n_kv_heads, self.cfg.head_dim),
                positions=torch.tensor([pos], device=x.device),
            )
            v = reshape_for_heads(block.v_proj(xn), self.cfg.n_kv_heads, self.cfg.head_dim)
            k_cat = torch.cat([past.k[i], k], dim=0)
            v_cat = torch.cat([past.v[i], v], dim=0)
            seq = k_cat.shape[0]
            # Query is a single row; scores [h, 1, seq]
            k_f, v_f = block._repeat_kv(k_cat), block._repeat_kv(v_cat)
            scale = self.cfg.head_dim ** -0.5
            scores = torch.einsum("qhd,khd->hqk", q, k_f) * scale
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.einsum("hqk,khd->qhd", attn, v_f)
            x = x + block.o_proj(merge_heads(ctx))
            x = x + block.mlp(block.mlp_norm(x))
            new_ks.append(k_cat)
            new_vs.append(v_cat)
        logits = self.lm_head(self.final_norm(x))
        new_kv = KVCache(
            k=torch.stack(new_ks, 0),
            v=torch.stack(new_vs, 0),
            version=self.version,
            token_ids=None
            if past.token_ids is None
            else torch.cat([past.token_ids, token_id.view(-1)]),
        )
        return logits.view(-1), new_kv

    def generate(self, token_ids: Tensor, max_new: int, kv: KVCache | None = None) -> Tensor:
        ids = token_ids.view(-1)
        if kv is None:
            logits, kv, _ = self.prefill(ids)
        else:
            logits = self.logits_from_kv(ids, kv).unsqueeze(0)
        pieces = [ids]
        for _ in range(max_new):
            nxt = torch.argmax(logits[-1] if logits.ndim == 2 else logits, dim=-1, keepdim=True)
            logits, kv = self.decode_one(nxt, kv)
            pieces.append(nxt)
            logits = logits.unsqueeze(0)
        return torch.cat(pieces, dim=0)

    def logits_from_kv(self, token_ids: Tensor, kv: KVCache) -> Tensor:
        """Compute next-token logits at the last position using provided prefix KV.

        Runs the last token through the stack, attending to ``kv`` (which must
        already include that last token). Used to score patched vs fresh KV.
        """
        ids = token_ids.view(-1)
        last = ids[-1:]
        x = self.embed(last)
        pos = kv.seq_len - 1
        for i, block in enumerate(self.blocks):
            xn = block.attn_norm(x)
            q = self.rope(
                reshape_for_heads(block.q_proj(xn), self.cfg.n_heads, self.cfg.head_dim),
                positions=torch.tensor([pos], device=x.device),
            )
            k_all, v_all = kv.k[i], kv.v[i]
            k_f, v_f = block._repeat_kv(k_all), block._repeat_kv(v_all)
            scale = self.cfg.head_dim ** -0.5
            scores = torch.einsum("qhd,khd->hqk", q, k_f) * scale
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.einsum("hqk,khd->qhd", attn, v_f)
            x = x + block.o_proj(merge_heads(ctx))
            x = x + block.mlp(block.mlp_norm(x))
        return self.lm_head(self.final_norm(x)).view(-1)

    def apply_delta(self, delta: WeightDelta, version: WeightVersion | None = None) -> None:
        """In-place W ← W + ΔW. Used to produce the fresh-prefill ground truth."""
        for idx, layer in delta.layers.items():
            block: ToyBlock = self.blocks[idx]  # type: ignore[assignment]
            mapping = {
                "q_proj": block.q_proj,
                "k_proj": block.k_proj,
                "v_proj": block.v_proj,
                "o_proj": block.o_proj,
                "gate_proj": block.gate_proj,
                "up_proj": block.up_proj,
                "down_proj": block.down_proj,
            }
            for name, fac in layer.projections.items():
                linear = mapping[name]
                linear.weight.data = linear.weight.data + fac.delta_w().to(
                    dtype=linear.weight.dtype, device=linear.weight.device
                )
        if version is not None:
            self.version = version

    def revert_delta(self, delta: WeightDelta, version: WeightVersion | None = None) -> None:
        for idx, layer in delta.layers.items():
            block: ToyBlock = self.blocks[idx]  # type: ignore[assignment]
            mapping = {
                "q_proj": block.q_proj,
                "k_proj": block.k_proj,
                "v_proj": block.v_proj,
                "o_proj": block.o_proj,
                "gate_proj": block.gate_proj,
                "up_proj": block.up_proj,
                "down_proj": block.down_proj,
            }
            for name, fac in layer.projections.items():
                linear = mapping[name]
                linear.weight.data = linear.weight.data - fac.delta_w().to(
                    dtype=linear.weight.dtype, device=linear.weight.device
                )
        if version is not None:
            self.version = version

    def block_weights(self) -> BlockWeights:
        bw = BlockWeights(self.cfg.n_layers)
        for b in self.blocks:
            bw.q.append(b.q_proj.weight.detach())
            bw.k.append(b.k_proj.weight.detach())
            bw.v.append(b.v_proj.weight.detach())
            bw.o.append(b.o_proj.weight.detach())
            bw.gate.append(b.gate_proj.weight.detach())
            bw.up.append(b.up_proj.weight.detach())
            bw.down.append(b.down_proj.weight.detach())
            bw.attn_scale.append(b.attn_norm.weight.detach())
            bw.mlp_scale.append(b.mlp_norm.weight.detach())
        return bw

    def rmsnorm_tensors(self, hidden: HiddenSnapshot) -> dict[int, Tensor]:
        out = {}
        for i, block in enumerate(self.blocks):
            out[i] = block.attn_norm(hidden.layer_in(i))
        return out

    def recompute_probe_kv(
        self,
        token_ids: Tensor,
        probe_index: Tensor,
        base_kv: KVCache,
    ) -> dict[int, tuple[Tensor, Tensor]]:
        """Fresh K/V for probe tokens under *current* (new) weights.

        Non-probe keys/values stay as ``base_kv`` (already patched zeroth-order
        or stale). Probe tokens attend to the mixed cache — the standard
        CacheBlend/AgentKVShift selective-forward.
        """
        ids = token_ids.view(-1)
        seq = ids.numel()
        x = self.embed(ids)
        mask = self._causal(seq, x.device, x.dtype)
        probe = set(int(i) for i in probe_index.tolist())
        fresh: dict[int, tuple[Tensor, Tensor]] = {}
        k_layers, v_layers = [], []
        for li, block in enumerate(self.blocks):
            xn = block.attn_norm(x)
            q, k_new, v_new = block.project_kv(xn, self.rope)
            k_mix = base_kv.k[li].clone()
            v_mix = base_kv.v[li].clone()
            for t in probe:
                k_mix[t] = k_new[t]
                v_mix[t] = v_new[t]
            x = x + block.attention(q, k_mix, v_mix, mask)
            x = x + block.mlp(block.mlp_norm(x))
            k_layers.append(k_new)
            v_layers.append(v_new)
            fresh[li] = (k_new, v_new)
        return fresh

    def copy_state(self) -> dict[str, Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}
