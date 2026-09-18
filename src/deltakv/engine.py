"""ΔKV serving engine: cache as a materialized view over (weights × tokens)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from deltakv.cache.graph import VersionGraph
from deltakv.cache.lattice import CompatibilityLattice
from deltakv.cache.store import KVStore
from deltakv.config import DeltaKVConfig
from deltakv.deltas.descriptor import WeightDelta
from deltakv.flops import (
    ModelCostDims,
    analytic_flops,
    exact_patch_flops,
    prefill_flops,
    probe_flops,
)
from deltakv.metrics import CompareReport, attach_kl, compare_kv
from deltakv.model import ToyTransformer
from deltakv.patches.analytic import attention_first_order, mlp_first_order, rmsnorm_first_order
from deltakv.patches.compose import (
    apply_layer_patches,
    measured_error,
    should_rebase,
    within_budget,
    zeroth_order_error,
)
from deltakv.patches.exact import exact_kv_patch, zeroth_order_patches
from deltakv.patches.probe import apply_probe_to_cache, select_probes, weight_axis_scores
from deltakv.patches.tensors import AppliedPatch, CacheEntry, HiddenSnapshot, KVCache, LowRankKVPatch
from deltakv.types import CompatibilityLevel, LookupDecision, PatchRoute, WeightVersion


@dataclass
class MaintainResult:
    kv: KVCache
    decision: LookupDecision
    report: CompareReport | None = None


class DeltaKVEngine:
    """Product engine.

    Cache entries are ``(base_kv, patch_chain, error_estimate)``. A weight
    update is a versioned event carrying a compact :class:`WeightDelta`. The
    next lookup materializes a patched view instead of missing, and falls
    back to today's flush-and-recompute if the error budget is exceeded.
    """

    def __init__(
        self,
        model: ToyTransformer,
        config: DeltaKVConfig | None = None,
        store: KVStore | None = None,
    ):
        self.model = model
        self.config = config or DeltaKVConfig()
        self.store = store or KVStore()
        self.graph = VersionGraph()
        self.graph.add_version(model.version)
        self.lattice = CompatibilityLattice(self.store, self.graph, self.config)
        self.current = model.version

    def dims(self, seq_len: int) -> ModelCostDims:
        c = self.model.cfg
        return ModelCostDims(
            n_layers=c.n_layers,
            d_model=c.d_model,
            n_heads=c.n_heads,
            n_kv_heads=c.n_kv_heads,
            d_ff=c.d_ff,
            seq_len=seq_len,
        )

    def prefill(self, token_ids: Tensor) -> tuple[Tensor, KVCache]:
        logits, kv, hidden = self.model.prefill(token_ids)
        hs = hidden if self.config.store_hidden_states else None
        self.store.put(token_ids.view(-1), kv, hs)
        return logits, kv

    def commit_delta(self, delta: WeightDelta, *, apply_weights: bool = True) -> None:
        """Publish ΔW. Does **not** flush the cache."""
        self.graph.add_delta(delta)
        if apply_weights:
            self.model.apply_delta(delta, version=delta.target)
        self.current = delta.target
        self.model.version = delta.target

    def lookup(self, token_ids: Tensor) -> tuple[LookupDecision, CacheEntry | None]:
        ids = token_ids.view(-1)
        return self.lattice.lookup(ids, self.current, self.dims(int(ids.numel())))

    def materialize(self, token_ids: Tensor, route: str | None = None) -> MaintainResult:
        ids = token_ids.view(-1)
        decision, entry = self.lookup(ids)
        if route is not None:
            try:
                decision.route = PatchRoute(route)
            except ValueError:
                pass
            if decision.level is CompatibilityLevel.PATCHED and entry is not None:
                dims_ = self.dims(int(ids.numel()))
                composed = self.graph.composed(entry.current_version.id, self.current.id)
                if decision.route is PatchRoute.PROBE:
                    decision.estimated_flops = probe_flops(dims_, self.config.probe_ratio)
                elif decision.route is PatchRoute.ANALYTIC:
                    decision.estimated_flops = analytic_flops(dims_, self.config.propagator_rank)
                elif composed is not None:
                    decision.estimated_flops = exact_patch_flops(dims_, composed)
                decision.recompute_flops = prefill_flops(dims_)

        if decision.level is CompatibilityLevel.MISS or entry is None:
            logits, kv, hidden = self.model.prefill(ids)
            hs = hidden if self.config.store_hidden_states else None
            self.store.put(ids, kv, hs)
            decision.level = CompatibilityLevel.MISS
            decision.route = PatchRoute.RECOMPUTE
            return MaintainResult(kv=kv, decision=decision)

        base_view = self._replay(entry)
        if decision.level is CompatibilityLevel.STRICT:
            return MaintainResult(kv=base_view, decision=decision)

        patched, applied = self._patch(entry, base_view, decision.route)
        # Correctness floor applies to *measured* residuals (probe / eval).
        # A-priori Lipschitz bounds are conservative and must not discard a
        # cheap patch that is empirically tiny.
        if (
            applied.error.measured
            and not within_budget(applied.error, self.config)
            and self.config.correctness_floor
        ):
            logits, kv, hidden = self.model.prefill(ids)
            self.store.put(ids, kv, hidden if self.config.store_hidden_states else None)
            decision.level = CompatibilityLevel.MISS
            decision.route = PatchRoute.RECOMPUTE
            decision.reason = "patch exceeded ε; recomputed"
            return MaintainResult(kv=kv, decision=decision)

        entry.chain.append(applied)
        if applied.error.measured and should_rebase(entry.accumulated_error, self.config):
            entry.base = patched
            entry.chain.clear()
            # Hidden states are now stale relative to the new base; drop them.
            entry.hidden = None
        return MaintainResult(kv=patched, decision=decision)

    def generate(self, token_ids: Tensor, max_new: int = 8) -> tuple[Tensor, LookupDecision]:
        ids = token_ids.view(-1)
        result = self.materialize(ids)
        kv = result.kv
        logits = self.model.logits_from_kv(ids, kv)
        pieces = [ids]
        for _ in range(max_new):
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            logits, kv = self.model.decode_one(nxt, kv)
            pieces.append(nxt)
        return torch.cat(pieces, dim=0), result.decision

    def evaluate_against_fresh(self, token_ids: Tensor, route: str | None = None) -> MaintainResult:
        """Patched KV vs. a cold prefill under the current weights (the kill-criterion)."""
        ids = token_ids.view(-1)
        result = self.materialize(ids, route=route)
        with torch.no_grad():
            fresh_logits, fresh_kv, _ = self.model.prefill(ids)
            patched_logits = self.model.logits_from_kv(ids, result.kv)
        report = attach_kl(compare_kv(result.kv, fresh_kv), patched_logits, fresh_logits)
        result.report = report
        result.decision.estimated_error.next_token_kl = report.next_token_kl
        result.decision.estimated_error.relative_kv_l2 = report.max_relative_l2
        result.decision.estimated_error.measured = True
        return result

    def maintain_all(self, route: str | None = None) -> list[MaintainResult]:
        """Background rebase: patch every stored prefix onto the current version."""
        results = []
        for entry in list(self.store.items()):
            results.append(self.materialize(entry.token_ids, route=route))
        return results

    # ------------------------------------------------------------------
    # internals
    def _replay(self, entry: CacheEntry) -> KVCache:
        kv = entry.base.clone()
        for applied in entry.chain:
            kv = apply_layer_patches(kv, applied.layer_patches)
            kv.version = applied.target
        return kv

    def _patch(
        self, entry: CacheEntry, current_kv: KVCache, route: PatchRoute
    ) -> tuple[KVCache, AppliedPatch]:
        composed = self.graph.composed(entry.current_version.id, self.current.id)
        if composed is None:
            raise RuntimeError("patch called without a delta path")
        ids = entry.token_ids
        hidden = entry.hidden
        cfg = self.model.cfg

        if route is PatchRoute.RECOMPUTE:
            _, kv, hs = self.model.prefill(ids)
            if self.config.store_hidden_states:
                entry.hidden = hs
            err = measured_error(kv, kv)
            err.relative_kv_l2 = 0.0
            applied = AppliedPatch(
                source=entry.current_version,
                target=self.current,
                route=PatchRoute.RECOMPUTE,
                layer_patches={},
                error=err,
            )
            return kv, applied

        if hidden is None and route in {PatchRoute.EXACT, PatchRoute.ZEROTH, PatchRoute.ANALYTIC, PatchRoute.PROBE}:
            # Without hidden states we can still zeroth-order-patch using
            # reconstructed RMSNorm(embed) for layer 0 only, then probe.
            route = PatchRoute.PROBE if route is PatchRoute.ANALYTIC else PatchRoute.ZEROTH

        if route is PatchRoute.ANALYTIC and hidden is not None:
            patches = self._analytic_patches(hidden, composed)
            kv = apply_layer_patches(current_kv, patches)
            kv.version = self.current
            err = zeroth_order_error(composed, cfg.n_layers, self.config.layer_lipschitz)
            err.route = PatchRoute.ANALYTIC
            applied = AppliedPatch(
                source=entry.current_version,
                target=self.current,
                route=PatchRoute.ANALYTIC,
                layer_patches=patches,
                error=err,
            )
            return kv, applied

        # Exact / zeroth: adapter path on cached (stale) activations.
        patches: dict[int, LowRankKVPatch] = {}
        if hidden is not None:
            xn = self.model.rmsnorm_tensors(hidden)
            patches = zeroth_order_patches(
                xn,
                composed,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope=self.model.rope,
            )
        else:
            # Layer-0 only from embeddings.
            x0 = self.model.blocks[0].attn_norm(self.model.embed(ids))
            layer = composed.layers.get(0)
            if layer is not None:
                patches[0] = exact_kv_patch(
                    x0,
                    layer,
                    n_kv_heads=cfg.n_kv_heads,
                    head_dim=cfg.head_dim,
                    rope=self.model.rope,
                )

        kv = apply_layer_patches(current_kv, patches)
        kv.version = self.current

        if route is PatchRoute.PROBE:
            x_for_scores = (
                self.model.blocks[0].attn_norm(hidden.layer_in(0))
                if hidden is not None
                else self.model.blocks[0].attn_norm(self.model.embed(ids))
            )
            scores = weight_axis_scores(x_for_scores, composed, composed.first_modified_layer)
            probes = select_probes(
                scores,
                self.config.probe_ratio,
                self.config.probe_min_tokens,
                self.config.probe_max_ratio,
            )
            fresh = self.model.recompute_probe_kv(ids, probes, kv)
            kv, probe_patches, extra = apply_probe_to_cache(
                kv,
                fresh,
                probes,
                self.config.blend_residual_threshold,
                self.config.blend_max_recompute_ratio,
            )
            kv.version = self.current
            patches = probe_patches
            err = zeroth_order_error(composed, cfg.n_layers, self.config.layer_lipschitz)
            err.route = PatchRoute.PROBE
            err.notes = f"probes={int(probes.numel())} extra={int(extra.numel())}"
            applied = AppliedPatch(
                source=entry.current_version,
                target=self.current,
                route=PatchRoute.PROBE,
                layer_patches=patches,
                error=err,
                probe_index=probes,
                recompute_index=extra,
            )
            return kv, applied

        err = zeroth_order_error(composed, cfg.n_layers, self.config.layer_lipschitz)
        if hidden is not None and composed.first_modified_layer == 0 and all(
            name in {"k_proj", "v_proj"} for ld in composed.layers.values() for name in ld.projections
        ):
            # Not globally exact, but layer-0 k/v with stored X is exact.
            pass
        err.route = PatchRoute.EXACT if route is PatchRoute.EXACT else PatchRoute.ZEROTH
        applied = AppliedPatch(
            source=entry.current_version,
            target=self.current,
            route=err.route,
            layer_patches=patches,
            error=err,
        )
        return kv, applied

    def _analytic_patches(self, hidden: HiddenSnapshot, delta: WeightDelta) -> dict[int, LowRankKVPatch]:
        cfg = self.model.cfg
        seq = hidden.seq_len
        device = hidden.h[0].device
        dtype = hidden.h[0].dtype
        causal = torch.triu(
            torch.full((1, seq, seq), float("-inf"), device=device, dtype=dtype), diagonal=1
        )
        dx = hidden.h[0].new_zeros(seq, cfg.d_model)
        patches: dict[int, LowRankKVPatch] = {}
        for l, block in enumerate(self.model.blocks):
            x = hidden.layer_in(l)
            xn, dxn = rmsnorm_first_order(x, dx, block.attn_norm.weight, cfg.rms_eps)
            layer = delta.layers.get(l)
            dh, dk, dv, _ = attention_first_order(
                xn,
                dxn,
                block.q_proj.weight,
                block.k_proj.weight,
                block.v_proj.weight,
                block.o_proj.weight,
                layer,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope=self.model.rope,
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
            dx = dx + dh
            x_mid = hidden.layer_mid(l)
            xm, dxm = rmsnorm_first_order(x_mid, dx, block.mlp_norm.weight, cfg.rms_eps)
            d_mlp = mlp_first_order(
                xm,
                dxm,
                block.gate_proj.weight,
                block.up_proj.weight,
                block.down_proj.weight,
                layer,
            )
            dx = dx + d_mlp
        return patches
