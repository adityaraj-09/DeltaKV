"""Compatibility lattice: strict → patched-ε → miss."""

from __future__ import annotations

from deltakv.cache.graph import VersionGraph
from deltakv.cache.store import KVStore
from deltakv.config import DeltaKVConfig
from deltakv.flops import (
    ModelCostDims,
    analytic_flops,
    exact_patch_flops,
    prefill_flops,
    probe_flops,
)
from deltakv.patches.compose import zeroth_order_error
from deltakv.patches.tensors import CacheEntry
from deltakv.types import (
    CompatibilityLevel,
    LookupDecision,
    PatchRoute,
    WeightVersion,
)
from torch import Tensor


class CompatibilityLattice:
    """The serving-time contract.

    Cache keys are token prefixes, *not* (prefix, adapter, weight_version).
    A hit under a new version is a patched view iff the estimated error is
    inside ε; otherwise we do exactly what every engine does today: recompute.
    """

    def __init__(self, store: KVStore, graph: VersionGraph, config: DeltaKVConfig):
        self.store = store
        self.graph = graph
        self.config = config

    def lookup(
        self,
        token_ids: Tensor,
        current: WeightVersion,
        dims: ModelCostDims,
    ) -> tuple[LookupDecision, CacheEntry | None]:
        entry = self.store.get(token_ids)
        recompute_cost = prefill_flops(dims)
        if entry is None:
            return (
                LookupDecision(
                    level=CompatibilityLevel.MISS,
                    route=PatchRoute.RECOMPUTE,
                    target_version=current,
                    estimated_flops=recompute_cost,
                    recompute_flops=recompute_cost,
                    reason="no prefix entry",
                ),
                None,
            )

        if entry.current_version.id == current.id:
            return (
                LookupDecision(
                    level=CompatibilityLevel.STRICT,
                    route=PatchRoute.EXACT,
                    base_version=entry.current_version,
                    target_version=current,
                    estimated_flops=0.0,
                    recompute_flops=recompute_cost,
                    reason="exact weight version",
                ),
                entry,
            )

        path = self.graph.path(entry.current_version.id, current.id)
        if path is None:
            return (
                LookupDecision(
                    level=CompatibilityLevel.MISS,
                    route=PatchRoute.RECOMPUTE,
                    base_version=entry.current_version,
                    target_version=current,
                    estimated_flops=recompute_cost,
                    recompute_flops=recompute_cost,
                    reason="no delta path; correctness floor",
                ),
                entry,
            )

        composed = self.graph.composed(entry.current_version.id, current.id)
        assert composed is not None
        err = zeroth_order_error(composed, dims.n_layers, self.config.layer_lipschitz)
        chain_err = entry.accumulated_error.relative_kv_l2 + err.relative_kv_l2
        err.relative_kv_l2 = chain_err

        mag = composed.relative_magnitude()
        if mag > self.config.patchable_regime:
            return (
                LookupDecision(
                    level=CompatibilityLevel.MISS,
                    route=PatchRoute.RECOMPUTE,
                    base_version=entry.current_version,
                    target_version=current,
                    estimated_error=err,
                    estimated_flops=recompute_cost,
                    recompute_flops=recompute_cost,
                    reason=f"ΔW proxy {mag:.3f} outside patchable regime; correctness floor",
                ),
                entry,
            )

        route, cost = self._pick_route(composed, dims, entry.hidden is not None)
        return (
            LookupDecision(
                level=CompatibilityLevel.PATCHED,
                route=route,
                base_version=entry.current_version,
                target_version=current,
                estimated_error=err,
                estimated_flops=cost,
                recompute_flops=recompute_cost,
                reason=f"patch via {route.value} ({len(path)} delta(s))",
                extra={"n_deltas": len(path)},
            ),
            entry,
        )

    def _pick_route(self, delta, dims: ModelCostDims, has_hidden: bool) -> tuple[PatchRoute, float]:
        preferred = self.config.preferred_route
        exact_cost = exact_patch_flops(dims, delta)
        probe_cost = probe_flops(dims, self.config.probe_ratio)
        analytic_cost = analytic_flops(dims, self.config.propagator_rank)

        if preferred != "auto":
            mapping = {
                "exact": (PatchRoute.EXACT, exact_cost),
                "zeroth": (PatchRoute.ZEROTH, exact_cost),
                "analytic": (PatchRoute.ANALYTIC, analytic_cost),
                "probe": (PatchRoute.PROBE, probe_cost),
            }
            return mapping.get(preferred, (PatchRoute.ZEROTH, exact_cost))

        # Auto: exact/zeroth (same FLOPs — adapter path) if we have hidden
        # states or only the first layer moved; else probe if cheaper than
        # analytic; never pick a route more expensive than recompute.
        rec = prefill_flops(dims)
        candidates: list[tuple[PatchRoute, float]] = [
            (PatchRoute.ZEROTH, exact_cost),
        ]
        if has_hidden:
            candidates.append((PatchRoute.ANALYTIC, analytic_cost))
        candidates.append((PatchRoute.PROBE, probe_cost))
        route, cost = min(candidates, key=lambda kv: kv[1])
        if cost >= rec:
            return PatchRoute.RECOMPUTE, rec
        return route, cost
