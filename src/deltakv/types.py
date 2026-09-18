"""Shared types: weight versions, compatibility lattice, error budgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeltaKind(str, Enum):
    """How the weights changed. All four are *small-delta by construction*."""

    LORA = "lora"
    ROME = "rome"
    QUANT = "quant"
    RL_STEP = "rl_step"
    DENSE = "dense"


class CompatibilityLevel(str, Enum):
    """Continuous cache consistency instead of binary match/miss."""

    STRICT = "strict"  # KV computed under the exact current weights
    PATCHED = "patched"  # KV maintained under a bounded-error patch
    MISS = "miss"  # must recompute (correctness floor)


class PatchRoute(str, Enum):
    """How a patched view was produced."""

    EXACT = "exact"  # linear projections with cached activations (layer-1 / stored X)
    ZEROTH = "zeroth"  # adapter path on stale hidden states
    ANALYTIC = "analytic"  # first-order Jacobian / low-rank ΔS propagation
    PROBE = "probe"  # AgentKVShift-style probe offset + selective recompute
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class WeightVersion:
    """Identity of a weight snapshot. Forms a lineage DAG via ``parent_id``."""

    id: str
    parent_id: str | None = None
    kind: str = "base"
    metadata: tuple[tuple[str, str], ...] = ()

    def lineage_key(self) -> str:
        return self.id


@dataclass
class ErrorBudget:
    """Hard ceiling on patched-KV staleness. Exceeding it forces recompute."""

    relative_kv_l2: float = 0.10
    next_token_kl: float = 1e-2
    rebase_fraction: float = 0.80

    def allows(self, relative_kv_l2: float, next_token_kl: float | None = None) -> bool:
        if relative_kv_l2 > self.relative_kv_l2:
            return False
        if next_token_kl is not None and next_token_kl > self.next_token_kl:
            return False
        return True

    def needs_rebase(self, relative_kv_l2: float) -> bool:
        return relative_kv_l2 >= self.rebase_fraction * self.relative_kv_l2


@dataclass
class ErrorEstimate:
    """Tracked error for a patch chain. Conservative: never under-report."""

    relative_kv_l2: float = 0.0
    per_layer: tuple[float, ...] = ()
    next_token_kl: float | None = None
    route: PatchRoute = PatchRoute.EXACT
    measured: bool = False
    notes: str = ""

    @property
    def max_layer_error(self) -> float:
        return max(self.per_layer) if self.per_layer else self.relative_kv_l2


@dataclass
class LookupDecision:
    """Cost-based choice among hit / patch / recompute."""

    level: CompatibilityLevel
    route: PatchRoute
    base_version: WeightVersion | None = None
    target_version: WeightVersion | None = None
    estimated_error: ErrorEstimate = field(default_factory=ErrorEstimate)
    estimated_flops: float = 0.0
    recompute_flops: float = 0.0
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def flop_ratio(self) -> float:
        if self.recompute_flops <= 0:
            return 0.0
        return self.estimated_flops / self.recompute_flops
