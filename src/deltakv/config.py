"""Runtime configuration for the ΔKV engine and connectors."""

from __future__ import annotations

from dataclasses import dataclass, field

from deltakv.types import ErrorBudget, PatchRoute


@dataclass
class DeltaKVConfig:
    """Product defaults: never worse than flush-and-recompute."""

    error_budget: ErrorBudget = field(default_factory=ErrorBudget)
    # Probe-anchored correction (AgentKVShift trigger, weight-axis scores).
    probe_ratio: float = 0.10
    probe_min_tokens: int = 4
    probe_max_ratio: float = 0.50
    # CacheBlend-style residual gate: tokens above this relative L2 are
    # fully recomputed even after the mean-shift.
    blend_residual_threshold: float = 0.15
    blend_max_recompute_ratio: float = 0.40
    # Rank used to compress dense ΔW (quant / RL) and ΔX during analytic
    # hidden-state propagation.
    propagator_rank: int = 64
    store_hidden_states: bool = True
    hidden_dtype: str = "float16"
    # Preferred patch route. ``auto`` picks the cheapest route that fits ε.
    preferred_route: str = "auto"
    # Lipschitz proxy for zeroth-order error compounding (LoRC-style).
    layer_lipschitz: float = 1.2
    # Beyond this ‖ΔW‖ proxy the delta is not "small by construction" and we
    # refuse to patch (full SFT, architecture swap, etc.).
    patchable_regime: float = 0.5
    # Prefix block size, matching vLLM/SGLang page granularity.
    block_size: int = 16
    # When True, a failed budget check always falls back to full prefill.
    correctness_floor: bool = True
    max_patch_chain: int = 8

    def resolve_route(self, requested: str | PatchRoute | None = None) -> str:
        if requested is None:
            return self.preferred_route
        return requested.value if isinstance(requested, PatchRoute) else requested
