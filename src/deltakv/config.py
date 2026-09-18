"""Runtime configuration for the ΔKV engine and connectors."""

from __future__ import annotations

from dataclasses import dataclass, field

from deltakv.types import ErrorBudget, PatchRoute


@dataclass
class DeltaKVConfig:
    """Product defaults: never worse than flush-and-recompute."""

    error_budget: ErrorBudget = field(default_factory=ErrorBudget)
    # Probe-anchored correction (weight-axis scores + LoRA-subspace shift).
    probe_ratio: float = 0.10
    probe_min_tokens: int = 4
    probe_max_ratio: float = 0.50
    # Progressive probe caps (LoRC: more budget in shallow / high-κ layers).
    probe_shallow_ratio: float = 0.20
    probe_deep_ratio: float = 0.05
    # CacheBlend-style residual gate: tokens above this relative L2 are
    # fully recomputed even after the subspace shift.
    blend_residual_threshold: float = 0.15
    blend_max_recompute_ratio: float = 0.40
    # Rank used to compress dense ΔW (quant / RL) and ΔX during analytic
    # hidden-state propagation.
    propagator_rank: int = 64
    # Store residual-stream snapshots; Route A / exact boundaries read them.
    store_hidden_states: bool = True
    hidden_dtype: str = "float16"
    # Exact-patch + ΔX reset every this many layers (0 = every layer stored,
    # reset every ``hidden_stride`` during analytic compounding).
    hidden_stride: int = 4
    # Layers with normalized cumulative κ at or above this get a PROBE strategy.
    kappa_probe_threshold: float = 0.65
    # Preferred patch route. ``hybrid`` / ``auto`` mix per-layer strategies.
    preferred_route: str = "hybrid"
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
