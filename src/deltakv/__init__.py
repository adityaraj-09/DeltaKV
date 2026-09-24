"""ΔKV: weight-delta-aware KV cache maintenance."""

from deltakv.config import DeltaKVConfig
from deltakv.deltas import (
    LayerDelta,
    LowRankFactors,
    WeightDelta,
    adapter_swap,
    compose_deltas,
    from_peft_state_dict,
    lora_delta,
    lora_layer,
    quant_delta,
    random_lora_factors,
    rl_step_delta,
    rome_delta,
)
from deltakv.engine import DeltaKVEngine, MaintainResult
from deltakv.metrics import (
    CompareReport,
    attach_quality,
    compare_kv,
    kivi_noise_floor,
    logprob_error,
    next_token_kl,
)
from deltakv.hf_model import copy_llama_weights, hf_available, load_hf_decoder, toy_config_from_hf
from deltakv.model import ToyConfig, ToyTransformer
from deltakv.types import (
    CompatibilityLevel,
    DeltaKind,
    ErrorBudget,
    ErrorEstimate,
    LayerStrategy,
    LookupDecision,
    PatchRoute,
    WeightVersion,
)

__version__ = "0.1.0"

__all__ = [
    "CompareReport",
    "CompatibilityLevel",
    "DeltaKVConfig",
    "DeltaKVEngine",
    "DeltaKind",
    "ErrorBudget",
    "ErrorEstimate",
    "LayerDelta",
    "LayerStrategy",
    "LookupDecision",
    "LowRankFactors",
    "MaintainResult",
    "PatchRoute",
    "ToyConfig",
    "ToyTransformer",
    "WeightDelta",
    "WeightVersion",
    "adapter_swap",
    "copy_llama_weights",
    "hf_available",
    "load_hf_decoder",
    "toy_config_from_hf",
    "attach_quality",
    "compare_kv",
    "kivi_noise_floor",
    "logprob_error",
    "compose_deltas",
    "from_peft_state_dict",
    "lora_delta",
    "lora_layer",
    "next_token_kl",
    "quant_delta",
    "random_lora_factors",
    "rl_step_delta",
    "rome_delta",
]
