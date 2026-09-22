"""Weight-aware cache maintenance on a Llama-style HF decoder (no download)."""

from __future__ import annotations

import pytest
import torch

from deltakv.config import DeltaKVConfig
from deltakv.engine import DeltaKVEngine
from deltakv.hf_model import copy_llama_weights, hf_available, toy_config_from_hf
from deltakv.model import ToyTransformer
from deltakv.types import CompatibilityLevel, ErrorBudget, WeightVersion
from tests.conftest import make_lora


pytestmark = pytest.mark.skipif(not hf_available(), reason="transformers not installed")


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


def test_copy_llama_weights_and_maintain():
    hf = _tiny_llama()
    dest = ToyTransformer(toy_config_from_hf(hf.config))
    copy_llama_weights(dest, hf)
    assert dest.cfg.n_layers == 2
    assert dest.cfg.n_kv_heads == 2
    tokens = torch.arange(16) % dest.cfg.vocab_size
    engine = DeltaKVEngine(
        dest, DeltaKVConfig(error_budget=ErrorBudget(relative_kv_l2=0.5))
    )
    engine.prefill(tokens)
    engine.commit_delta(make_lora(dest, rank=2, scale=0.01))
    decision, entry = engine.lookup(tokens)
    assert entry is not None
    assert decision.level is CompatibilityLevel.PATCHED
    assert entry.current_version.id == "base"
    maintained = engine.maintain_all(route="hybrid")
    assert len(maintained) == 1
    assert maintained[0].decision.level in {
        CompatibilityLevel.PATCHED,
        CompatibilityLevel.MISS,
    }
    scored = engine.evaluate_against_fresh(tokens, route="hybrid")
    assert scored.report is not None
    assert scored.report.next_token_kl is not None
    assert scored.report.next_token_kl < 1e-2
    assert scored.report.within_kivi4 is True


def test_cli_maintain_toy():
    from deltakv.cli import main

    assert main(["maintain", "--layers", "2", "--d-model", "32", "--seq", "16", "--rank", "2"]) == 0
