"""Compatibility lattice and cost-based route selection."""

from __future__ import annotations

import torch

from deltakv.cache.graph import VersionGraph
from deltakv.cache.lattice import CompatibilityLattice
from deltakv.cache.store import KVStore, prefix_hash
from deltakv.config import DeltaKVConfig
from deltakv.engine import DeltaKVEngine
from deltakv.flops import ModelCostDims, exact_patch_flops, prefill_flops
from deltakv.types import CompatibilityLevel, ErrorBudget, PatchRoute, WeightVersion
from tests.conftest import make_lora


def test_prefix_hash_ignores_weight_version():
    t = torch.arange(16)
    assert prefix_hash(t) == prefix_hash(t.clone())
    assert prefix_hash(t, extra="loraA") != prefix_hash(t, extra="loraB")


def test_strict_hit_after_prefill(toy):
    engine = DeltaKVEngine(toy)
    tokens = torch.arange(24) % toy.cfg.vocab_size
    engine.prefill(tokens)
    decision, entry = engine.lookup(tokens)
    assert decision.level is CompatibilityLevel.STRICT
    assert entry is not None
    assert decision.estimated_flops == 0.0


def test_miss_without_delta_path(toy):
    engine = DeltaKVEngine(toy)
    tokens = torch.arange(24) % toy.cfg.vocab_size
    engine.prefill(tokens)
    engine.current = WeightVersion(id="unknown")
    decision, _ = engine.lookup(tokens)
    assert decision.level is CompatibilityLevel.MISS
    assert decision.route is PatchRoute.RECOMPUTE


def test_patched_when_error_in_budget(toy):
    engine = DeltaKVEngine(
        toy, DeltaKVConfig(error_budget=ErrorBudget(relative_kv_l2=0.5), preferred_route="zeroth")
    )
    tokens = torch.arange(32) % toy.cfg.vocab_size
    engine.prefill(tokens)
    engine.commit_delta(make_lora(toy, rank=4, scale=0.005, layers=[0]))
    decision, entry = engine.lookup(tokens)
    assert decision.level is CompatibilityLevel.PATCHED
    assert entry is not None
    assert decision.flop_ratio < 1.0


def test_correctness_floor_on_huge_delta(toy):
    engine = DeltaKVEngine(
        toy,
        DeltaKVConfig(
            error_budget=ErrorBudget(relative_kv_l2=1e-6),
            layer_lipschitz=4.0,
            correctness_floor=True,
        ),
    )
    tokens = torch.arange(32) % toy.cfg.vocab_size
    engine.prefill(tokens)
    engine.commit_delta(make_lora(toy, rank=8, scale=1.0))
    decision, _ = engine.lookup(tokens)
    assert decision.level is CompatibilityLevel.MISS


def test_patch_flops_orders_of_magnitude_below_prefill(toy):
    delta = make_lora(toy, rank=4, scale=0.01)
    dims = ModelCostDims(toy.cfg.n_layers, toy.cfg.d_model, toy.cfg.n_heads, toy.cfg.n_kv_heads, toy.cfg.d_ff, 2048)
    rec = prefill_flops(dims)
    pat = exact_patch_flops(dims, delta)
    assert pat / rec < 0.15
