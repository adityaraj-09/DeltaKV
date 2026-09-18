"""Rank-r subspace, κ anchors, hybrid routing, and serving/RL metrics."""

from __future__ import annotations

import torch

from deltakv.config import DeltaKVConfig
from deltakv.engine import DeltaKVEngine
from deltakv.metrics import (
    attach_quality,
    compare_kv,
    kivi_noise_floor,
    logprob_error,
    logit_l2,
    next_token_kl,
)
from deltakv.patches.anchors import (
    cumulative_sensitivity,
    global_probe_ratio,
    layer_kappas,
    plan_layers,
    probe_ratio_schedule,
    spectral_condition,
)
from deltakv.patches.probe import probe_offset_correct
from deltakv.patches.subspace import flatten_kv, subspace_correct
from deltakv.patches.tensors import KVCache
from deltakv.types import CompatibilityLevel, ErrorBudget, LayerStrategy, PatchRoute, WeightVersion
from tests.conftest import make_lora


def test_subspace_recovers_column_space_plus_orthogonal_mu():
    torch.manual_seed(0)
    seq, h, d, r = 32, 2, 8, 3
    reused = torch.randn(seq, h, d)
    basis = torch.randn(h * d, r)
    codes = torch.randn(seq, r) * 0.05
    mu = torch.randn(h * d)
    # Residual = per-token offset in col(B) plus a constant orthogonal μ.
    gram_mu = torch.linalg.lstsq(basis, mu.unsqueeze(-1)).solution.squeeze(-1)
    mu = mu - basis @ gram_mu
    fresh = reused + (codes @ basis.T).view(seq, h, d) + mu.view(1, h, d)
    probes = torch.arange(0, 16)
    corr, subspace, orth = subspace_correct(reused, fresh, probes, basis=basis, rank=r)
    assert torch.allclose(corr[probes], fresh[probes], atol=1e-5)
    # With true residuals (unit-test fresh), per-token codes recover col(B).
    assert float((corr - fresh).norm() / (fresh.norm() + 1e-8)) < 0.02
    mu_k = (fresh[probes] - reused[probes]).mean(dim=0)
    rank1 = reused + mu_k
    rank1[probes] = fresh[probes]
    assert float((corr - fresh).norm()) < float((rank1 - fresh).norm())


def test_rank1_svd_recovers_constant_shift():
    torch.manual_seed(1)
    reused = torch.randn(24, 2, 4)
    mu = torch.randn(2, 4)
    fresh = reused + mu
    probes = torch.tensor([0, 2, 4, 6, 8, 10])
    corr, _, _ = subspace_correct(reused, fresh, probes, basis=None, rank=1)
    k_c, _, _, _ = probe_offset_correct(reused, reused, fresh, fresh, probes)
    # Both should land near the true offset; subspace must not be worse.
    assert float((corr - fresh).abs().mean()) < 0.05
    assert float((corr - fresh).norm()) <= float((k_c - fresh).norm()) + 1e-5


def test_spectral_condition_and_cumulative_product():
    eye = torch.eye(8)
    assert abs(spectral_condition(eye) - 1.0) < 1e-4
    k = [
        torch.diag(torch.tensor([4.0, 1.0, 1.0, 1.0])),
        torch.diag(torch.tensor([3.0, 1.0, 1.0, 1.0])),
    ]
    v = [torch.eye(4), torch.diag(torch.tensor([4.0, 1.0, 1.0, 1.0]))]
    kappas = layer_kappas(k, v)
    assert abs(kappas[0] - 4.0) < 1e-4
    assert abs(kappas[1] - 12.0) < 1e-4
    cum = cumulative_sensitivity([2.0, 3.0, 4.0])
    assert abs(cum[2] - 4.0) < 1e-9
    assert abs(cum[1] - 12.0) < 1e-9
    assert abs(cum[0] - 24.0) < 1e-9


def test_probe_schedule_front_loads_and_plan_layers(toy):
    cum = [8.0, 4.0, 2.0, 1.0, 1.0, 1.0]
    ratios = probe_ratio_schedule(cum, shallow_ratio=0.20, deep_ratio=0.05)
    assert ratios[0] >= ratios[-1]
    assert max(ratios) <= 0.20 + 1e-9
    assert min(ratios) >= 0.05 - 1e-9

    delta = make_lora(toy, rank=4, scale=0.01, layers=[0, 2], projs=("k_proj", "v_proj"))
    cfg = DeltaKVConfig(hidden_stride=4, kappa_probe_threshold=0.65)
    plans = plan_layers(delta, toy.cfg.n_layers, [10.0, 5.0, 2.0, 1.0], has_hidden=True, config=cfg)
    assert plans[0].strategy is LayerStrategy.EXACT
    # Layer 1 has no local ΔW but sits above a modified layer.
    assert plans[1].strategy is LayerStrategy.SUBSPACE
    assert plans[2].strategy in {LayerStrategy.SUBSPACE, LayerStrategy.PROBE}
    assert global_probe_ratio(plans, 0.10) >= 0.05


def test_hybrid_beats_stale_and_replays(toy):
    tokens = torch.arange(48) % toy.cfg.vocab_size
    engine = DeltaKVEngine(
        toy,
        DeltaKVConfig(error_budget=ErrorBudget(relative_kv_l2=0.5), preferred_route="hybrid"),
    )
    _, kv0 = engine.prefill(tokens)
    engine.commit_delta(make_lora(toy, rank=4, scale=0.01))
    result = engine.evaluate_against_fresh(tokens, route="hybrid")
    assert result.decision.route is PatchRoute.HYBRID
    assert result.report is not None
    chain = engine.store.get(tokens)
    assert chain is not None and chain.chain
    assert "plan=" in chain.chain[-1].error.notes
    _, fresh, _ = toy.prefill(tokens)
    stale = compare_kv(kv0, fresh)
    assert result.report.max_relative_l2 < stale.max_relative_l2
    assert result.report.next_token_kl is not None
    assert result.report.logprob_mae is not None
    assert result.report.kivi4_floor is not None
    assert result.report.kivi4_floor > 0.0
    assert result.report.within_kivi4 is True
    assert result.report.next_token_kl < 1e-2

    first = result.kv
    again = engine.materialize(tokens)
    assert again.decision.level is CompatibilityLevel.STRICT
    assert torch.allclose(first.k, again.kv.k, atol=1e-5)
    assert torch.allclose(first.v, again.kv.v, atol=1e-5)


def test_default_lookup_picks_hybrid(toy):
    engine = DeltaKVEngine(toy, DeltaKVConfig(error_budget=ErrorBudget(relative_kv_l2=0.5)))
    tokens = torch.arange(32) % toy.cfg.vocab_size
    engine.prefill(tokens)
    engine.commit_delta(make_lora(toy, rank=4, scale=0.008, layers=[0, 1]))
    decision, entry = engine.lookup(tokens)
    assert decision.level is CompatibilityLevel.PATCHED
    assert decision.route is PatchRoute.HYBRID
    assert entry is not None
    assert decision.flop_ratio < 1.0


def test_kivi_floor_and_logprob_identity():
    torch.manual_seed(0)
    version = WeightVersion(id="base")
    k = torch.randn(2, 16, 2, 8)
    v = torch.randn(2, 16, 2, 8)
    fresh = KVCache(k=k, v=v, version=version)
    floor = kivi_noise_floor(fresh, bits=4, group_size=8)
    assert floor > 0.0
    identical = compare_kv(fresh, fresh)
    assert identical.max_relative_l2 < 1e-12
    logits = torch.randn(16, 32)
    assert next_token_kl(logits, logits) < 1e-6
    mae, mx = logprob_error(logits, logits)
    assert mae < 1e-6 and mx < 1e-6
    assert logit_l2(logits, logits) < 1e-6
    report = attach_quality(identical, logits, logits, fresh)
    assert report.within_kivi4 is True
    assert report.kivi4_floor == floor

    # Flattened KV helper used by the subspace path.
    t = torch.randn(5, 2, 4)
    assert flatten_kv(t).shape == (5, 8)
