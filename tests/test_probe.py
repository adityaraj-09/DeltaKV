"""Probe-anchored correction and CacheBlend residual gate."""

from __future__ import annotations

import torch

from deltakv.patches.probe import probe_offset_correct, residual_gate, select_probes
from deltakv.patches.exact import lora_activation_scores
from deltakv.deltas.factors import LowRankFactors


def test_select_probes_includes_zero_and_respects_cap():
    scores = torch.arange(20.0)
    idx = select_probes(scores, ratio=0.2, min_tokens=2, max_ratio=0.5)
    assert 0 in idx.tolist()
    assert idx.numel() <= 10


def test_probe_offset_recovers_constant_shift():
    seq, h, d = 32, 2, 8
    reused_k = torch.randn(seq, h, d)
    mu = torch.randn(h, d)
    fresh_k = reused_k + mu
    reused_v = torch.randn(seq, h, d)
    fresh_v = reused_v + mu * 0.5
    probes = torch.tensor([0, 3, 7, 11])
    k_c, v_c, mu_k, mu_v = probe_offset_correct(reused_k, reused_v, fresh_k, fresh_v, probes)
    assert torch.allclose(mu_k, mu, atol=1e-5)
    # Non-probe tokens should be within the weighted shift of the true offset.
    assert (k_c - fresh_k).abs().mean() < 0.05


def test_residual_gate_picks_outliers():
    seq, h, d = 16, 1, 4
    reused = torch.zeros(seq, h, d)
    corrected = torch.zeros(seq, h, d)
    corrected[5] = 10.0
    fresh = torch.zeros(seq, h, d)
    fresh[5] = 10.0
    probes = torch.tensor([0, 1])
    extra = residual_gate(reused, corrected, fresh, probes, threshold=0.5, max_ratio=0.5)
    assert 5 in extra.tolist()


def test_lora_activation_scores_rank():
    fac = LowRankFactors(A=torch.ones(2, 6), B=torch.ones(8, 2), scale=1.0)
    x = torch.zeros(5, 6)
    x[2] = 1.0
    s = lora_activation_scores(x, fac)
    assert int(torch.argmax(s)) == 2
