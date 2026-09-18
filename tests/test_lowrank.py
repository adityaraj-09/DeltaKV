"""Low-rank patch structure: ΔK_t lives in col(B); storage is n·r + r·d."""

from __future__ import annotations

import torch

from deltakv.deltas.factors import LowRankFactors, concat_factors, difference, from_dense
from deltakv.flops import patch_storage_values
from tests.conftest import make_lora


def test_apply_matches_dense_matmul():
    A = torch.randn(8, 32)
    B = torch.randn(64, 8)
    fac = LowRankFactors(A=A, B=B, scale=0.5)
    x = torch.randn(16, 32)
    y = fac.apply(x)
    dense = x @ fac.delta_w().t()
    assert torch.allclose(y, dense, atol=1e-5)


def test_codes_materialize_identity():
    fac = LowRankFactors(A=torch.randn(4, 20), B=torch.randn(12, 4), scale=0.25)
    x = torch.randn(7, 20)
    assert torch.allclose(fac.codes(x) @ fac.B.t(), fac.apply(x), atol=1e-5)


def test_difference_is_new_minus_old():
    a = LowRankFactors(A=torch.randn(2, 10), B=torch.randn(6, 2), scale=0.1)
    b = LowRankFactors(A=torch.randn(3, 10), B=torch.randn(6, 3), scale=0.2)
    d = difference(b, a)
    assert torch.allclose(d.delta_w(), b.delta_w() - a.delta_w(), atol=1e-5)


def test_concat_adds():
    a = LowRankFactors(A=torch.randn(2, 10), B=torch.randn(6, 2), scale=0.1)
    b = LowRankFactors(A=torch.randn(2, 10), B=torch.randn(6, 2), scale=0.2)
    c = concat_factors(a, b)
    assert torch.allclose(c.delta_w(), a.delta_w() + b.delta_w(), atol=1e-5)
    assert c.rank == 4


def test_svd_roundtrip_full_rank():
    dw = torch.randn(10, 7)
    fac = from_dense(dw, rank=7)
    assert torch.allclose(fac.delta_w(), dw, atol=1e-4)


def test_compression_claim():
    n, d, r = 100_000, 4096, 64
    comp, full, ratio = patch_storage_values(n, d, r)
    assert comp == n * r + r * d
    assert abs(ratio - full / comp) < 1e-9
    assert ratio > 60  # ~64× as in the design note


def test_delta_payload_much_smaller_than_kv(toy):
    delta = make_lora(toy, rank=4, scale=0.01)
    n = 512
    d_kv = toy.cfg.n_kv_heads * toy.cfg.head_dim
    full_kv = toy.cfg.n_layers * n * d_kv * 2
    assert delta.payload_values() < full_kv / 8
