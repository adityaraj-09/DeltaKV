"""First-order analytic helpers: RMSNorm / SwiGLU / softmax Jacobian."""

from __future__ import annotations

import torch

from deltakv.metrics import next_token_kl
from deltakv.patches.analytic import mlp_first_order, rmsnorm_first_order, silu, softmax_jacobian_product


def test_rmsnorm_first_order_matches_finite_diff():
    torch.manual_seed(0)
    x = torch.randn(6, 16)
    dx = torch.randn(6, 16) * 1e-3
    w = torch.ones(16)
    y, dy = rmsnorm_first_order(x, dx, w, 1e-6)
    # Finite difference through the same RMSNorm.
    def rms(z):
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    dy_fd = rms(x + dx) - rms(x)
    rel = (dy - dy_fd).norm() / (dy_fd.norm() + 1e-12)
    assert float(rel) < 0.15


def test_softmax_jacobian_row():
    s = torch.randn(4, 5)
    attn = torch.softmax(s, dim=-1)
    ds = torch.randn_like(s) * 0.01
    d_attn = softmax_jacobian_product(attn, ds)
    fd = torch.softmax(s + ds, dim=-1) - attn
    rel = (d_attn - fd).norm() / (fd.norm() + 1e-12)
    assert float(rel) < 0.2


def test_next_token_kl_identity_is_zero():
    torch.manual_seed(0)
    logits = torch.randn(8, 50)
    assert next_token_kl(logits, logits) < 1e-6
    # Mixed rank: prefill [seq, vocab] vs last-token [vocab]
    assert next_token_kl(logits, logits[-1]) < 1e-6


def test_silu_positive_gate():
    x = torch.linspace(-2, 2, 9)
    y = silu(x)
    assert y[0] < 0 and y[-1] > 0
