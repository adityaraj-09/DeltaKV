"""Layer-1 exactness: ΔK = RoPE(X ΔW_K), ΔV = X ΔW_V, RoPE is linear."""

from __future__ import annotations

import torch

from deltakv.config import DeltaKVConfig
from deltakv.engine import DeltaKVEngine
from deltakv.patches.compose import apply_layer_patches
from deltakv.patches.exact import exact_kv_patch
from deltakv.rope import RotaryEmbedding, apply_rope, rotate_half
from deltakv.types import ErrorBudget, WeightVersion
from tests.conftest import make_lora


def test_rope_is_linear():
    rope = RotaryEmbedding(head_dim=16, max_seq=64)
    a = torch.randn(32, 2, 16)
    b = torch.randn(32, 2, 16)
    assert torch.allclose(rope(a + b), rope(a) + rope(b), atol=1e-5)


def test_rotate_half_roundtrip_shape():
    x = torch.randn(4, 8)
    y = rotate_half(rotate_half(x))
    # rotate_half twice is NOT identity (it's a 180° in 2D pairs of halves).
    assert y.shape == x.shape


def test_apply_rope_matches_module():
    rope = RotaryEmbedding(8, max_seq=16)
    x = torch.randn(10, 2, 8)
    cos, sin = rope.tables(10, x.device, x.dtype)
    assert torch.allclose(apply_rope(x, cos, sin), rope(x), atol=1e-6)


def test_layer0_kv_patch_is_exact(toy):
    tokens = torch.arange(48) % toy.cfg.vocab_size
    _, kv0, hidden = toy.prefill(tokens)
    delta = make_lora(toy, rank=4, scale=0.05, layers=[0], projs=("k_proj", "v_proj"))
    xn = toy.blocks[0].attn_norm(hidden.layer_in(0))
    patch = exact_kv_patch(
        xn,
        delta.layers[0],
        n_kv_heads=toy.cfg.n_kv_heads,
        head_dim=toy.cfg.head_dim,
        rope=toy.rope,
    )
    patched = apply_layer_patches(kv0, {0: patch})
    toy.apply_delta(delta, version=WeightVersion(id="lora"))
    _, kv1, _ = toy.prefill(tokens)
    dk = (patched.k[0] - kv1.k[0]).abs().max()
    dv = (patched.v[0] - kv1.v[0]).abs().max()
    assert float(dk) < 2e-5, dk
    assert float(dv) < 2e-5, dv


def test_layer0_engine_exact_route(toy):
    tokens = torch.arange(40) % toy.cfg.vocab_size
    engine = DeltaKVEngine(
        toy, DeltaKVConfig(error_budget=ErrorBudget(relative_kv_l2=0.5), preferred_route="exact")
    )
    engine.prefill(tokens)
    delta = make_lora(toy, rank=4, scale=0.03, layers=[0], projs=("k_proj", "v_proj"))
    engine.commit_delta(delta)
    result = engine.evaluate_against_fresh(tokens, route="zeroth")
    assert result.report is not None
    # Layer 0 must be numerically exact; deeper layers are approximate.
    l0 = result.report.layers[0]
    assert l0.relative_l2_k < 1e-5
    assert l0.relative_l2_v < 1e-5
    assert l0.cosine_k > 0.9999
