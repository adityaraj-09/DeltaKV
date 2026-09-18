"""Patch composition and lineage: git-rebase for KV."""

from __future__ import annotations

import torch

from deltakv.cache.graph import VersionGraph
from deltakv.deltas.descriptor import compose_deltas
from deltakv.deltas.factors import LowRankFactors
from deltakv.patches.compose import compose_applied
from deltakv.patches.tensors import AppliedPatch, LowRankKVPatch
from deltakv.types import ErrorEstimate, PatchRoute, WeightVersion
from tests.conftest import make_lora


def test_compose_deltas_adds_factors(toy):
    d1 = make_lora(toy, rank=2, scale=0.01, layers=[0])
    d2 = make_lora(toy, rank=2, scale=0.02, layers=[0])
    d2.source = d1.target
    d2.target = WeightVersion(id="lora2", parent_id=d1.target.id, kind="lora")
    both = compose_deltas(d1, d2)
    w1 = d1.layers[0].projections["k_proj"].delta_w()
    w2 = d2.layers[0].projections["k_proj"].delta_w()
    w12 = both.layers[0].projections["k_proj"].delta_w()
    assert torch.allclose(w12, w1 + w2, atol=1e-5)


def test_version_graph_path():
    g = VersionGraph()
    v0 = WeightVersion(id="v0")
    v1 = WeightVersion(id="v1", parent_id="v0")
    v2 = WeightVersion(id="v2", parent_id="v1")
    g.add_version(v0)
    # deltas
    from deltakv.deltas.lora import lora_delta
    from deltakv.deltas.descriptor import LayerDelta

    d01 = lora_delta(source=v0, target=v1, layers={})
    d12 = lora_delta(source=v1, target=v2, layers={})
    g.add_delta(d01)
    g.add_delta(d12)
    p = g.path("v0", "v2")
    assert p is not None and len(p) == 2
    assert g.path("v2", "v0") is None


def test_compose_applied_adds_dense_patches():
    v0, v1, v2 = WeightVersion(id="a"), WeightVersion(id="b"), WeightVersion(id="c")
    dk1 = torch.ones(4, 2, 8)
    p1 = AppliedPatch(
        source=v0,
        target=v1,
        route=PatchRoute.ZEROTH,
        layer_patches={
            0: LowRankKVPatch(0, None, None, None, None, k_dense=dk1, v_dense=dk1)
        },
        error=ErrorEstimate(relative_kv_l2=0.01),
    )
    p2 = AppliedPatch(
        source=v1,
        target=v2,
        route=PatchRoute.PROBE,
        layer_patches={
            0: LowRankKVPatch(0, None, None, None, None, k_dense=dk1 * 2, v_dense=dk1 * 2)
        },
        error=ErrorEstimate(relative_kv_l2=0.02),
    )
    m = compose_applied(p1, p2, n_kv_heads=2, head_dim=8)
    dk, dv = m.layer_patches[0].materialize(2, 8)
    assert torch.allclose(dk, dk1 * 3)
    assert m.error.relative_kv_l2 == 0.03
    assert m.target.id == "c"
