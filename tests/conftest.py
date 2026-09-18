from __future__ import annotations

import pytest
import torch

from deltakv.deltas.descriptor import lora_layer
from deltakv.deltas.factors import random_lora_factors
from deltakv.deltas.lora import lora_delta
from deltakv.model import ToyConfig, ToyTransformer
from deltakv.types import WeightVersion


@pytest.fixture
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


@pytest.fixture
def toy(rng: torch.Generator) -> ToyTransformer:
    torch.manual_seed(0)
    return ToyTransformer(ToyConfig(n_layers=4, d_model=64, n_heads=4, n_kv_heads=4, d_ff=128, vocab_size=128))


def make_lora(model: ToyTransformer, rank: int = 4, scale: float = 0.01, layers=None, projs=("k_proj", "v_proj")):
    g = torch.Generator().manual_seed(1)
    c = model.cfg
    out = {}
    which = range(c.n_layers) if layers is None else layers
    for i in which:
        kwargs = {}
        if "k_proj" in projs:
            kwargs["k"] = random_lora_factors(
                c.d_model, c.n_kv_heads * c.head_dim, rank, scale=scale, generator=g, name="k_proj"
            )
        if "v_proj" in projs:
            kwargs["v"] = random_lora_factors(
                c.d_model, c.n_kv_heads * c.head_dim, rank, scale=scale, generator=g, name="v_proj"
            )
        if "q_proj" in projs:
            kwargs["q"] = random_lora_factors(
                c.d_model, c.n_heads * c.head_dim, rank, scale=scale, generator=g, name="q_proj"
            )
        if "down_proj" in projs:
            kwargs["down"] = random_lora_factors(c.d_ff, c.d_model, rank, scale=scale, generator=g, name="down_proj")
        out[i] = lora_layer(i, **kwargs)
    return lora_delta(
        source=WeightVersion(id="base"),
        target=WeightVersion(id="lora", parent_id="base", kind="lora"),
        layers=out,
    )
