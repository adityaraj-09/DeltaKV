"""Multi-LoRA shared-prefix: one physical cache, per-adapter patch views."""

from __future__ import annotations

import torch

from deltakv import DeltaKVConfig, DeltaKVEngine, ToyConfig, ToyTransformer, WeightVersion
from deltakv.deltas import adapter_swap, lora_delta, lora_layer, random_lora_factors
from deltakv.types import ErrorBudget


def _adapter(model, name: str, scale: float):
    g = torch.Generator().manual_seed(hash(name) % (2**31))
    c = model.cfg
    layers = {}
    for i in range(c.n_layers):
        layers[i] = lora_layer(
            i,
            k=random_lora_factors(c.d_model, c.n_kv_heads * c.head_dim, 4, scale, generator=g, name="k_proj"),
            v=random_lora_factors(c.d_model, c.n_kv_heads * c.head_dim, 4, scale, generator=g, name="v_proj"),
        )
    return lora_delta(
        source=WeightVersion(id="base"),
        target=WeightVersion(id=name, parent_id="base", kind="lora"),
        layers=layers,
        notes=name,
    )


def main() -> None:
    torch.manual_seed(0)
    model = ToyTransformer(ToyConfig(n_layers=3, d_model=48, n_heads=4, n_kv_heads=4, d_ff=96))
    engine = DeltaKVEngine(model, DeltaKVConfig(error_budget=ErrorBudget(relative_kv_l2=0.4)))
    prompt = torch.arange(64) % model.cfg.vocab_size
    engine.prefill(prompt)

    a = _adapter(model, "tenant-A", 0.008)
    b = _adapter(model, "tenant-B", 0.008)
    engine.commit_delta(a)
    ra = engine.evaluate_against_fresh(prompt, route="zeroth")
    print("adapter A", ra.decision.level.value, ra.report.as_dict() if ra.report else None)

    # Swap A → B without flushing the shared prefix.
    engine.model.revert_delta(a, version=WeightVersion(id="base"))
    engine.current = WeightVersion(id="tenant-A")
    swap = adapter_swap(a, b)
    engine.graph.add_delta(swap)
    engine.commit_delta(b)  # apply B's weights; lineage A→B is also registered
    rb = engine.evaluate_against_fresh(prompt, route="probe")
    print("adapter B", rb.decision.level.value, rb.report.as_dict() if rb.report else None)
    print("physical rows in store:", len(engine.store))


if __name__ == "__main__":
    main()
