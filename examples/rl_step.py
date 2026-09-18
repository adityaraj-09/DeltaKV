"""RL policy step: patch the shared rollout prompt instead of hard-resetting."""

from __future__ import annotations

import torch

from deltakv import DeltaKVConfig, DeltaKVEngine, ToyConfig, ToyTransformer, WeightVersion
from deltakv.deltas import rl_step_delta
from deltakv.types import ErrorBudget


def main() -> None:
    torch.manual_seed(0)
    model = ToyTransformer(ToyConfig(n_layers=3, d_model=48, n_heads=4, n_kv_heads=4, d_ff=96))
    engine = DeltaKVEngine(model, DeltaKVConfig(error_budget=ErrorBudget(relative_kv_l2=0.4)))
    prompt = torch.arange(96) % model.cfg.vocab_size
    engine.prefill(prompt)

    # Simulated small-LR step: 1% of each k/v matrix, SVD-sketched to rank 4.
    dense = {}
    for i, block in enumerate(model.blocks):
        dense[(i, "k_proj")] = block.k_proj.weight.detach() * 0.01
        dense[(i, "v_proj")] = block.v_proj.weight.detach() * 0.01
    delta = rl_step_delta(
        source=WeightVersion(id="base"),
        target=WeightVersion(id="step-1", parent_id="base", kind="rl_step"),
        dense_deltas=dense,
        rank=4,
        relative_scale=0.01,
    )
    engine.commit_delta(delta)
    result = engine.evaluate_against_fresh(prompt, route="probe")
    print("RL step-1", result.decision.reason)
    print(result.report.as_dict() if result.report else None)
    print("flop_ratio vs re-prefill:", round(result.decision.flop_ratio, 4))


if __name__ == "__main__":
    main()
