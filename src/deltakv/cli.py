"""CLI: ``python -m deltakv bench`` runs the smallest validator on CPU."""

from __future__ import annotations

import argparse
import sys

import torch

from deltakv.config import DeltaKVConfig
from deltakv.deltas.descriptor import WeightDelta, lora_layer
from deltakv.deltas.factors import random_lora_factors
from deltakv.deltas.lora import lora_delta
from deltakv.deltas.rome import rome_delta
from deltakv.engine import DeltaKVEngine
from deltakv.flops import ModelCostDims, exact_patch_flops, patch_storage_values, prefill_flops
from deltakv.model import ToyConfig, ToyTransformer
from deltakv.types import ErrorBudget, WeightVersion


def _build_lora(model: ToyTransformer, rank: int, scale: float, rng: torch.Generator) -> WeightDelta:
    layers = {}
    c = model.cfg
    for i, block in enumerate(model.blocks):
        k = random_lora_factors(
            c.d_model, c.n_kv_heads * c.head_dim, rank, scale=scale, generator=rng, name="k_proj"
        )
        v = random_lora_factors(
            c.d_model, c.n_kv_heads * c.head_dim, rank, scale=scale, generator=rng, name="v_proj"
        )
        q = random_lora_factors(
            c.d_model, c.n_heads * c.head_dim, rank, scale=scale, generator=rng, name="q_proj"
        )
        layers[i] = lora_layer(i, k=k, v=v, q=q)
    return lora_delta(
        source=WeightVersion(id="base"),
        target=WeightVersion(id="lora-v1", parent_id="base", kind="lora"),
        layers=layers,
    )


def cmd_bench(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    cfg = ToyConfig(
        n_layers=args.layers,
        d_model=args.d_model,
        n_heads=max(1, args.d_model // 16) if args.d_model >= 16 else 4,
        n_kv_heads=max(1, args.d_model // 16) if args.d_model >= 16 else 4,
        d_ff=args.d_model * 2,
        vocab_size=256,
    )
    # Keep head counts valid.
    while cfg.d_model % cfg.n_heads != 0:
        cfg.n_heads -= 1
    cfg.n_kv_heads = cfg.n_heads
    model = ToyTransformer(cfg)
    engine = DeltaKVEngine(
        model,
        DeltaKVConfig(error_budget=ErrorBudget(relative_kv_l2=args.epsilon, next_token_kl=1e-2)),
    )
    tokens = torch.randint(0, cfg.vocab_size, (args.seq,))
    engine.prefill(tokens)
    delta = _build_lora(model, args.rank, args.scale, rng)
    engine.commit_delta(delta)

    print(f"ΔKV bench  layers={cfg.n_layers} d={cfg.d_model} seq={args.seq} rank={args.rank} scale={args.scale}")
    d_kv = cfg.n_kv_heads * cfg.head_dim
    comp, full, ratio = patch_storage_values(args.seq, d_kv, args.rank)
    print(f"low-rank ΔK storage: {comp} values vs {full} dense  ({ratio:.1f}× compressed, one layer)")

    dims = ModelCostDims(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff, args.seq)
    rec = prefill_flops(dims)
    pat = exact_patch_flops(dims, delta)
    print(f"FLOPs  re-prefill={rec:.0f}  exact/zeroth patch={pat:.0f}  ratio={pat / rec:.4f}")

    for route in ("zeroth", "analytic", "probe", "hybrid"):
        # Fresh engine state per route: restore store by re-prefill under OLD
        # weights. Easier: evaluate_against_fresh uses current (new) weights
        # and the cached old KV still in the store.
        result = engine.evaluate_against_fresh(tokens, route=route)
        r = result.report
        assert r is not None
        kl = f"{r.next_token_kl:.4e}" if r.next_token_kl is not None else "n/a"
        lp = f"{r.logprob_mae:.4e}" if r.logprob_mae is not None else "n/a"
        kivi = f"{r.kivi4_floor:.4f}" if r.kivi4_floor is not None else "n/a"
        print(
            f"  route={route:10s}  decision={result.decision.level.value:8s}  "
            f"maxRelL2={r.max_relative_l2:.4f}  minCos={r.min_cosine:.4f}  "
            f"KL={kl}  logprobMAE={lp}  kivi4={kivi}  "
            f"withinKIVI={r.within_kivi4}  flop_ratio={result.decision.flop_ratio:.4f}"
        )
        # After probe/analytic the chain is dirty; reset by dropping and
        # re-prefilling under *old* weights would require revert. For the
        # bench we revert weights, clear store, re-prefill, re-apply.
        engine.model.revert_delta(delta, version=WeightVersion(id="base"))
        engine.current = WeightVersion(id="base")
        engine.model.version = engine.current
        engine.store.clear()
        engine.prefill(tokens)
        engine.commit_delta(delta)

    print("kill criterion: patched-KV KL ≲ 1e-2 nats, mid-depth rel L2 ≲ 0.10, and maxRelL2 ≲ KIVI-4bit floor")
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    torch.manual_seed(0)
    model = ToyTransformer(ToyConfig(n_layers=3, d_model=48, n_heads=4, n_kv_heads=4, d_ff=96))
    engine = DeltaKVEngine(model)
    tokens = torch.arange(32) % model.cfg.vocab_size
    engine.prefill(tokens)
    # Rank-1 ROME write into layer-1 down_proj.
    u = torch.randn(model.cfg.d_model)
    v = torch.randn(model.cfg.d_ff)
    u = u / u.norm()
    v = v / v.norm() * 0.05
    delta = rome_delta(
        source=WeightVersion(id="base"),
        target=WeightVersion(id="rome-1", parent_id="base", kind="rome"),
        layer_idx=1,
        projection="down_proj",
        u=u,
        v=v,
    )
    engine.commit_delta(delta)
    result = engine.evaluate_against_fresh(tokens, route="probe")
    r = result.report
    assert r is not None
    print("ROME edit demo")
    print(f"  level={result.decision.level.value} route={result.decision.route.value}")
    print(f"  maxRelL2={r.max_relative_l2:.4f} KL={r.next_token_kl:.4e}")
    print(f"  logprobMAE={r.logprob_mae:.4e} kivi4={r.kivi4_floor:.4f} withinKIVI={r.within_kivi4}")
    print(f"  reason={result.decision.reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="deltakv", description="Weight-delta-aware KV cache maintenance")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="smallest validator: LoRA patch vs fresh prefill")
    b.add_argument("--layers", type=int, default=4)
    b.add_argument("--d-model", type=int, default=64)
    b.add_argument("--seq", type=int, default=128)
    b.add_argument("--rank", type=int, default=8)
    b.add_argument("--scale", type=float, default=0.01)
    b.add_argument("--epsilon", type=float, default=0.15)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_bench)

    d = sub.add_parser("demo", help="ROME rank-1 edit patched through the cache")
    d.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
