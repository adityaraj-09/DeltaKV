# ΔKV — Weight-Delta-Aware KV Cache Maintenance

> *Patch the cache when the weights change* — the third axis of KV reuse.

Today every serving stack treats a weight update as a cache apocalypse. vLLM namespaces prefix caches per LoRA name (and corrupts them on same-name reload — [#42125](https://github.com/vllm-project/vllm/issues/42125), RFC [#48312](https://github.com/vllm-project/vllm/issues/48312)). SGLang isolates by `weight_version` ([PR #27886](https://github.com/sgl-project/sglang/pull/27886)) and flushes HiCache on every update. veRL + Mooncake hard-reset every KV byte on every policy step. The universal algorithm is: **weights changed → throw away all computed state → recompute from scratch.**

**ΔKV replaces invalidation with maintenance.** Given a weight delta ΔW (LoRA swap, RL step, ROME edit, quantization change), it computes the corresponding change to cached KV — ΔKV — at a small fraction of re-prefill cost, with a tracked error budget and selective recompute as the correctness floor.

In database terms: *KV is a materialized view over (weights × tokens); ΔKV is incremental view maintenance under weight updates.*

## What this repo is

A **product-shaped library**, not a paper sketch:

| Piece | Role |
|---|---|
| Core math | Exact layer-1 / projection patches, low-rank compressed ΔKV, first-order analytic propagation, AgentKVShift-style probe correction, CacheBlend residual gate |
| Cache contract | Entries are `(base_kv, patch_chain, error_estimate)`. Compatibility lattice: `strict → patched-ε → miss` |
| Toy reference engine | Real Llama-style stack (RMSNorm, RoPE, GQA, SwiGLU) that prefills, patches, decodes, and scores KL against a fresh prefill — on CPU, in tests |
| Connectors | **vLLM** `KVConnectorBase_V1` plugin, **SGLang** `update_weights_*` hook, **LMCache HiddenStateStore** adapter for Route A, **HuggingFace/PEFT** LoRA extractor |

It does **not** fork vLLM or SGLang. It plugs into the extension points those engines already have.

## Install

```bash
pip install -e ".[dev]"
python -m deltakv bench
pytest
```

## Smallest validator (the kill criterion)

```python
from deltakv import DeltaKVEngine, ToyTransformer, ToyConfig, WeightVersion
from deltakv.deltas import lora_delta, lora_layer, random_lora_factors
import torch

model = ToyTransformer(ToyConfig())
engine = DeltaKVEngine(model)
tokens = torch.randint(0, 256, (128,))
engine.prefill(tokens)                     # cache under W0

# rank-r LoRA on k/v of every layer — the production LoRA-swap case
...
engine.commit_delta(delta)                 # does NOT flush
result = engine.evaluate_against_fresh(tokens, route="probe")
print(result.report.as_dict())             # rel L2, cosine, next-token KL
```

If patched-KV next-token KL stays under ~10⁻² nats, the idea is alive. If error explodes by mid-depth even for tiny ΔW, the budget forces a recompute — you cannot do worse than today.

On the CPU toy (4 layers, d=64, seq=64, rank-4 LoRA, scale=0.01) the validator already clears that bar:

```
route=zeroth    patched  maxRelL2=0.0035  KL≈0     flop_ratio=0.034
route=analytic  patched  maxRelL2≈0       KL≈0     flop_ratio ~ analytic
route=probe     patched  maxRelL2=0.0024  KL≈0     flop_ratio ~ 0.10×prefill + add
```

```bash
python -m deltakv bench --seq 256 --rank 8 --scale 0.01
python -m deltakv demo          # rank-1 ROME edit
```

## How it reuses the existing stack

The field already proved the adjacent facts and filed them under other names. ΔKV is the missing *weight-axis* cell:

| Existing tech | What we reuse | What we add |
|---|---|---|
| vLLM `KVConnectorBase_V1` + `kv_connector_module_path` | Load this repo as an external connector; scatter patched KV into paged blocks | Stop hashing LoRA name / weight version into the block key |
| SGLang radix `extra_key`, `update_weights_*`, `flush_cache` | Keep prefix matching; hook the weight-update RPC | Patch instead of `--enable-weight-version-kv-isolation` |
| LMCache `HiddenStateStore` | fp8/fp16 boundary activations for Route A | Analytic ΔX → ΔKV propagator |
| PEFT LoRA (`ΔW = (α/r) BA`) | Adapter files *are* the delta descriptor | Exact `ΔK = RoPE(X ΔW_K)` at projection layers |
| AgentKVShift / CacheBlend | Probe offset + residual-gated recompute | New trigger (weight delta, not context staleness) and low-rank ΔW structure |
| RoPE | Linearity: `RoPE(K+ΔK) = RoPE(K)+RoPE(ΔK)` | Exactness of the layer-1 key patch |

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for the exact vLLM / SGLang / LMCache wiring, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the math.

## Routes

On a cache hit under a new weight version the engine materializes a *patched view*:

1. **Exact / zeroth-order** — run the adapter path on cached activations. Exact at any layer whose input X is stored (layer 1 always, if embeddings are frozen). `O(n·d·r)` vs `O(n·d²)` re-prefill. For rank-r LoRA the patch itself compresses ~`d/r`× (64× at d=4096, r=64).
2. **Analytic (Route A)** — first-order Jacobians of attention (softmax) and SwiGLU, using LMCache-style hidden snapshots. Cheap when ΔK/ΔQ/ΔV are low-rank.
3. **Probe (Route B)** — recompute 5–15% of tokens under the new weights, estimate the chunk-level KV offset, shift the rest (AgentKVShift). High-residual tokens fall back to full recompute (CacheBlend).

A cost-based scheduler picks the cheapest route that fits ε. Exceeding ε is a **miss**: flush-and-recompute, which is exactly today's behavior.

## Status

This is the full reference implementation: the math, the cache contract, the engine, the connectors, and tests that prove layer-1 exactness to `1e-5`. Production rollout is *wiring* `DeltaKVConnector` into a vLLM/SGLang deployment — the hook surface is implemented, the kernels run in PyTorch (the paged scatter is the vLLM write path). GPU fused kernels for the adapter-path patch are the next mechanical step, not a research blocker.
