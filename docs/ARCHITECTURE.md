# Architecture

ΔKV treats the KV cache as a **materialized view** over `(weights × tokens)`. A
weight update is a versioned event that ships a compact ΔW descriptor. The next
lookup **patches** the stored prefix instead of namespacing or flushing it, and
falls back to today's flush-and-recompute if the error budget is exceeded.

This document is the math and control plane. Wiring into vLLM / SGLang / LMCache
is in [INTEGRATION.md](INTEGRATION.md).

## Cache contract

```
CacheEntry = (base_kv, patch_chain, error_estimate)
lookup(tokens, W_current) →
    STRICT   if entry.version == W_current
    PATCHED  if a ΔW path exists and ε̂ ≤ ε
    MISS     otherwise   # today's flush-and-recompute, the correctness floor
```

Cache keys are **token prefixes only**. Adapter identity and weight version are
*edges in a lineage graph*, not part of the key. That is the opposite of vLLM's
`lora_name` extra hash and SGLang's `weight_version` radix namespace.

A patched view is stored as `base + Σ patches`. Probe / hybrid routes write the
**full** patched−base increment so a later STRICT replay does not drop the
zeroth-order stage.

## Delta descriptor

A weight update ships a compact object, not a full checkpoint:

| Regime | Payload | Rank |
|---|---|---|
| LoRA / adapter swap | PEFT `A [r, in]`, `B [out, r]`, scale `α/r` | r (swap: r₁+r₂) |
| ROME / MEMIT | outer product `u vᵀ` | 1 |
| Quant swap | SVD sketch of `W_qnew − W_qold` | chosen r |
| RL step | SVD sketch of a KL-bounded `ΔW` | chosen r |

PEFT convention, used everywhere: `y = x Wᵀ + s (x Aᵀ) Bᵀ`. Every per-token
`ΔV` therefore lives in the column space of `B`. After RoPE, `ΔK` does **not**
— rotation mixes `col(B)` per position — so K uses an SVD basis of the probe
residual while V uses `B` directly.

## Exact projection patch

At a layer whose input activations `X` are known (cached hidden, or embeddings
for layer 0):

```
ΔK = RoPE(X ΔW_Kᵀ)     # RoPE is linear, so this is exact
ΔV = X ΔW_Vᵀ
```

For LoRA this is the adapter path, `O(n d r)`. Storage is `n·r + r·d` codes+basis
until RoPE mixes the basis (then we keep a dense increment of size `n·d_kv`,
still a single layer, still not a re-prefill).

Boundary snapshots every `hidden_stride` layers (default 4) re-anchor this
exact patch so deep-layer error does not compound from a single stale `X_0`.

## LoRA-subspace correction

AgentKVShift's rank-1 mean-shift `μ̂` is the special case `rank=1` with no
basis. Production LoRA is rank-r, so the shared offset lives in `col(B)`:

```
residual_probes ≈ C @ Bᵀ + μ_⊥
ΔKV_j            = w_j · (c̄ @ Bᵀ + μ_⊥)
```

- **V**: per-token least-squares onto LoRA `B`, then a residual mean-shift
  orthogonal to `col(B)` (CacheBlend-weighted).
- **K**: RoPE mixes `col(B)`, so the basis is the leading-r right singular
  vectors of the probe residual (same rank as the adapter).
- Probe tokens are overwritten with fresh values. CacheBlend residual gating
  then fully recomputes outliers above `blend_residual_threshold`.

This is the highest-leverage quality upgrade over a scalar `μ`: the correction
matches the column space of ΔW **per token** instead of collapsing it to a
single rank-1 offset.

## Condition-number anchors (LoRC)

Not every layer deserves the same probe budget. Let `κ_ℓ = κ(W_k) · κ(W_v)`
and

```
κ̃_ℓ = ∏_{j=ℓ}^{L-1} κ_j
```

Shallow errors are amplified more. From `κ̃`:

- **Progressive probe ratio**: first third of the stack and high-`κ̃` layers
  get `probe_shallow_ratio` (default 0.20); the rest get down to
  `probe_deep_ratio` (default 0.05).
- **Hybrid plan** (`plan_layers`):

  | Condition | Strategy |
  |---|---|
  | No local ΔW and no upstream ΔW | `skip` |
  | No local ΔW, upstream ΔX exists | `subspace` (probe-correct ΔX drift) |
  | Stored X at layer 0 / every `hidden_stride`, k/v ΔW | `exact` |
  | Normalized `κ̃ ≥ kappa_probe_threshold` (0.65) | `probe` (subspace + residual gate) |
  | Otherwise | `subspace` (rank-r shift, no extra blend) |

The default `preferred_route` is **`hybrid`**.

## Analytic first-order (Route A)

Snapshot `h_in` and `h_mid` (LMCache HiddenStateStore). Propagate `ΔX` with:

- RMSNorm Jacobian (closed form)
- Attention: `ΔS = (ΔQ Kᵀ + Q ΔKᵀ)/√d`, `ΔA = J_softmax(A)[ΔS]`,
  `Δctx = ΔA V + A ΔV`
- SwiGLU Jacobian for the MLP
- Rank-`propagator_rank` truncation of `ΔX` after every layer
- `ΔX` **reset to 0** at every `hidden_stride` boundary, then exact-patch from
  the stored snapshot (stops first-order compounding)
- Second-order `ΔX ΔW`: the engine path runs **after** `commit_delta`, so
  `dx @ W'` already contains `ΔX ΔW`. The standalone propagator has an explicit
  `base_weights=True` switch for the uncommitted-W case.

When `ΔQ, ΔK, ΔV` are rank-r, `ΔS` is `O(n² r)` not `O(n² d)`.

## Error budget and quality metrics

Zeroth-order carries a Lipschitz compound bound (LoRC-style) so we *never
under-report*. Probe/analytic can replace it with a measured residual.
Chains add errors conservatively and rebase when `ε̂ ≥ 0.8 ε`.

The kill-criterion is not KV-L2 alone. `evaluate_against_fresh` reports:

| Metric | Why |
|---|---|
| max / mean relative KV L2, min cosine | geometric cache error |
| next-token KL (nats) | serving: sampled token distribution |
| logprob MAE / max | RL / TIS: `|log π_fresh − log π_patched|` |
| logit relative L2 | unnormalized head drift |
| KIVI-4bit floor | patched L2 should sit **inside** 4-bit grouped KV quant noise |

A patch that beats a 4-bit KV cache is in the noise of a standard serving
optimization. `within_kivi4` is that boolean.

## Cost model

```
cost_recompute ≈ L · (n d² + n² d + n d d_ff)
cost_patch     ≈ (#lora maps) · n d r
cost_probe     ≈ ρ · cost_recompute + L n d
cost_hybrid    ≈ cost_patch + 0.5 · cost_probe
cost_analytic  ≈ L · (n d r + n² r)
```

The scheduler will not pick an `auto` route whose FLOPs exceed re-prefill.
Combined with the ε floor, the engine is *pointwise* never worse than
vLLM/SGLang today.

## Paged KV

vLLM stores `[num_blocks, block_size, n_kv_heads, head_dim]` (and 5-D K/V-stacked
variants). `deltakv.paged.scatter_into_paged` writes a materialized ΔKV into
those blocks so a connector does not have to fight the block allocator.

## Module map

```
deltakv.engine          lookup / materialize / evaluate_against_fresh
deltakv.cache.lattice   strict → patched-ε → miss; hybrid cost model
deltakv.patches.exact   ΔK = RoPE(X ΔW_K), zeroth-order adapter path
deltakv.patches.subspace  rank-r LS onto B + orthogonal μ
deltakv.patches.anchors   κ, κ̃, probe schedule, per-layer plan
deltakv.patches.probe     weight-axis scores, blend gate, subspace apply
deltakv.patches.analytic  RMSNorm / softmax / SwiGLU Jacobians, ΔX stride
deltakv.metrics           KL, logprob L1, logit L2, KIVI-4bit floor
deltakv.connectors.*      vLLM / SGLang / LMCache / PEFT, no forks
```
