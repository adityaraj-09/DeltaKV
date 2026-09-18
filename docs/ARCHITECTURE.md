# Architecture

## Cache contract

```
CacheEntry = (base_kv, patch_chain, error_estimate)
lookup(tokens, W_current) →
    STRICT   if entry.version == W_current
    PATCHED  if a ΔW path exists and ε̂ ≤ ε
    MISS     otherwise   # today's flush-and-recompute, the correctness floor
```

Cache keys are **token prefixes only**. Adapter identity and weight version are *edges in a lineage graph*, not part of the key. That is the opposite of vLLM's `lora_name` extra hash and SGLang's `weight_version` radix namespace.

## Delta descriptor

A weight update ships a compact object, not a full checkpoint:

| Regime | Payload | Rank |
|---|---|---|
| LoRA / adapter swap | PEFT `A [r, in]`, `B [out, r]`, scale `α/r` | r (swap: r₁+r₂) |
| ROME / MEMIT | outer product `u vᵀ` | 1 |
| Quant swap | SVD sketch of `W_qnew − W_qold` | chosen r |
| RL step | SVD sketch of a KL-bounded `ΔW` | chosen r |

PEFT convention, used everywhere: `y = x Wᵀ + s (x Aᵀ) Bᵀ`. Every per-token `ΔK` therefore lives in the column space of `B`.

## Exact projection patch

At a layer whose input activations `X` are known (cached hidden, or embeddings for layer 0):

```
ΔK = RoPE(X ΔW_Kᵀ)     # RoPE is linear, so this is exact
ΔV = X ΔW_Vᵀ
```

For LoRA this is literally the adapter path, `O(n d r)`. Storage is `n·r + r·d` codes+basis until RoPE mixes the basis (then we keep a dense increment of size `n·d_kv`, still a single layer, still not a re-prefill).

## Deep layers

`X_ℓ` is not a function of `W` at layer 0 only. Two routes:

**Analytic.** Snapshot `h_in` and `h_mid` (LMCache HiddenStateStore). Propagate `ΔX` with:

- RMSNorm Jacobian (closed form)
- Attention: `ΔS = (ΔQ Kᵀ + Q ΔKᵀ)/√d`, `ΔA = J_softmax(A)[ΔS]`, `Δctx = ΔA V + A ΔV`
- SwiGLU Jacobian for the MLP
- Drop the second-order `ΔX ΔW` term

When `ΔQ, ΔK, ΔV` are rank-r, `ΔS` is `O(n² r)` not `O(n² d)`.

**Probe.** Score tokens by LoRA activation `‖x Aᵀ‖` (weight-axis analogue of CacheBlend's layer-1 KV deviation). Recompute top-b under the new weights against a mixed cache, estimate `μ̂_K, μ̂_V`, shift everyone else, then residual-gate outliers onto full recompute.

## Error

Zeroth-order carries a Lipschitz compound bound (LoRC-style) so we *never under-report*. Probe/analytic can replace it with a measured residual on the probe set. Chains add errors conservatively and rebase (full recompute of that prefix) when `ε̂ ≥ 0.8 ε`.

## Cost model

```
cost_recompute ≈ L · (n d² + n² d + n d d_ff)
cost_patch     ≈ (#lora maps) · n d r
cost_probe     ≈ ρ · cost_recompute + L n d
```

The scheduler will not pick a route whose FLOPs exceed re-prefill. Combined with the ε floor, the engine is *pointwise* never worse than vLLM/SGLang today.

## Paged KV

vLLM stores `[num_blocks, block_size, n_kv_heads, head_dim]` (and 5-D K/V-stacked variants). `deltakv.paged.scatter_into_paged` writes a materialized ΔKV into those blocks so a connector does not have to fight the block allocator.
