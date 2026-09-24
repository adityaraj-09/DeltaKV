# Integration with vLLM, SGLang, LMCache, PEFT

ΔKV is a library. The serving engines stay the serving engines. This document is the wiring.

## vLLM

**Problem we inherit.** Prefix-block hashes include `lora_name` only
([`kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py),
[PR #27211](https://github.com/vllm-project/vllm/pull/27211)). Same-name reload
reuses stale blocks ([#42125](https://github.com/vllm-project/vllm/issues/42125)).
RL weight reload RFC [#48312](https://github.com/vllm-project/vllm/issues/48312)
currently demands *invalidation or a new identity*. ΔKV is the third option.

**Hook.** vLLM V1 loads external connectors from a module path
([PR #18142](https://github.com/vllm-project/vllm/pull/18142)):

```python
from vllm.config import KVTransferConfig

KVTransferConfig(
    kv_connector="DeltaKVConnector",
    kv_role="kv_both",
    kv_connector_module_path="deltakv.connectors.vllm",
)
```

`DeltaKVConnector` duck-types `KVConnectorBase_V1` (and subclasses it when
vLLM is installed). Call these from the LoRA / RL paths that today flush:

```python
connector.register_lora_delta(adapter_name, delta)   # load_lora_adapter
connector.register_rl_step(delta)                    # weight broadcast
connector.patch_paged(kv, k_caches, v_caches, block_table, block_size)
```

**Do not** put adapter name or weight version in the block hash. Keep the
hash as `(tokens, cache_salt, mm_keys)` so one physical prefix is shared.
The connector looks up that prefix, materializes a patched view for the
*current* `WeightVersion`, and scatters it into the already-allocated
blocks. If ε is exceeded, `get_num_new_matched_tokens` returns 0 and vLLM
prefills — the correctness floor.

A small engine-side change still required upstream (or via a plugin hook):
on `load_lora_adapter` / in-place reload, pass the new PEFT factors to
`register_lora_delta` instead of bumping a generation counter. Until that
lands, wrap the HTTP `/v1/load_lora_adapter` handler.

## SGLang

**Problem we inherit.** `--enable-weight-version-kv-isolation`
([PR #27886](https://github.com/sgl-project/sglang/pull/27886)) puts
`weight_version` in the radix `extra_key`. HiCache persistent files were
not even flushed ([#26792](https://github.com/sgl-project/sglang/issues/26792),
fixed by *clearing storage* in [#29443](https://github.com/sgl-project/sglang/pull/29443)).

**Hook.** Leave isolation **off**. Wrap `update_weights_from_tensor` /
`update_weights_from_distributed`:

```python
from deltakv.connectors.sglang import SGLangConnector

conn = SGLangConnector(flush_cache=engine.flush_cache)
conn.bind_engine(deltakv_engine)
action = conn.on_update_weights(delta, weight_version="step-42")
# action == "patch"  or  "flush" (only if every prefix exceeded ε)
```

Keep `weight_version` as a *label on the delta*, not as a radix key.
`conn.extra_key(lora_id)` is tenant/adapter name only.

For RL: `update_weights_*(..., flush_cache=False)` plus the connector.
Live in-flight requests keep their private KV; the radix tree is patched
in the background (`engine.maintain_all()`), which is LOCAL's
stale-coverage prefill with patches instead of recomputes.

## LMCache

Route A needs boundary activations. LMCache already stores them:

- [`HiddenStateStore`](https://docs.lmcache.ai/non_kv_cache/hidden_states.html)
- engine flag `enable_hidden_state_cache`

```python
from deltakv.connectors.lmcache import LMCacheConnector

conn = LMCacheConnector(lmcache_engine)
conn.hidden.store_hidden_states(token_ids, h, layer_idx=ℓ)
snap = conn.hidden.snapshot_from_local(token_ids, n_layers, embed)
```

Same chunk keys as KV, same eviction coupling. ΔKV does not grow a second
store.

## HuggingFace / PEFT

```python
from deltakv.connectors.huggingface import HuggingFaceConnector
from deltakv import WeightVersion

conn = HuggingFaceConnector()
delta = conn.delta_from_peft(peft_model, source=v0, target=v1)
# or conn.delta_from_state_dict(state_dict, ...)
engine.commit_delta(delta)
```

`from_peft_state_dict` accepts the usual
`model.layers.{i}.self_attn.k_proj.lora_A.weight` keys and PEFT scaling
`α/r`. Adapter swap is `adapter_swap(old_delta, new_delta)` (rank adds).

To **run** maintenance on a published Llama-style checkpoint (SmolLM2,
TinyLlama, …) copy the weights into the reference decoder and use the
same engine as the toy validator:

```python
from deltakv import DeltaKVEngine, load_hf_decoder

model = load_hf_decoder("HuggingFaceTB/SmolLM2-135M")
engine = DeltaKVEngine(model)
engine.prefill(token_ids)          # cache under W0
engine.commit_delta(delta)         # PEFT / LoRA WeightDelta; does not flush
engine.maintain_all(route="hybrid")
engine.evaluate_against_fresh(token_ids)
```

`python -m deltakv maintain --model HuggingFaceTB/SmolLM2-135M` is that loop.

## veRL / Mooncake / sleep-wake

Do **not** call `sleep(mode="recompute")` or Mooncake's store flush on a
KL-bounded policy step. Broadcast the SVD-sketched `rl_step_delta` (or
the LoRA factors, if the policy is adapter-tuned) and `register_rl_step`.
Rollout workers keep the shared prompt KV and patch it. If a worker's
error estimate exceeds ε, *that worker* prefills — others do not wait.

## Multi-LoRA shared prefix

Physical cache: one row keyed by the system-prompt tokens.
Per adapter: a `WeightDelta` from base → adapter (or adapter A → B).
Request for adapter B materializes `patch(base_kv, ΔW_B)` (or
`patch(patch(base_kv, ΔW_A), ΔW_{A→B})`). No N namespaced copies
(the LMCache #2961 / vLLM corruption failure mode).

## What still has to land upstream

1. vLLM: a one-line call from `load_lora_adapter` into the connector;
   stop putting `lora_name` in the block hash when the connector is
   installed (keep it as a *delta lookup key*).
2. SGLang: `flush_cache=False` by default when a ΔKV connector is
   registered; HiCache keys stay token-only.
3. Optional fused CUDA kernel for `codes = X @ Aᵀ; Δ = codes @ Bᵀ` in
   paged layout. The PyTorch path is correct and is what the tests run.
