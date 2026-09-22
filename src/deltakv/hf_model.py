"""Load a Llama-style HuggingFace checkpoint into the ΔKV decoder stack.

The serving engine stays on :class:`deltakv.model.ToyTransformer` (RMSNorm,
RoPE, GQA, SwiGLU). This module copies real HF weights — including SmolLM2 —
so weight-aware cache maintenance can run on a published model without
forking transformers or vLLM.
"""

from __future__ import annotations

from typing import Any

import torch

from deltakv.model import ToyConfig, ToyTransformer


def hf_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _rope_base(cfg: Any) -> float:
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    theta = getattr(cfg, "rope_theta", None)
    if theta is not None:
        return float(theta)
    return 10000.0


def _inner(hf_model: Any) -> Any:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model
    if hasattr(hf_model, "layers"):
        return hf_model
    raise TypeError(
        f"{type(hf_model).__name__} is not a Llama-style decoder (need .model.layers)"
    )


def toy_config_from_hf(hf_config: Any) -> ToyConfig:
    n_heads = int(hf_config.num_attention_heads)
    d_model = int(hf_config.hidden_size)
    head_dim = int(getattr(hf_config, "head_dim", 0) or d_model // n_heads)
    if head_dim * n_heads != d_model:
        raise ValueError(
            f"hidden_size {d_model} is not n_heads {n_heads} × head_dim {head_dim}"
        )
    n_kv = int(getattr(hf_config, "num_key_value_heads", n_heads) or n_heads)
    return ToyConfig(
        vocab_size=int(hf_config.vocab_size),
        d_model=d_model,
        n_layers=int(hf_config.num_hidden_layers),
        n_heads=n_heads,
        n_kv_heads=n_kv,
        d_ff=int(hf_config.intermediate_size),
        max_seq=int(getattr(hf_config, "max_position_embeddings", 2048) or 2048),
        rms_eps=float(getattr(hf_config, "rms_norm_eps", 1e-6)),
        rope_base=_rope_base(hf_config),
        tie_embeddings=bool(getattr(hf_config, "tie_word_embeddings", True)),
    )


def copy_llama_weights(dest: ToyTransformer, hf_model: Any) -> ToyTransformer:
    """Copy LlamaForCausalLM (or LlamaModel) parameters into ``dest``."""
    inner = _inner(hf_model)
    if len(inner.layers) != dest.cfg.n_layers:
        raise ValueError(
            f"layer count {len(inner.layers)} != dest {dest.cfg.n_layers}"
        )
    dest.embed.weight.data.copy_(inner.embed_tokens.weight.detach().float())
    dest.final_norm.weight.data.copy_(inner.norm.weight.detach().float())
    if dest.cfg.tie_embeddings:
        dest.lm_head.weight = dest.embed.weight
    else:
        lm = getattr(hf_model, "lm_head", None)
        if lm is None:
            raise TypeError("untied model is missing lm_head")
        dest.lm_head.weight.data.copy_(lm.weight.detach().float())
    for i, layer in enumerate(inner.layers):
        block = dest.blocks[i]
        attn = layer.self_attn
        mlp = layer.mlp
        block.attn_norm.weight.data.copy_(layer.input_layernorm.weight.detach().float())
        block.mlp_norm.weight.data.copy_(
            layer.post_attention_layernorm.weight.detach().float()
        )
        block.q_proj.weight.data.copy_(attn.q_proj.weight.detach().float())
        block.k_proj.weight.data.copy_(attn.k_proj.weight.detach().float())
        block.v_proj.weight.data.copy_(attn.v_proj.weight.detach().float())
        block.o_proj.weight.data.copy_(attn.o_proj.weight.detach().float())
        block.gate_proj.weight.data.copy_(mlp.gate_proj.weight.detach().float())
        block.up_proj.weight.data.copy_(mlp.up_proj.weight.detach().float())
        block.down_proj.weight.data.copy_(mlp.down_proj.weight.detach().float())
    dest.eval()
    return dest


def load_hf_decoder(
    model_id: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ToyTransformer:
    """Download (or reuse cache of) ``model_id`` and return a ΔKV decoder."""
    if not hf_available():
        raise ImportError("transformers is required: pip install 'deltakv[hf]'")
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=str(device),
        low_cpu_mem_usage=True,
    )
    dest = ToyTransformer(toy_config_from_hf(hf.config))
    copy_llama_weights(dest, hf)
    dest.to(device)
    dest.eval()
    del hf
    return dest
