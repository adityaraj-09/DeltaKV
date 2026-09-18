"""HuggingFace / PEFT connector: turn adapter publishes into WeightDeltas."""

from __future__ import annotations

from typing import Any

from deltakv.connectors.base import ConnectorBase
from deltakv.deltas.lora import from_peft_state_dict, lora_delta
from deltakv.types import WeightVersion


def peft_available() -> bool:
    try:
        import peft  # noqa: F401

        return True
    except ImportError:
        return False


class HuggingFaceConnector(ConnectorBase):
    """Extract LoRA factors from a PEFT model or a raw state dict.

    Usage::

        conn = HuggingFaceConnector()
        delta = conn.delta_from_peft(peft_model, source, target)
        engine.commit_delta(delta)
    """

    def delta_from_state_dict(
        self,
        state: dict[str, Any],
        source: WeightVersion,
        target: WeightVersion,
        *,
        alpha: float | None = None,
        rank: int | None = None,
    ):
        return from_peft_state_dict(state, source=source, target=target, alpha=alpha, rank=rank)

    def delta_from_peft(
        self,
        peft_model: Any,
        source: WeightVersion,
        target: WeightVersion,
    ):
        """Read ``lora_A`` / ``lora_B`` parameters off an injected PEFT model."""
        cfg = getattr(peft_model, "peft_config", None)
        alpha = rank = None
        if cfg:
            # peft_config may be a dict of adapter_name -> LoraConfig
            first = next(iter(cfg.values())) if isinstance(cfg, dict) else cfg
            alpha = getattr(first, "lora_alpha", None)
            rank = getattr(first, "r", None)
        state = {
            name: p.detach()
            for name, p in peft_model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        }
        if not state:
            # Some PEFT versions store adapters in `.lora_A` modules.
            state = {k: v for k, v in peft_model.state_dict().items() if "lora_" in k}
        return self.delta_from_state_dict(state, source, target, alpha=alpha, rank=rank)
