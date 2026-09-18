"""vLLM KVConnector plugin: shared prefix cache + per-version patch views.

vLLM today hashes prefix blocks with ``lora_name`` (PR #27211) and has live
corruption bugs on same-name reload (#42125, RFC #48312). This connector is
the integration point that *stops namespacing on adapter identity* and instead
asks ΔKV for a patched view.

Load it without forking vLLM::

    KVTransferConfig(
        kv_connector="DeltaKVConnector",
        kv_role="kv_both",
        kv_connector_module_path="deltakv.connectors.vllm",
    )

The class duck-types ``KVConnectorBase_V1`` so unit tests run without vLLM
installed. When vLLM is present we subclass the real base.
"""

from __future__ import annotations

from typing import Any

from deltakv.connectors.base import ConnectorBase
from deltakv.deltas.descriptor import WeightDelta
from deltakv.paged import scatter_into_paged
from deltakv.patches.tensors import KVCache
from deltakv.types import WeightVersion


def _vllm_base():
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore
            KVConnectorBase_V1,
        )

        return KVConnectorBase_V1
    except ImportError:
        return None


_VLLM_BASE = _vllm_base()
_CONNECTOR_BASES: tuple[type, ...] = (
    (_VLLM_BASE, ConnectorBase) if _VLLM_BASE is not None else (ConnectorBase,)
)


class DeltaKVConnector(*_CONNECTOR_BASES):  # type: ignore[misc,valid-type]
    """Scheduler/worker connector. See module docstring for the vLLM wiring."""

    def __init__(self, vllm_config: Any = None, role: Any = None, kv_cache_config: Any = None, **kwargs: Any):
        ConnectorBase.__init__(self)
        self.vllm_config = vllm_config
        self.role = role
        self.kv_cache_config = kv_cache_config
        self._metadata: Any = None
        self._pending_loads: dict[str, int] = {}
        self._adapter_deltas: dict[str, WeightDelta] = {}

    # ----- ΔKV-specific API (call these from LoRA load / RL weight sync) -----
    def register_lora_delta(self, adapter_name: str, delta: WeightDelta) -> None:
        """Called from a ``load_lora_adapter`` monkey-patch or engine hook."""
        self._adapter_deltas[adapter_name] = delta
        self.on_weight_update(delta)

    def register_rl_step(self, delta: WeightDelta) -> None:
        """Called from veRL / sleep-wake weight broadcast instead of flush."""
        self.on_weight_update(delta)

    def patch_paged(
        self,
        kv: KVCache,
        k_caches: list[Any],
        v_caches: list[Any],
        block_table: list[int],
        block_size: int,
    ) -> None:
        """Write a materialized patched view into vLLM's paged buffers."""
        for layer in range(kv.n_layers):
            scatter_into_paged(kv.k[layer], k_caches[layer], block_table, block_size, kv_dim=0)
            scatter_into_paged(kv.v[layer], v_caches[layer], block_table, block_size, kv_dim=0)

    # ----- KVConnectorBase_V1 surface -----
    def bind_connector_metadata(self, connector_metadata: Any) -> None:
        self._metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        self._metadata = None

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        return None

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int | None, bool]:
        """Report extra prefix tokens ΔKV can supply (possibly patched).

        Without the engine bound we cannot inspect GPU blocks; return 0 so
        vLLM falls through to its local cache — still correct, just no
        external hit. When ``self.engine`` is bound we look up the token
        prefix and report a hit if a patched view is in budget.
        """
        if self.engine is None:
            return 0, False
        token_ids = getattr(request, "all_token_ids", None)
        if token_ids is None:
            return 0, False
        import torch

        ids = token_ids if hasattr(token_ids, "numel") else torch.tensor(list(token_ids), dtype=torch.long)
        decision, entry = self.engine.lookup(ids)
        if entry is None or decision.level.value == "miss":
            return 0, False
        extra = max(0, entry.base.seq_len - int(num_computed_tokens))
        return extra, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        rid = getattr(request, "request_id", None)
        if rid is not None and num_external_tokens:
            self._pending_loads[str(rid)] = int(num_external_tokens)

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        return self._metadata

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        return set(), set()

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def take_events(self) -> list[Any]:
        return []


# vLLM factory looks up this name on the module.
__all__ = ["DeltaKVConnector"]
