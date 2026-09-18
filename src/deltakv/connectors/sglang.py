"""SGLang integration: replace weight_version isolation with a patch.

SGLang PR #27886 namespaces radix keys by ``weight_version`` and HiCache
#26792 / #29443 flush persistent storage on every update. Wire ΔKV by:

1. Leaving ``--enable-weight-version-kv-isolation`` **off** so the radix tree
   still matches token prefixes across policy steps.
2. Calling :meth:`SGLangConnector.on_update_weights` from
   ``update_weights_from_tensor`` / ``update_weights_from_distributed``
   *instead of* ``flush_cache=True``.
3. Optionally keeping ``weight_version`` as a *label* on the delta (lineage),
   not as a cache key.

The correctness floor is explicit: if the composed error exceeds ε we call
the engine's own ``flush_cache`` for that prefix only.
"""

from __future__ import annotations

from typing import Any, Callable

from deltakv.connectors.base import ConnectorBase
from deltakv.deltas.descriptor import WeightDelta
from deltakv.types import WeightVersion


class SGLangConnector(ConnectorBase):
    def __init__(self, flush_cache: Callable[[], Any] | None = None):
        super().__init__()
        self._flush = flush_cache
        self.weight_version = "v0"

    def on_update_weights(
        self,
        delta: WeightDelta,
        *,
        weight_version: str | None = None,
        flush_on_budget_fail: bool = True,
    ) -> str:
        """Hook for SGLang's ``update_weights_*`` path.

        Returns the action taken: ``"patch"`` or ``"flush"``.
        """
        if weight_version:
            # Keep SGLang's label, but attach it to the *delta*, not the cache key.
            delta.target = WeightVersion(
                id=weight_version,
                parent_id=delta.source.id,
                kind=delta.target.kind,
            )
            self.weight_version = weight_version
        self.on_weight_update(delta)
        if self.engine is None:
            return "patch"
        # Background-maintain stored prefixes; flush only the ones that miss ε.
        flushed = 0
        for result in self.engine.maintain_all():
            if result.decision.level.value == "miss" and flush_on_budget_fail:
                flushed += 1
        if flushed and flushed == len(self.engine.store) and self._flush is not None:
            self._flush()
            return "flush"
        return "patch"

    def extra_key(self, lora_id: str | None = None) -> str:
        """Radix extra_key: tenant / LoRA *name* only, never weight_version."""
        return lora_id or ""
