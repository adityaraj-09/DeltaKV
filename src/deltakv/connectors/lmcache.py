"""LMCache HiddenStateStore adapter — Route A plumbing that already exists.

LMCache's HiddenStateStore (PR #3221) caches per-token hidden states next to
KV chunks, which is exactly the snapshot ΔKV's analytic propagator needs.
This module speaks the same store/retrieve shape so a deployment that already
runs LMCache can feed Route A without a second cache.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from deltakv.connectors.base import ConnectorBase
from deltakv.patches.tensors import HiddenSnapshot


class HiddenStateAdapter:
    """Duck-types LMCache ``engine.hidden_state_store``.

    When a real LMCache engine is passed in, we delegate. Otherwise we keep a
    local dict keyed by ``(token_key, layer_idx)`` so tests and the toy engine
    share one code path.
    """

    def __init__(self, lmcache_store: Any | None = None):
        self._remote = lmcache_store
        self._local: dict[tuple[str, int], Tensor] = {}

    def store_hidden_states(
        self,
        token_ids: Tensor,
        hidden_states: Tensor,
        *,
        layer_idx: int = 0,
        token_offset: int = 0,
    ) -> int:
        if self._remote is not None:
            return int(
                self._remote.store_hidden_states(
                    token_ids, hidden_states, layer_idx=layer_idx, token_offset=token_offset
                )
            )
        key = (_tok_key(token_ids), layer_idx)
        self._local[key] = hidden_states.detach().contiguous()
        return 1

    def retrieve_hidden_states(self, token_ids: Tensor, *, layer_idx: int = 0) -> Tensor | None:
        if self._remote is not None:
            return self._remote.retrieve_hidden_states(token_ids, layer_idx=layer_idx)
        return self._local.get((_tok_key(token_ids), layer_idx))

    def snapshot_from_local(self, token_ids: Tensor, n_layers: int, embed: Tensor) -> HiddenSnapshot | None:
        hs = []
        for l in range(n_layers):
            t = self.retrieve_hidden_states(token_ids, layer_idx=l)
            if t is None:
                return None
            hs.append(t)
        return HiddenSnapshot(h=tuple(hs), embed=embed, dtype=embed.dtype)


def _tok_key(token_ids: Tensor) -> str:
    return ",".join(str(int(t)) for t in token_ids.view(-1).tolist())


class LMCacheConnector(ConnectorBase):
    def __init__(self, lmcache_engine: Any | None = None):
        super().__init__()
        store = None
        if lmcache_engine is not None:
            store = getattr(lmcache_engine, "hidden_state_store", None)
        self.hidden = HiddenStateAdapter(store)
