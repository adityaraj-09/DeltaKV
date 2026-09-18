"""Prefix-keyed store of ``(base_kv, patch_chain, error)`` rows."""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import torch
from torch import Tensor

from deltakv.patches.tensors import CacheEntry, HiddenSnapshot, KVCache
from deltakv.types import WeightVersion


def prefix_hash(token_ids: Tensor, extra: str = "") -> str:
    """Stable content hash. Intentionally *omits* adapter / weight version.

    That is the whole point: one physical prefix row is shared across versions
    and patched, instead of namespaced per LoRA name (vLLM #27211) or
    weight_version (SGLang #27886).
    """
    ids = token_ids.detach().to(torch.int64).cpu().contiguous().numpy().tobytes()
    h = hashlib.sha256()
    h.update(ids)
    h.update(extra.encode("utf-8"))
    return h.hexdigest()


class KVStore:
    """In-process LRU of cache entries, product-shaped so connectors can wrap it."""

    def __init__(self, max_entries: int = 1024):
        self.max_entries = max_entries
        self._rows: OrderedDict[str, CacheEntry] = OrderedDict()

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, token_ids: Tensor, extra: str = "") -> CacheEntry | None:
        key = prefix_hash(token_ids, extra)
        row = self._rows.get(key)
        if row is None:
            return None
        self._rows.move_to_end(key)
        row.hits += 1
        return row

    def put(
        self,
        token_ids: Tensor,
        kv: KVCache,
        hidden: HiddenSnapshot | None = None,
        extra: str = "",
    ) -> CacheEntry:
        key = prefix_hash(token_ids, extra)
        entry = CacheEntry(key_hash=key, token_ids=token_ids.detach().clone(), base=kv, hidden=hidden)
        if key in self._rows:
            self._rows.move_to_end(key)
        self._rows[key] = entry
        while len(self._rows) > self.max_entries:
            self._rows.popitem(last=False)
        return entry

    def drop(self, token_ids: Tensor, extra: str = "") -> None:
        self._rows.pop(prefix_hash(token_ids, extra), None)

    def items(self) -> list[CacheEntry]:
        return list(self._rows.values())

    def clear(self) -> None:
        self._rows.clear()

    def bytes(self) -> int:
        return sum(e.nbytes() for e in self._rows.values())
