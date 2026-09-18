"""Engine-agnostic connector protocol. Optional deps (vLLM, SGLang, HF) are lazy."""

from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor

from deltakv.cache.graph import VersionGraph
from deltakv.cache.store import KVStore
from deltakv.config import DeltaKVConfig
from deltakv.deltas.descriptor import WeightDelta
from deltakv.engine import DeltaKVEngine
from deltakv.types import WeightVersion


class EngineConnector(Protocol):
    """What a serving engine must give ΔKV so it can maintain KV across ΔW."""

    def on_weight_update(self, delta: WeightDelta) -> None: ...

    def lookup_prefix(self, token_ids: Tensor, extra: str = "") -> Any: ...


class ConnectorBase:
    """Shared registry used by every concrete connector."""

    def __init__(self, config: DeltaKVConfig | None = None, store: KVStore | None = None):
        self.config = config or DeltaKVConfig()
        self.store = store or KVStore()
        self.graph = VersionGraph()
        self.current = WeightVersion(id="base", kind="base")
        self.graph.add_version(self.current)
        self.engine: DeltaKVEngine | None = None

    def bind_engine(self, engine: DeltaKVEngine) -> None:
        self.engine = engine
        self.store = engine.store
        self.graph = engine.graph
        self.config = engine.config
        self.current = engine.current

    def on_weight_update(self, delta: WeightDelta) -> None:
        self.graph.add_delta(delta)
        self.current = delta.target
        if self.engine is not None:
            self.engine.commit_delta(delta, apply_weights=False)
