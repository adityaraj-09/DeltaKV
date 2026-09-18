"""Lineage of weight versions and the deltas that connect them."""

from __future__ import annotations

from deltakv.deltas.descriptor import WeightDelta, compose_deltas
from deltakv.types import WeightVersion


class VersionGraph:
    """Directed lineage: each delta is an edge source → target."""

    def __init__(self) -> None:
        self.versions: dict[str, WeightVersion] = {}
        self._edges: dict[tuple[str, str], WeightDelta] = {}
        self._children: dict[str, list[str]] = {}

    def add_version(self, version: WeightVersion) -> None:
        self.versions[version.id] = version

    def add_delta(self, delta: WeightDelta) -> None:
        self.add_version(delta.source)
        self.add_version(delta.target)
        self._edges[(delta.source.id, delta.target.id)] = delta
        self._children.setdefault(delta.source.id, []).append(delta.target.id)

    def get(self, source_id: str, target_id: str) -> WeightDelta | None:
        return self._edges.get((source_id, target_id))

    def path(self, source_id: str, target_id: str) -> list[WeightDelta] | None:
        """BFS shortest path of deltas. None if unreachable (forces recompute)."""
        if source_id == target_id:
            return []
        if source_id not in self.versions or target_id not in self.versions:
            return None
        parent: dict[str, str | None] = {source_id: None}
        q = [source_id]
        while q:
            cur = q.pop(0)
            if cur == target_id:
                break
            for nxt in self._children.get(cur, []):
                if nxt not in parent:
                    parent[nxt] = cur
                    q.append(nxt)
        if target_id not in parent:
            return None
        nodes = [target_id]
        while parent[nodes[-1]] is not None:
            nodes.append(parent[nodes[-1]])  # type: ignore[arg-type]
        nodes.reverse()
        deltas = []
        for a, b in zip(nodes, nodes[1:]):
            d = self._edges.get((a, b))
            if d is None:
                return None
            deltas.append(d)
        return deltas

    def composed(self, source_id: str, target_id: str) -> WeightDelta | None:
        p = self.path(source_id, target_id)
        if p is None:
            return None
        if not p:
            return None
        acc = p[0]
        for d in p[1:]:
            acc = compose_deltas(acc, d)
        return acc
