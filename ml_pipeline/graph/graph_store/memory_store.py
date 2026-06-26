"""In-memory graph store for tests and offline builds."""

from __future__ import annotations

from typing import Any

from ml_pipeline.graph.schema.types import GraphEdge, GraphNode, GraphSnapshot


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._snapshots: list[GraphSnapshot] = []

    def upsert_node(self, node: GraphNode) -> GraphNode:
        self._nodes[node.node_id] = node
        return node

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        e = edge.with_defaults()
        self._edges.append(e)
        return e

    def nodes_by_label(self, label: str) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.label == label]

    def edges_from(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self._edges if e.source_id == node_id]

    def edges_to(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self._edges if e.target_id == node_id]

    def snapshot(self, version: str = "1.0.0", metadata: dict[str, Any] | None = None) -> GraphSnapshot:
        snap = GraphSnapshot(
            nodes=list(self._nodes.values()),
            edges=list(self._edges),
            version=version,
            metadata=metadata or {},
        )
        self._snapshots.append(snap)
        return snap

    def restore(self, snapshot: GraphSnapshot) -> None:
        self._nodes = {n.node_id: n for n in snapshot.nodes}
        self._edges = list(snapshot.edges)

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)
