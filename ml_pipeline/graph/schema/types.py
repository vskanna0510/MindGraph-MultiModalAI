"""Graph domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class GraphNode:
    node_id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    version: str = "1.0.0"

    @classmethod
    def create(cls, label: str, key: str, value: str, **props: Any) -> GraphNode:
        return cls(node_id=f"{label}:{value}", label=label, properties={key: value, **props})


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    rel_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    edge_id: str = field(default_factory=lambda: str(uuid4()))

    def with_defaults(self) -> GraphEdge:
        props = {"timestamp": _now(), "weight": 1.0, "confidence": 1.0, "version": "1.0.0", **self.properties}
        return GraphEdge(self.source_id, self.target_id, self.rel_type, props, self.edge_id)


@dataclass
class GraphSnapshot:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    version: str = "1.0.0"
    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)
