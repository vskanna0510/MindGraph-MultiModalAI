"""Graph relationship types and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from graph.schema.relationships.definitions import (
    ALLOWED_EDGES,
    RELATIONSHIP_PROPERTY_SCHEMA,
    RELATIONSHIP_TYPES,
)

__all__ = [
    "RELATIONSHIP_TYPES",
    "ALLOWED_EDGES",
    "RELATIONSHIP_PROPERTY_SCHEMA",
    "GraphRelationship",
    "edge_properties",
]


def edge_properties(
    *,
    confidence: float = 1.0,
    weight: float = 1.0,
    reason: str = "",
    model_version: str = "1.0.0",
    evidence_source: str = "inference",
    **extra: Any,
) -> dict[str, Any]:
    """Standard relationship properties per MP4 Part 2 spec."""
    now = datetime.now(UTC).isoformat()
    return {
        "uuid": str(uuid4()),
        "created_at": now,
        "updated_at": now,
        "confidence": confidence,
        "weight": weight,
        "reason": reason,
        "model_version": model_version,
        "evidence_source": evidence_source,
        **extra,
    }


@dataclass(frozen=True)
class GraphRelationship:
    """Immutable relationship between two nodes."""

    rel_type: str
    source_key: str
    source_value: str
    target_key: str
    target_value: str
    properties: dict[str, Any] = field(default_factory=dict)
    edge_id: str = field(default_factory=lambda: str(uuid4()))

    def with_defaults(self) -> GraphRelationship:
        props = edge_properties(**self.properties)
        return GraphRelationship(
            self.rel_type,
            self.source_key,
            self.source_value,
            self.target_key,
            self.target_value,
            props,
            self.edge_id,
        )

    def to_cypher_merge(self, source_label: str, target_label: str) -> tuple[str, dict[str, Any]]:
        rel = self.with_defaults()
        cypher = (
            f"MATCH (a:{source_label} {{{rel.source_key}: $src_val}}), "
            f"(b:{target_label} {{{rel.target_key}: $tgt_val}}) "
            f"MERGE (a)-[r:{rel.rel_type}]->(b) SET r += $props RETURN r"
        )
        return cypher, {
            "src_val": rel.source_value,
            "tgt_val": rel.target_value,
            "props": rel.properties,
        }

    def to_cypher_create(self, source_label: str, target_label: str) -> tuple[str, dict[str, Any]]:
        """Append-only edge — always CREATE new relationship."""
        rel = self.with_defaults()
        cypher = (
            f"MATCH (a:{source_label} {{{rel.source_key}: $src_val}}), "
            f"(b:{target_label} {{{rel.target_key}: $tgt_val}}) "
            f"CREATE (a)-[r:{rel.rel_type} $props]->(b) RETURN r"
        )
        return cypher, {
            "src_val": rel.source_value,
            "tgt_val": rel.target_value,
            "props": rel.properties,
        }
