"""Base graph entity with identity and versioning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _checksum(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class GraphIdentity:
    """Universal node identity — never rely on Neo4j internal IDs."""

    uuid: str
    created_at: str
    last_updated: str
    version: str
    source: str
    owner: str
    confidence: float
    checksum: str
    schema_version: str = "1.0.0"
    pipeline_version: str = "1.0.0"
    model_version: str = "1.0.0"
    embedding_version: str = "1.0.0"
    prediction_version: str = "1.0.0"
    reasoning_version: str = "1.0.0"
    migration_version: str = "1.0.0"


@dataclass(frozen=True)
class GraphEntity:
    """Immutable graph entity with label and properties."""

    label: str
    identity: GraphIdentity
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        label: str,
        *,
        owner: str,
        identity_source: str = "api",
        confidence: float = 1.0,
        version: str = "1.0.0",
        **properties: Any,
    ) -> GraphEntity:
        now = _now()
        identity = GraphIdentity(
            uuid=str(uuid4()),
            created_at=now,
            last_updated=now,
            version=version,
            source=identity_source,
            owner=owner,
            confidence=confidence,
            checksum=_checksum(properties),
        )
        props = dict(properties)
        if "confidence" not in props:
            props["confidence"] = confidence
        return cls(label=label, identity=identity, properties=props)

    def to_neo4j_props(self) -> dict[str, Any]:
        """Flatten identity + properties for MERGE/SET."""
        return {
            "uuid": self.identity.uuid,
            "created_at": self.identity.created_at,
            "last_updated": self.identity.last_updated,
            "version": self.identity.version,
            "source": self.identity.source,
            "owner": self.identity.owner,
            "confidence": self.identity.confidence,
            "checksum": self.identity.checksum,
            "schema_version": self.identity.schema_version,
            "pipeline_version": self.identity.pipeline_version,
            "model_version": self.identity.model_version,
            "embedding_version": self.identity.embedding_version,
            "prediction_version": self.identity.prediction_version,
            "reasoning_version": self.identity.reasoning_version,
            "migration_version": self.identity.migration_version,
            **self.properties,
        }
