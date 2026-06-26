"""Graph mutation audit trail."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger("graph.audit")


@dataclass
class AuditRecord:
    audit_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    operation: str = ""
    request_id: str | None = None
    transaction_id: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    affected_nodes: int = 0
    affected_relationships: int = 0
    execution_time_ms: float = 0.0
    rollback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditService:
    """Append-only audit log for graph mutations."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def log_mutation(
        self,
        operation: str,
        *,
        request_id: str | None = None,
        transaction_id: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        affected_nodes: int = 0,
        affected_relationships: int = 0,
        execution_time_ms: float = 0.0,
        rollback: bool = False,
        **metadata: Any,
    ) -> AuditRecord:
        record = AuditRecord(
            operation=operation,
            request_id=request_id,
            transaction_id=transaction_id,
            entity_type=entity_type,
            entity_id=entity_id,
            affected_nodes=affected_nodes,
            affected_relationships=affected_relationships,
            execution_time_ms=execution_time_ms,
            rollback=rollback,
            metadata=metadata,
        )
        self._records.append(record)
        logger.info("graph_audit", extra=asdict(record))
        return record

    def recent(self, limit: int = 100) -> list[AuditRecord]:
        return self._records[-limit:]
