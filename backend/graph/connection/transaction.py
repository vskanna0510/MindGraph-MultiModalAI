"""Neo4j transaction wrapper with audit metadata."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from neo4j import AsyncSession

T = TypeVar("T")


@dataclass
class GraphTransaction:
    """Metadata for a graph database transaction."""

    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str | None = None
    cypher: str | None = None
    affected_nodes: int = 0
    affected_relationships: int = 0
    execution_time_ms: float = 0.0
    rollback: bool = False
    error: str | None = None


async def run_in_transaction(
    session: AsyncSession,
    work: Callable[[AsyncSession], Awaitable[T]],
    *,
    request_id: str | None = None,
) -> tuple[T, GraphTransaction]:
    """Execute work inside a managed write transaction."""
    meta = GraphTransaction(request_id=request_id)
    start = time.perf_counter()
    try:
        async def _tx(tx_session: AsyncSession) -> T:
            return await work(tx_session)

        result = await session.execute_write(_tx)
        meta.execution_time_ms = (time.perf_counter() - start) * 1000
        return result, meta
    except Exception as exc:
        meta.rollback = True
        meta.error = str(exc)
        meta.execution_time_ms = (time.perf_counter() - start) * 1000
        raise


async def run_read(
    session: AsyncSession,
    cypher: str,
    params: dict[str, Any] | None = None,
    *,
    request_id: str | None = None,
) -> tuple[list[dict[str, Any]], GraphTransaction]:
    """Execute a read query with timing metadata."""
    meta = GraphTransaction(request_id=request_id, cypher=cypher)
    start = time.perf_counter()
    result = await session.run(cypher, params or {})
    records = [dict(r) async for r in result]
    meta.execution_time_ms = (time.perf_counter() - start) * 1000
    meta.affected_nodes = len(records)
    return records, meta
