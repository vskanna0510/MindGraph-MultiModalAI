"""Base graph repository — async Cypher execution."""

from __future__ import annotations

import time
from abc import ABC
from typing import Any

from neo4j import AsyncSession

from graph.connection.transaction import GraphTransaction, run_read


class BaseGraphRepository(ABC):
    """Repositories own Cypher; services never write Cypher."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _run(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], GraphTransaction]:
        return await run_read(self._session, cypher, params, request_id=request_id)

    async def _write(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], GraphTransaction]:
        meta = GraphTransaction(request_id=request_id, cypher=cypher)
        start = time.perf_counter()

        async def _tx(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(cypher, params or {})
            return [dict(r) async for r in result]

        records = await self._session.execute_write(_tx)
        meta.execution_time_ms = (time.perf_counter() - start) * 1000
        meta.affected_nodes = len(records)
        return records, meta

    def _node_props(self, record: dict[str, Any], key: str = "n") -> dict[str, Any]:
        node = record.get(key) or record.get("s") or record.get("r")
        if node is None:
            return dict(record)
        if hasattr(node, "items"):
            return dict(node)
        return dict(record)
