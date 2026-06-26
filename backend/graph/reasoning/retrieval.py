"""Graph retrieval engine — subgraph and neighborhood extraction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from neo4j import AsyncSession

from graph.cache import GraphCache
from graph.config.loader import get_graph_config
from graph.contracts import GraphServiceResponse, TimedOperation
from graph.queries import session_queries as sq
from graph.queries import subgraph_queries as sg
from graph.repositories.base import BaseGraphRepository


class TemporalWindow(str, Enum):
    PREVIOUS_SESSION = "previous_session"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    LAST_6_MONTHS = "last_6_months"
    ENTIRE_HISTORY = "entire_history"


WINDOW_DAYS: dict[TemporalWindow, int] = {
    TemporalWindow.LAST_7_DAYS: 7,
    TemporalWindow.LAST_30_DAYS: 30,
    TemporalWindow.LAST_90_DAYS: 90,
    TemporalWindow.LAST_6_MONTHS: 180,
    TemporalWindow.ENTIRE_HISTORY: -1,
}


class GraphRetrievalEngine:
    """User/session/neighborhood subgraph retrieval with temporal windows."""

    def __init__(self, session: AsyncSession, *, cache: GraphCache | None = None) -> None:
        self._repo = BaseGraphRepository(session)
        self._cache = cache or GraphCache()
        self._config = get_graph_config()

    def _since_iso(self, window: TemporalWindow) -> str | None:
        days = WINDOW_DAYS.get(window, -1)
        if days < 0:
            return None
        return (datetime.now(UTC) - timedelta(days=days)).isoformat()

    async def _run_cached(
        self,
        cache_key: str,
        cypher: str,
        params: dict[str, Any],
        *,
        correlation_id: str,
    ) -> GraphServiceResponse[list[dict[str, Any]]]:
        cached = self._cache.get(cache_key)
        if cached is not None:
            return GraphServiceResponse(
                result=cached,
                correlation_id=correlation_id,
                cache_hit=True,
                metadata={"rows": len(cached)},
            )
        with TimedOperation() as timer:
            records, _ = await self._repo._run(cypher, params)
        result = list(records)
        self._cache.set(cache_key, result)
        return GraphServiceResponse(
            result=result,
            execution_time_ms=timer.elapsed_ms,
            correlation_id=correlation_id,
            metadata={"rows": len(result)},
        )

    async def user_graph(
        self,
        user_id: str,
        hops: int = 2,
        *,
        correlation_id: str = "",
    ) -> GraphServiceResponse[list[dict[str, Any]]]:
        hops = min(hops, self._config.max_subgraph_depth)
        cypher, params = sg.user_neighborhood(user_id, hops)
        return await self._run_cached(f"retrieve:user:{user_id}:{hops}", cypher, params, correlation_id=correlation_id)

    async def session_graph(
        self,
        session_id: str,
        hops: int = 2,
        *,
        correlation_id: str = "",
    ) -> GraphServiceResponse[list[dict[str, Any]]]:
        hops = min(hops, self._config.max_subgraph_depth)
        cypher, params = sg.session_subgraph(session_id, hops)
        return await self._run_cached(
            f"retrieve:session:{session_id}:{hops}", cypher, params, correlation_id=correlation_id
        )

    async def radius_subgraph(
        self,
        user_id: str,
        radius: str,
        hops: int = 2,
        *,
        correlation_id: str = "",
    ) -> GraphServiceResponse[list[dict[str, Any]]]:
        cypher, params = sg.radius_subgraph(user_id, radius, hops)
        return await self._run_cached(
            f"retrieve:radius:{user_id}:{radius}:{hops}", cypher, params, correlation_id=correlation_id
        )

    async def temporal_window_sessions(
        self,
        user_id: str,
        window: TemporalWindow = TemporalWindow.LAST_30_DAYS,
        *,
        correlation_id: str = "",
    ) -> GraphServiceResponse[list[dict[str, Any]]]:
        since = self._since_iso(window)
        if since is None:
            cypher, params = sq.session_timeline(user_id)
        else:
            cypher, params = sq.sessions_in_window(user_id, since)
        cache_key = f"retrieve:window:{user_id}:{window.value}"
        return await self._run_cached(cache_key, cypher, params, correlation_id=correlation_id)
