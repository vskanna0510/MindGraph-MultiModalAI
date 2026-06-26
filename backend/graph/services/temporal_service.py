"""Temporal graph service."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession

from graph.cache import GraphCache
from graph.config.loader import get_graph_config
from graph.queries import temporal_chain
from graph.repositories import SessionRepository, UserRepository


class TemporalService:
    """Longitudinal timeline and temporal chain queries."""

    def __init__(self, session: AsyncSession, *, cache: GraphCache | None = None) -> None:
        self._session = session
        self._users = UserRepository(session)
        self._sessions = SessionRepository(session)
        self._cache = cache or GraphCache()
        self._config = get_graph_config()

    async def timeline(self, user_id: str) -> list[dict[str, Any]]:
        cache_key = f"graph:temporal:{user_id}:full"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        sessions = await self._users.timeline(user_id)
        self._cache.set(cache_key, sessions)
        return sessions

    async def session_history(self, user_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        max_sessions = limit or int(self._config.temporal.get("temporal", {}).get("max_history_sessions", 12))
        return await self._sessions.history(user_id, max_sessions)

    async def temporal_chain(self, user_id: str) -> dict[str, Any]:
        cypher, params = temporal_chain(user_id)
        records, meta = await self._users._run(cypher, params)
        return {
            "user_id": user_id,
            "chain_length": len(records),
            "execution_time_ms": meta.execution_time_ms,
        }

    async def session_distance(self, user_id: str) -> dict[str, Any]:
        sessions = await self.timeline(user_id)
        if len(sessions) < 2:
            return {"user_id": user_id, "distances": [], "count": len(sessions)}
        distances = []
        for i in range(1, len(sessions)):
            prev_ts = sessions[i - 1].get("timestamp", "")
            curr_ts = sessions[i].get("timestamp", "")
            distances.append({"from_index": i - 1, "to_index": i, "from_ts": prev_ts, "to_ts": curr_ts})
        return {"user_id": user_id, "distances": distances, "count": len(sessions)}
