"""Longitudinal query service."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession

from graph.cache import GraphCache
from graph.queries import longitudinal as lq
from graph.repositories.base import BaseGraphRepository


class LongitudinalService:
    """Risk/emotion/symptom/behaviour evolution and trend queries."""

    def __init__(self, session: AsyncSession, *, cache: GraphCache | None = None) -> None:
        self._repo = BaseGraphRepository(session)
        self._cache = cache or GraphCache()

    async def _cached_query(self, cache_key: str, query_fn: Any, *args: Any) -> list[dict[str, Any]]:
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        cypher, params = query_fn(*args)
        records, _ = await self._repo._run(cypher, params)
        result = list(records)
        self._cache.set(cache_key, result)
        return result

    async def risk_over_time(self, user_id: str) -> list[dict[str, Any]]:
        return await self._cached_query(f"long:risk:{user_id}", lq.risk_over_time, user_id)

    async def emotion_evolution(self, user_id: str) -> list[dict[str, Any]]:
        return await self._cached_query(f"long:emo:{user_id}", lq.emotion_evolution, user_id)

    async def symptom_progression(self, user_id: str) -> list[dict[str, Any]]:
        return await self._cached_query(f"long:sym:{user_id}", lq.symptom_progression, user_id)

    async def behaviour_evolution(self, user_id: str) -> list[dict[str, Any]]:
        return await self._cached_query(f"long:beh:{user_id}", lq.behaviour_evolution, user_id)

    async def recommendation_history(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cypher, params = lq.recommendation_history(user_id, limit)
        records, _ = await self._repo._run(cypher, params)
        return list(records)

    async def improvement_trend(self, user_id: str) -> dict[str, Any]:
        cypher, params = lq.improvement_trend(user_id)
        records, meta = await self._repo._run(cypher, params)
        if not records:
            return {"user_id": user_id, "trend": "insufficient_data"}
        return {**records[0], "user_id": user_id, "execution_time_ms": meta.execution_time_ms}

    async def decline_trend(self, user_id: str) -> dict[str, Any]:
        cypher, params = lq.decline_trend(user_id)
        records, _ = await self._repo._run(cypher, params)
        if not records:
            return {"user_id": user_id, "declining": False}
        return {**records[0], "user_id": user_id}

    async def full_longitudinal_report(self, user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "risk": await self.risk_over_time(user_id),
            "emotions": await self.emotion_evolution(user_id),
            "symptoms": await self.symptom_progression(user_id),
            "behaviours": await self.behaviour_evolution(user_id),
            "recommendations": await self.recommendation_history(user_id),
            "improvement": await self.improvement_trend(user_id),
            "decline": await self.decline_trend(user_id),
        }
