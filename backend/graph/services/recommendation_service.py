"""Graph recommendation service."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession

from app.exceptions.base import ValidationError
from graph.cache import GraphCache
from graph.config.loader import get_graph_config
from graph.entities.factories import RecommendationFactory
from graph.repositories import RecommendationRepository
from graph.validators import GraphValidator

INTERVENTION_RULES: list[dict[str, Any]] = [
    {"intervention_type": "Breathing Exercise", "min_risk": 0.3, "max_risk": 0.6},
    {"intervention_type": "Journaling", "min_risk": 0.2, "max_risk": 0.5},
    {"intervention_type": "Professional Referral", "min_risk": 0.7, "max_risk": 1.0},
    {"intervention_type": "Mindfulness", "min_risk": 0.4, "max_risk": 0.7},
]


class RecommendationService:
    """Intervention recommendations from graph state."""

    def __init__(self, session: AsyncSession, *, cache: GraphCache | None = None) -> None:
        self._repo = RecommendationRepository(session)
        self._cache = cache or GraphCache()
        self._validator = GraphValidator()
        self._config = get_graph_config()

    async def for_user(self, user_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        cache_key = f"graph:rec:{user_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        max_results = limit or int(self._config.recommendation.get("recommendation", {}).get("max_results", 10))
        recs = await self._repo.for_user(user_id, max_results)
        self._cache.set(cache_key, recs)
        return recs

    async def suggest_from_risk(
        self,
        session_id: str,
        user_id: str,
        *,
        owner: str,
        risk_score: float,
    ) -> list[dict[str, Any]]:
        min_conf = float(self._config.recommendation.get("recommendation", {}).get("min_confidence", 0.6))
        created: list[dict[str, Any]] = []

        for rule in INTERVENTION_RULES:
            if rule["min_risk"] <= risk_score <= rule["max_risk"]:
                entity = RecommendationFactory.create(
                    rule["intervention_type"],
                    owner=owner,
                    confidence=min(1.0, risk_score + 0.1),
                )
                result = self._validator.validate_entity(entity)
                if not result.valid:
                    raise ValidationError("; ".join(result.errors))
                if entity.identity.confidence < min_conf:
                    continue
                rec_id = entity.properties["recommendation_id"]
                saved = await self._repo.save(entity)
                await self._repo.link_to_session(session_id, rec_id)
                created.append(saved)

        self._cache.invalidate_user(user_id)
        return created
