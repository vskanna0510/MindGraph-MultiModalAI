"""Graph analytics service."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession

from graph.config.loader import get_graph_config
from graph.repositories import EmotionRepository, PredictionRepository, RiskRepository, SymptomRepository


class AnalyticsService:
    """Clinical trend analysis and graph statistics."""

    def __init__(self, session: AsyncSession) -> None:
        self._predictions = PredictionRepository(session)
        self._risks = RiskRepository(session)
        self._emotions = EmotionRepository(session)
        self._symptoms = SymptomRepository(session)
        self._config = get_graph_config()

    async def risk_progression(self, user_id: str) -> list[dict[str, Any]]:
        return await self._predictions.risk_progression(user_id)

    async def emotion_evolution(self, user_id: str) -> list[dict[str, Any]]:
        return await self._emotions.evolution(user_id)

    async def symptom_evolution(self, user_id: str) -> list[dict[str, Any]]:
        return await self._symptoms.evolution(user_id)

    async def high_risk_sessions(self, threshold: float | None = None) -> list[dict[str, Any]]:
        cfg = self._config.analytics.get("risk", {})
        thresh = threshold or float(cfg.get("high_threshold", 0.7))
        return await self._risks.high_risk_sessions(thresh)

    async def clinical_trends(self, user_id: str) -> dict[str, Any]:
        risk = await self.risk_progression(user_id)
        emotions = await self.emotion_evolution(user_id)
        symptoms = await self.symptom_evolution(user_id)
        min_sessions = int(self._config.analytics.get("trends", {}).get("min_sessions", 3))
        return {
            "user_id": user_id,
            "risk_points": len(risk),
            "emotion_points": len(emotions),
            "symptom_points": len(symptoms),
            "trend_available": len(risk) >= min_sessions,
            "risk": risk,
            "emotions": emotions,
            "symptoms": symptoms,
        }
