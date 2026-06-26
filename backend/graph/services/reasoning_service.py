"""Graph reasoning service."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession

from graph.config.loader import get_graph_config
from graph.repositories import PredictionRepository, SessionRepository


class ReasoningService:
    """Inference paths and longitudinal reasoning over the graph."""

    def __init__(self, session: AsyncSession) -> None:
        self._sessions = SessionRepository(session)
        self._predictions = PredictionRepository(session)
        self._config = get_graph_config()

    async def infer_risk_trend(self, user_id: str) -> dict[str, Any]:
        progression = await self._predictions.risk_progression(user_id)
        if len(progression) < 2:
            return {"user_id": user_id, "trend": "insufficient_data", "delta": 0.0}

        scores = [float(p.get("risk", 0)) for p in progression if p.get("risk") is not None]
        if len(scores) < 2:
            return {"user_id": user_id, "trend": "insufficient_data", "delta": 0.0}

        delta = scores[-1] - scores[0]
        threshold = float(self._config.reasoning.get("reasoning", {}).get("confidence_threshold", 0.5))
        trend = "stable"
        if delta > 0.1:
            trend = "increasing"
        elif delta < -0.1:
            trend = "decreasing"

        return {
            "user_id": user_id,
            "trend": trend,
            "delta": round(delta, 4),
            "confidence": min(1.0, abs(delta) + threshold),
            "sessions_analyzed": len(scores),
        }

    async def session_context(self, session_id: str) -> dict[str, Any]:
        session = await self._sessions.get_by_id(session_id)
        if not session:
            return {"session_id": session_id, "found": False}
        return {"session_id": session_id, "found": True, "session": session}
