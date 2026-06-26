"""Graph explainability service."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession

from graph.config.loader import get_graph_config
from graph.repositories import EmotionRepository, PredictionRepository, SymptomRepository


class ExplainabilityService:
    """Explainable paths from session → symptoms → prediction."""

    def __init__(self, session: AsyncSession) -> None:
        self._predictions = PredictionRepository(session)
        self._emotions = EmotionRepository(session)
        self._symptoms = SymptomRepository(session)
        self._config = get_graph_config()

    async def explain_user_state(self, user_id: str) -> dict[str, Any]:
        risk = await self._predictions.risk_progression(user_id)
        emotions = await self._emotions.evolution(user_id)
        symptoms = await self._symptoms.evolution(user_id)

        latest_risk = risk[-1] if risk else None
        top_emotions = sorted(emotions, key=lambda e: float(e.get("prob", 0)), reverse=True)[:3]
        top_symptoms = sorted(symptoms, key=lambda s: float(s.get("conf", 0)), reverse=True)[:3]

        max_path = int(self._config.reasoning.get("explainability", {}).get("max_path_length", 6))
        path = ["User", "Session"]
        if top_emotions:
            path.append("Emotion")
        if top_symptoms:
            path.append("Symptom")
        path.append("Prediction")
        path = path[:max_path]

        return {
            "user_id": user_id,
            "reasoning_path": path,
            "latest_risk": latest_risk,
            "contributing_emotions": top_emotions,
            "contributing_symptoms": top_symptoms,
            "explanation": self._build_explanation(latest_risk, top_emotions, top_symptoms),
        }

    @staticmethod
    def _build_explanation(
        risk: dict[str, Any] | None,
        emotions: list[dict[str, Any]],
        symptoms: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        if risk:
            parts.append(f"Latest risk score {risk.get('risk', 'N/A')} for session {risk.get('session', 'unknown')}.")
        if emotions:
            emo = emotions[0].get("emotion", "unknown")
            parts.append(f"Dominant emotion: {emo}.")
        if symptoms:
            sym = symptoms[0].get("symptom", "unknown")
            parts.append(f"Primary symptom indicator: {sym}.")
        return " ".join(parts) if parts else "Insufficient graph data for explanation."
