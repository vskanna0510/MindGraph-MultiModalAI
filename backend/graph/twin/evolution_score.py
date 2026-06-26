"""Graph evolution score — twin quality metrics."""

from __future__ import annotations

from typing import Any

from graph.twin.digital_twin import DigitalTwin


class EvolutionScoreCalculator:
    """Measure knowledge growth, stability, and twin completeness."""

    @staticmethod
    def compute(twin: DigitalTwin, *, session_count: int = 0) -> dict[str, Any]:
        sessions = session_count or twin.temporal_memory_sessions
        knowledge_growth = min(1.0, sessions / 20)
        behaviour_stability = 1.0 - min(1.0, twin.emotion.variability)
        prediction_stability = twin.recovery.stability
        rec_effectiveness = twin.recovery.recommendation_compliance
        embedding_consistency = min(1.0, len(twin.embedding_refs) / max(sessions, 1))
        temporal_completeness = min(1.0, sessions / 12)

        components = {
            "knowledge_growth": round(knowledge_growth, 4),
            "behaviour_stability": round(behaviour_stability, 4),
            "prediction_stability": round(prediction_stability, 4),
            "recommendation_effectiveness": round(rec_effectiveness, 4),
            "embedding_consistency": round(embedding_consistency, 4),
            "temporal_completeness": round(temporal_completeness, 4),
        }
        overall = round(sum(components.values()) / len(components), 4)

        return {
            "overall_score": overall,
            "components": components,
            "maturity": "mature" if overall >= 0.7 else "developing" if overall >= 0.4 else "nascent",
        }
