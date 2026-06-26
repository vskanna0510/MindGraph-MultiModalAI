"""Research evaluation metrics for graph intelligence."""

from __future__ import annotations

from typing import Any

from graph.twin.digital_twin import DigitalTwin


class ResearchEvaluator:
    """Evaluate graph contribution, twin consistency, and reasoning latency."""

    @staticmethod
    def evaluate(
        *,
        twin: DigitalTwin,
        evolution_score: dict[str, Any],
        reasoning_latency_ms: float,
        recommendation_count: int,
        trend_accuracy: float | None = None,
    ) -> dict[str, Any]:
        return {
            "graph_contribution": evolution_score.get("components", {}).get("knowledge_growth", 0),
            "temporal_contribution": evolution_score.get("components", {}).get("temporal_completeness", 0),
            "recommendation_count": recommendation_count,
            "trend_detection_accuracy": trend_accuracy,
            "digital_twin_consistency": evolution_score.get("overall_score", 0),
            "graph_growth_sessions": twin.temporal_memory_sessions,
            "reasoning_latency_ms": reasoning_latency_ms,
            "twin_maturity": evolution_score.get("maturity", "nascent"),
            "research_mode": True,
        }
