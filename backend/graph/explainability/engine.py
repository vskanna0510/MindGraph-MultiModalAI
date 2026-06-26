"""Enhanced graph explainability engine."""

from __future__ import annotations

from typing import Any

from graph.twin.digital_twin import DigitalTwin


class GraphExplainabilityEngine:
    """Full explanation path for every recommendation."""

    @staticmethod
    def explain_recommendation(
        recommendation: dict[str, Any],
        twin: DigitalTwin,
        *,
        supporting_sessions: list[dict[str, Any]] | None = None,
        model_version: str = "MODEL_010",
    ) -> dict[str, Any]:
        rec_type = recommendation.get("type", recommendation.get("recommendation_type", "Unknown"))
        tier = recommendation.get("tier", "medium")

        supporting_symptoms = list(twin.symptom.recurring[:3])
        supporting_emotions = [twin.emotion.dominant_emotion] + twin.emotion.rare_emotions[:2]
        supporting_behaviours = [
            k for k, v in twin.behaviour.to_dict().items()
            if isinstance(v, float) and v > 0.3 and k != "session_count"
        ][:3]

        path = [
            "Recommendation",
            "Supporting Symptoms" if supporting_symptoms else None,
            "Supporting Behaviour" if supporting_behaviours else None,
            "Supporting Sessions",
            "Historical Evidence",
            "Confidence",
        ]
        path = [p for p in path if p]

        why = GraphExplainabilityEngine._build_why(
            rec_type, tier, twin, supporting_symptoms, supporting_emotions
        )

        return {
            "recommendation": rec_type,
            "why": why,
            "which_sessions": [s.get("session_id", "") for s in (supporting_sessions or [])[:5]],
            "which_symptoms": supporting_symptoms,
            "which_emotions": supporting_emotions,
            "which_behaviours": supporting_behaviours,
            "temporal_pattern": twin.emotion.weekly_trend,
            "graph_relationships": ["TARGETS", "SUPPORTED_BY", "ADDRESSES"],
            "model_version": model_version,
            "explanation_path": path,
            "confidence": recommendation.get("confidence", 0.5),
            "guidance_disclaimer": "Supportive guidance only — not a clinical diagnosis.",
        }

    @staticmethod
    def _build_why(
        rec_type: str,
        tier: str,
        twin: DigitalTwin,
        symptoms: list[str],
        emotions: list[str],
    ) -> str:
        parts = [f"Recommended {rec_type} based on your recorded interaction history."]
        if twin.risk_scores:
            parts.append(f"Current risk level supports {tier}-priority guidance.")
        if symptoms:
            parts.append(f"Recurring indicators: {', '.join(symptoms)}.")
        if emotions:
            parts.append(f"Emotional patterns include {emotions[0]}.")
        parts.append(
            f"Recommendation confidence reflects {twin.recovery.recommendation_compliance:.0%} "
            "historical engagement with similar suggestions."
        )
        return " ".join(parts)
