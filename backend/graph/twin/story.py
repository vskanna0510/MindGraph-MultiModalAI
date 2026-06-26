"""Human-readable graph story generation."""

from __future__ import annotations

from typing import Any

from graph.twin.digital_twin import DigitalTwin


class GraphStoryGenerator:
    """Generate factual, non-diagnostic narrative summaries."""

    DISCLAIMERS = (
        "This summary reflects recorded interaction patterns only and is not a clinical diagnosis.",
        "Recommendations are supportive guidance, not medical advice.",
    )

    @staticmethod
    def generate(
        twin: DigitalTwin,
        *,
        window_days: int = 30,
        risk_evolution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parts: list[str] = []
        sessions = twin.temporal_memory_sessions

        if sessions == 0:
            narrative = "No session history is available yet to build your digital twin profile."
        else:
            parts.append(
                f"Over the past {window_days} days, {sessions} recorded interaction(s) "
                f"contributed to your evolving profile."
            )

            if twin.emotion.dominant_emotion != "Neutral":
                parts.append(
                    f"Dominant emotional tone observed: {twin.emotion.dominant_emotion} "
                    f"(stability score: {twin.emotion.stability:.2f})."
                )

            if twin.recovery.risk_reduction > 0.05:
                parts.append(
                    "Recorded interactions show a gradual improvement in engagement indicators, "
                    "with decreasing risk scores over time."
                )
            elif twin.recovery.risk_reduction < -0.05:
                parts.append(
                    "Recent sessions show increased risk indicators. "
                    "Consider the supportive recommendations provided."
                )
            else:
                parts.append("Your interaction patterns have remained relatively stable.")

            if twin.symptom.recurring:
                parts.append(
                    f"Recurring indicators noted: {', '.join(twin.symptom.recurring[:3])}."
                )

            if risk_evolution and risk_evolution.get("recovery_probability", 0) > 0.6:
                parts.append("Patterns suggest positive momentum in recent sessions.")

            narrative = " ".join(parts)

        return {
            "user_id": twin.user_id,
            "window_days": window_days,
            "narrative": narrative,
            "disclaimers": list(GraphStoryGenerator.DISCLAIMERS),
            "tone": "supportive",
            "is_diagnostic": False,
        }
