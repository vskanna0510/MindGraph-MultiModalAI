"""Personalized recommendation prioritization engine."""

from __future__ import annotations

from typing import Any

RECOMMENDATION_CATALOG: list[dict[str, Any]] = [
    # Low risk
    {"type": "Positive Reinforcement", "tier": "low", "min_risk": 0.0, "max_risk": 0.35, "benefit": 0.6, "urgency": 0.2},
    {"type": "Journaling", "tier": "low", "min_risk": 0.1, "max_risk": 0.4, "benefit": 0.7, "urgency": 0.3},
    {"type": "Mindfulness", "tier": "low", "min_risk": 0.2, "max_risk": 0.45, "benefit": 0.65, "urgency": 0.25},
    {"type": "Exercise", "tier": "low", "min_risk": 0.0, "max_risk": 0.4, "benefit": 0.7, "urgency": 0.2},
    {"type": "Sleep Reminder", "tier": "low", "min_risk": 0.15, "max_risk": 0.45, "benefit": 0.6, "urgency": 0.3},
    # Medium risk
    {"type": "Breathing Exercise", "tier": "medium", "min_risk": 0.35, "max_risk": 0.65, "benefit": 0.75, "urgency": 0.5},
    {"type": "Mood Tracking", "tier": "medium", "min_risk": 0.4, "max_risk": 0.65, "benefit": 0.7, "urgency": 0.45},
    {"type": "Scheduled Reflection", "tier": "medium", "min_risk": 0.35, "max_risk": 0.6, "benefit": 0.65, "urgency": 0.4},
    {"type": "Professional Counselling Suggestion", "tier": "medium", "min_risk": 0.55, "max_risk": 0.7, "benefit": 0.8, "urgency": 0.6},
    # High risk
    {"type": "Immediate Mental Health Resource", "tier": "high", "min_risk": 0.7, "max_risk": 0.85, "benefit": 0.9, "urgency": 0.85},
    {"type": "Emergency Helpline", "tier": "high", "min_risk": 0.8, "max_risk": 1.0, "benefit": 0.95, "urgency": 1.0},
    {"type": "Trusted Contact Prompt", "tier": "high", "min_risk": 0.65, "max_risk": 0.9, "benefit": 0.85, "urgency": 0.8},
    {"type": "Escalation Notice", "tier": "high", "min_risk": 0.75, "max_risk": 1.0, "benefit": 0.9, "urgency": 0.9},
]


class RecommendationPrioritizer:
    """Score and rank personalized recommendations."""

    def __init__(
        self,
        *,
        low_threshold: float = 0.4,
        medium_threshold: float = 0.7,
        diversity_weight: float = 0.15,
    ) -> None:
        self._low = low_threshold
        self._medium = medium_threshold
        self._diversity = diversity_weight

    def generate(
        self,
        *,
        current_risk: float,
        twin_summary: dict[str, Any] | None = None,
        compliance_history: float = 0.5,
        language: str = "en",
        offline_available: bool = True,
        max_results: int = 8,
        previous_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        prev = set(previous_types or [])
        candidates: list[dict[str, Any]] = []

        for item in RECOMMENDATION_CATALOG:
            if not item["min_risk"] <= current_risk <= item["max_risk"]:
                continue
            score = self._score(item, current_risk, compliance_history, item["type"] in prev)
            candidates.append({
                **item,
                "priority_score": round(score, 4),
                "confidence": round(min(1.0, score * 0.9 + 0.1), 4),
                "language": language,
                "offline_capable": offline_available,
                "guidance_only": True,
                "is_diagnosis": False,
            })

        candidates.sort(key=lambda x: x["priority_score"], reverse=True)
        return self._diversify(candidates, max_results)

    def _score(
        self,
        item: dict[str, Any],
        risk: float,
        compliance: float,
        seen_before: bool,
    ) -> float:
        benefit = float(item["benefit"])
        urgency = float(item["urgency"])
        historical_success = compliance * 0.3
        diversity_penalty = self._diversity if seen_before else 0.0
        clinical_priority = urgency * 0.2 if item["tier"] == "high" else 0.0
        risk_fit = 1.0 - abs(risk - (item["min_risk"] + item["max_risk"]) / 2)

        return benefit * 0.35 + urgency * 0.25 + historical_success + risk_fit * 0.1 + clinical_priority - diversity_penalty

    @staticmethod
    def _diversify(candidates: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_tiers: set[str] = set()
        for c in candidates:
            if len(selected) >= max_results:
                break
            tier = c.get("tier", "low")
            if tier in seen_tiers and len(selected) >= 2:
                continue
            selected.append(c)
            seen_tiers.add(tier)
        if len(selected) < max_results:
            for c in candidates:
                if c not in selected:
                    selected.append(c)
                if len(selected) >= max_results:
                    break
        return selected
