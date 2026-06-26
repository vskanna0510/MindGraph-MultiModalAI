"""Temporal reasoning — detect improvement, deterioration, relapse, recovery."""

from __future__ import annotations

from enum import Enum
from typing import Any

from graph.reasoning.risk_trend import RiskTrendEngine, RiskTrendMetrics


class TemporalState(str, Enum):
    IMPROVEMENT = "improvement"
    DETERIORATION = "deterioration"
    SUDDEN_CHANGE = "sudden_change"
    STABLE = "stable"
    RECOVERY = "recovery"
    RELAPSE = "relapse"
    MISSING_CHECKINS = "missing_checkins"
    INSUFFICIENT_DATA = "insufficient_data"


class TemporalReasoningEngine:
    """Detect longitudinal behavioural and risk patterns."""

    def __init__(
        self,
        *,
        sudden_change_threshold: float = 0.2,
        deterioration_slope: float = 0.05,
        recovery_threshold: float = -0.15,
    ) -> None:
        self._sudden = sudden_change_threshold
        self._deterioration_slope = deterioration_slope
        self._recovery = recovery_threshold

    def analyze_risk(
        self,
        scores: list[float],
        *,
        missing_sessions: int = 0,
    ) -> dict[str, Any]:
        if len(scores) < 2:
            return {
                "state": TemporalState.INSUFFICIENT_DATA.value,
                "confidence": 0.0,
                "metrics": RiskTrendEngine.to_dict(RiskTrendEngine.compute(scores)),
            }

        metrics = RiskTrendEngine.compute(scores)
        state = self._classify(metrics, scores, missing_sessions)

        return {
            "state": state.value,
            "confidence": min(1.0, abs(metrics.slope) + 0.5),
            "metrics": RiskTrendEngine.to_dict(metrics),
            "long_term_trend": "increasing" if metrics.slope > 0 else "decreasing" if metrics.slope < 0 else "flat",
            "missing_checkins": missing_sessions,
        }

    def analyze_emotion_stability(self, probabilities: list[float]) -> dict[str, Any]:
        if len(probabilities) < 2:
            return {"stability": 1.0, "volatility": 0.0}
        mean = sum(probabilities) / len(probabilities)
        variance = sum((p - mean) ** 2 for p in probabilities) / len(probabilities)
        volatility = variance ** 0.5
        return {
            "stability": round(max(0.0, 1.0 - volatility), 4),
            "volatility": round(volatility, 4),
            "mean": round(mean, 4),
        }

    def _classify(
        self,
        metrics: RiskTrendMetrics,
        scores: list[float],
        missing_sessions: int,
    ) -> TemporalState:
        if missing_sessions > 0:
            return TemporalState.MISSING_CHECKINS

        recent_delta = scores[-1] - scores[-2] if len(scores) >= 2 else 0.0
        if abs(recent_delta) >= self._sudden:
            return TemporalState.SUDDEN_CHANGE

        if metrics.slope >= self._deterioration_slope:
            if len(scores) >= 4 and scores[-1] > metrics.average_risk and scores[-4] < metrics.average_risk:
                return TemporalState.RELAPSE
            return TemporalState.DETERIORATION

        if metrics.slope <= self._recovery:
            if scores[-1] < metrics.average_risk:
                return TemporalState.RECOVERY
            return TemporalState.IMPROVEMENT

        return TemporalState.STABLE
