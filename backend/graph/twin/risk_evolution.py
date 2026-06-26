"""Risk evolution and forecasting model."""

from __future__ import annotations

from typing import Any

from graph.reasoning.risk_trend import RiskTrendEngine
from graph.reasoning.temporal_reasoning import TemporalReasoningEngine


class RiskEvolutionModel:
    """Predict current/future risk, recovery and relapse probability."""

    def __init__(self, min_sessions: int = 3) -> None:
        self._min_sessions = min_sessions
        self._temporal = TemporalReasoningEngine()

    def analyze(self, risk_scores: list[float]) -> dict[str, Any]:
        if not risk_scores:
            return self._empty()

        metrics = RiskTrendEngine.compute(risk_scores)
        temporal = self._temporal.analyze_risk(risk_scores)
        current = metrics.current_risk

        return {
            "current_risk": current,
            "future_risk": self._forecast(risk_scores, 7),
            "forecast_7d": self._forecast(risk_scores, 7),
            "forecast_30d": self._forecast(risk_scores, 30),
            "forecast_90d": self._forecast(risk_scores, 90),
            "recovery_probability": self._recovery_probability(risk_scores, metrics.slope),
            "relapse_probability": self._relapse_probability(risk_scores, metrics.slope),
            "prediction_confidence": min(1.0, metrics.n_points / 10),
            "temporal_state": temporal["state"],
            "metrics": RiskTrendEngine.to_dict(metrics),
            "sufficient_data": len(risk_scores) >= self._min_sessions,
        }

    @staticmethod
    def _forecast(scores: list[float], days: int) -> float:
        if len(scores) < 2:
            return scores[-1] if scores else 0.0
        slope = RiskTrendEngine._slope(scores)
        horizon_factor = days / 30
        projected = scores[-1] + slope * horizon_factor * len(scores)
        return round(max(0.0, min(1.0, projected)), 4)

    @staticmethod
    def _recovery_probability(scores: list[float], slope: float) -> float:
        if len(scores) < 2:
            return 0.5
        if slope < -0.05:
            return round(min(1.0, 0.6 + abs(slope)), 4)
        return round(max(0.0, 0.4 - slope), 4)

    @staticmethod
    def _relapse_probability(scores: list[float], slope: float) -> float:
        if len(scores) < 3:
            return 0.2
        if slope > 0.05 and scores[-1] > sum(scores) / len(scores):
            return round(min(1.0, 0.5 + slope), 4)
        return round(max(0.0, 0.3 + slope * 0.5), 4)

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "current_risk": 0.0,
            "future_risk": 0.0,
            "forecast_7d": 0.0,
            "forecast_30d": 0.0,
            "forecast_90d": 0.0,
            "recovery_probability": 0.5,
            "relapse_probability": 0.2,
            "prediction_confidence": 0.0,
            "temporal_state": "insufficient_data",
            "metrics": {},
            "sufficient_data": False,
        }
