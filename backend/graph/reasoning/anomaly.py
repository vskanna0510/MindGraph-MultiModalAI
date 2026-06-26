"""Graph anomaly detection."""

from __future__ import annotations

from typing import Any

from graph.reasoning.risk_trend import RiskTrendEngine


class AnomalyDetector:
    """Detect unexpected risk increases, mood shifts, outlier behaviour."""

    @staticmethod
    def unexpected_risk_increase(
        scores: list[float],
        *,
        z_threshold: float = 2.0,
    ) -> dict[str, Any] | None:
        if len(scores) < 3:
            return None
        mean = sum(scores[:-1]) / len(scores[:-1])
        variance = sum((s - mean) ** 2 for s in scores[:-1]) / max(len(scores[:-1]) - 1, 1)
        std = variance ** 0.5 or 0.01
        latest = scores[-1]
        z = (latest - mean) / std
        if z >= z_threshold:
            return {
                "type": "unexpected_risk_increase",
                "z_score": round(z, 4),
                "latest": latest,
                "baseline_mean": round(mean, 4),
            }
        return None

    @staticmethod
    def sudden_mood_shift(
        probabilities: list[float],
        *,
        shift_threshold: float = 0.3,
    ) -> dict[str, Any] | None:
        if len(probabilities) < 2:
            return None
        delta = probabilities[-1] - probabilities[-2]
        if abs(delta) >= shift_threshold:
            return {
                "type": "sudden_mood_shift",
                "delta": round(delta, 4),
                "from": probabilities[-2],
                "to": probabilities[-1],
            }
        return None

    @staticmethod
    def missing_sessions(gap_days: list[int], threshold: int = 7) -> dict[str, Any] | None:
        if any(g > threshold for g in gap_days):
            return {
                "type": "missing_sessions",
                "max_gap_days": max(gap_days),
                "threshold": threshold,
            }
        return None

    @staticmethod
    def scan(
        risk_scores: list[float],
        emotion_probs: list[float] | None = None,
        gap_days: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        anomalies: list[dict[str, Any]] = []
        risk_anomaly = AnomalyDetector.unexpected_risk_increase(risk_scores)
        if risk_anomaly:
            anomalies.append(risk_anomaly)
        if emotion_probs:
            mood = AnomalyDetector.sudden_mood_shift(emotion_probs)
            if mood:
                anomalies.append(mood)
        if gap_days:
            missing = AnomalyDetector.missing_sessions(gap_days)
            if missing:
                anomalies.append(missing)
        return anomalies
