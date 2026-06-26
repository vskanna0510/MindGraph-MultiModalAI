"""Risk trend computation engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class RiskTrendMetrics:
    current_risk: float
    average_risk: float
    peak_risk: float
    minimum_risk: float
    slope: float
    acceleration: float
    moving_average: float
    exponential_trend: float
    confidence_interval: tuple[float, float]
    n_points: int


class RiskTrendEngine:
    """Compute risk statistics from historical risk scores."""

    @staticmethod
    def compute(scores: list[float]) -> RiskTrendMetrics:
        if not scores:
            return RiskTrendMetrics(0, 0, 0, 0, 0, 0, 0, 0, (0, 0), 0)

        n = len(scores)
        current = scores[-1]
        avg = sum(scores) / n
        peak = max(scores)
        minimum = min(scores)
        slope = RiskTrendEngine._slope(scores)
        acceleration = RiskTrendEngine._acceleration(scores)
        ma = sum(scores[-min(3, n):]) / min(3, n)
        exp_trend = RiskTrendEngine._exponential_trend(scores)
        ci = RiskTrendEngine._confidence_interval(scores)

        return RiskTrendMetrics(
            current_risk=round(current, 4),
            average_risk=round(avg, 4),
            peak_risk=round(peak, 4),
            minimum_risk=round(minimum, 4),
            slope=round(slope, 4),
            acceleration=round(acceleration, 4),
            moving_average=round(ma, 4),
            exponential_trend=round(exp_trend, 4),
            confidence_interval=(round(ci[0], 4), round(ci[1], 4)),
            n_points=n,
        )

    @staticmethod
    def to_dict(metrics: RiskTrendMetrics) -> dict[str, Any]:
        return {
            "current_risk": metrics.current_risk,
            "average_risk": metrics.average_risk,
            "peak_risk": metrics.peak_risk,
            "minimum_risk": metrics.minimum_risk,
            "slope": metrics.slope,
            "acceleration": metrics.acceleration,
            "moving_average": metrics.moving_average,
            "exponential_trend": metrics.exponential_trend,
            "confidence_interval": list(metrics.confidence_interval),
            "n_points": metrics.n_points,
        }

    @staticmethod
    def _slope(values: list[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
        return num / den

    @staticmethod
    def _acceleration(values: list[float]) -> float:
        if len(values) < 3:
            return 0.0
        slopes = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        return slopes[-1] - slopes[-2]

    @staticmethod
    def _exponential_trend(values: list[float], alpha: float = 0.3) -> float:
        if not values:
            return 0.0
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    @staticmethod
    def _confidence_interval(values: list[float], z: float = 1.96) -> tuple[float, float]:
        n = len(values)
        if n < 2:
            m = values[0] if values else 0.0
            return m, m
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std_err = math.sqrt(variance / n)
        return mean - z * std_err, mean + z * std_err
