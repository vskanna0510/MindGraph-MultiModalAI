"""Longitudinal reasoning rules."""

from __future__ import annotations

from typing import Any

import torch


def assess_risk_trend(risk_scores: list[float]) -> str:
    if len(risk_scores) < 2:
        return "insufficient_history"
    delta = risk_scores[-1] - risk_scores[0]
    if delta > 0.15:
        return "risk_increasing"
    if delta < -0.15:
        return "risk_decreasing"
    if max(risk_scores) - min(risk_scores) < 0.05:
        return "stable"
    return "fluctuating"


def detect_sudden_change(scores: list[float], threshold: float = 0.25) -> bool:
    if len(scores) < 2:
        return False
    return abs(scores[-1] - scores[-2]) >= threshold


def detect_missing_checkins(expected: int, actual: int) -> bool:
    return actual < expected


def reason_over_history(risk_history: torch.Tensor | list[float]) -> dict[str, Any]:
    scores = risk_history.tolist() if isinstance(risk_history, torch.Tensor) else list(risk_history)
    trend = assess_risk_trend(scores)
    return {
        "trend": trend,
        "sudden_change": detect_sudden_change(scores),
        "recovery": trend == "risk_decreasing",
        "deterioration": trend == "risk_increasing",
        "latest_risk": scores[-1] if scores else 0.0,
    }
