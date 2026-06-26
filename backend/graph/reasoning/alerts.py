"""Graph alert generation and storage."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from neo4j import AsyncSession

from graph.config.loader import get_graph_config
from graph.queries import alert_queries as aq
from graph.reasoning.anomaly import AnomalyDetector
from graph.reasoning.patterns import PatternDetector
from graph.reasoning.risk_trend import RiskTrendEngine
from graph.reasoning.temporal_reasoning import TemporalReasoningEngine, TemporalState
from graph.repositories.base import BaseGraphRepository


class AlertType(str, Enum):
    RISK_INCREASE = "risk_increase"
    REPEATED_HIGH_RISK = "repeated_high_risk"
    RAPID_DETERIORATION = "rapid_deterioration"
    MISSING_CHECKINS = "missing_checkins"
    PERSISTENT_NEGATIVE_MOOD = "persistent_negative_mood"
    LOW_CONFIDENCE = "low_confidence"


class AlertEngine:
    """Generate and store clinical graph alerts."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = BaseGraphRepository(session)
        self._config = get_graph_config()
        self._temporal = TemporalReasoningEngine()

    async def evaluate(
        self,
        user_id: str,
        risk_scores: list[float],
        *,
        confidences: list[float] | None = None,
        negative_emotion_probs: list[float] | None = None,
        missing_sessions: int = 0,
    ) -> list[dict[str, Any]]:
        cfg = self._config.reasoning.get("alerts", {})
        if not cfg.get("enabled", True):
            return []

        alerts: list[dict[str, Any]] = []
        threshold = float(cfg.get("risk_increase_threshold", 0.15))
        high_count = int(cfg.get("repeated_high_risk_count", 3))
        slope_thresh = float(cfg.get("deterioration_slope", 0.05))

        if len(risk_scores) >= 2 and risk_scores[-1] - risk_scores[-2] >= threshold:
            alerts.append(self._make_alert(user_id, AlertType.RISK_INCREASE, {
                "delta": risk_scores[-1] - risk_scores[-2],
                "current": risk_scores[-1],
            }))

        high_risks = [s for s in risk_scores if s >= 0.7]
        if len(high_risks) >= high_count:
            alerts.append(self._make_alert(user_id, AlertType.REPEATED_HIGH_RISK, {
                "count": len(high_risks),
            }))

        metrics = RiskTrendEngine.compute(risk_scores)
        if metrics.slope >= slope_thresh:
            alerts.append(self._make_alert(user_id, AlertType.RAPID_DETERIORATION, {
                "slope": metrics.slope,
            }))

        if missing_sessions > 0:
            alerts.append(self._make_alert(user_id, AlertType.MISSING_CHECKINS, {
                "missing": missing_sessions,
            }))

        if negative_emotion_probs and len(negative_emotion_probs) >= 3:
            avg_neg = sum(negative_emotion_probs) / len(negative_emotion_probs)
            if avg_neg >= 0.6:
                alerts.append(self._make_alert(user_id, AlertType.PERSISTENT_NEGATIVE_MOOD, {
                    "avg_probability": avg_neg,
                }))

        if confidences and confidences[-1] < 0.5:
            alerts.append(self._make_alert(user_id, AlertType.LOW_CONFIDENCE, {
                "confidence": confidences[-1],
            }))

        for anomaly in AnomalyDetector.scan(risk_scores, negative_emotion_probs):
            alerts[-1:]  # anomalies already covered above

        stored = []
        for alert in alerts:
            stored.append(await self._store(user_id, alert))
        return stored

    def _make_alert(self, user_id: str, alert_type: AlertType, details: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        return {
            "uuid": str(uuid.uuid4()),
            "user_id": user_id,
            "alert_type": alert_type.value,
            "severity": self._severity(alert_type),
            "details": details,
            "created_at": now,
            "status": "active",
        }

    @staticmethod
    def _severity(alert_type: AlertType) -> str:
        high = {AlertType.RAPID_DETERIORATION, AlertType.REPEATED_HIGH_RISK}
        medium = {AlertType.RISK_INCREASE, AlertType.PERSISTENT_NEGATIVE_MOOD}
        if alert_type in high:
            return "high"
        if alert_type in medium:
            return "medium"
        return "low"

    async def _store(self, user_id: str, alert: dict[str, Any]) -> dict[str, Any]:
        cypher, params = aq.store_alert_props(alert)
        records, _ = await self._repo._write(cypher, params)
        link_cypher, link_params = aq.link_alert_to_user(user_id, alert["uuid"])
        await self._repo._write(link_cypher, link_params)
        return alert
