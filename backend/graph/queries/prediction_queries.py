"""Prediction graph Cypher queries."""

from __future__ import annotations

from typing import Any


def latest_prediction(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "RETURN p, s.session_id AS session_id "
        "ORDER BY p.created_at DESC LIMIT 1",
        {"user_id": user_id},
    )


def prediction_history(user_id: str, limit: int = 50) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "RETURN p.prediction_id AS prediction_id, "
        "coalesce(p.risk_probability, p.risk_score) AS risk, "
        "p.confidence AS confidence, p.created_at AS ts, s.session_id AS session "
        "ORDER BY p.created_at DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )
