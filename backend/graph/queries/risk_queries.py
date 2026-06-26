"""Risk graph Cypher queries."""

from __future__ import annotations

from typing import Any


def risk_timeline(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "-[:PRODUCED|PREDICTED]->(p:Prediction)-[:ESTIMATES|HAS_RISK]->(r:Risk) "
        "RETURN r.risk_id AS risk_id, r.risk_score AS score, r.risk_level AS level, "
        "r.trend AS trend, r.confidence AS confidence, p.created_at AS ts "
        "ORDER BY ts ASC",
        {"user_id": user_id},
    )


def average_risk(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(:Session) "
        "-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "RETURN avg(coalesce(p.risk_probability, p.risk_score)) AS avg_risk, count(p) AS n",
        {"user_id": user_id},
    )


def high_risk_sessions(threshold: float = 0.7) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (s:Session)-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "WHERE coalesce(p.risk_probability, p.risk_score) >= $threshold "
        "RETURN s.session_id AS session, coalesce(p.risk_probability, p.risk_score) AS risk "
        "ORDER BY risk DESC",
        {"threshold": threshold},
    )
