"""Longitudinal graph queries."""

from __future__ import annotations

from typing import Any


def risk_over_time(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "OPTIONAL MATCH (p)-[:ESTIMATES|HAS_RISK]->(r:Risk) "
        "RETURN s.session_id AS session, s.timestamp AS session_ts, "
        "p.prediction_id AS prediction_id, "
        "coalesce(p.risk_probability, p.risk_score) AS risk, "
        "p.confidence AS confidence, p.created_at AS ts, "
        "r.risk_level AS risk_level, r.trend AS trend "
        "ORDER BY ts ASC",
        {"user_id": user_id},
    )


def emotion_evolution(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "MATCH (s)-[:EXPRESSES|IDENTIFIED]->(e:Emotion) "
        "RETURN coalesce(e.emotion_name, e.emotion) AS emotion, "
        "e.probability AS prob, e.confidence AS conf, e.timestamp AS ts "
        "ORDER BY ts ASC",
        {"user_id": user_id},
    )


def symptom_progression(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "MATCH (s)-[:INDICATES|IDENTIFIED]->(sym:Symptom) "
        "RETURN coalesce(sym.symptom_name, sym.symptom) AS symptom, "
        "sym.severity AS severity, sym.confidence AS conf, sym.timestamp AS ts "
        "ORDER BY ts ASC",
        {"user_id": user_id},
    )


def behaviour_evolution(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "MATCH (p)-[:IDENTIFIED]->(b:Behaviour) "
        "RETURN b.behaviour_name AS behaviour, b.value AS value, "
        "b.confidence AS conf, b.timestamp AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def recommendation_history(user_id: str, limit: int = 50) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "MATCH (s)-[:RECOMMENDS|GENERATED_REC]->(r:Recommendation) "
        "RETURN r.recommendation_id AS id, "
        "coalesce(r.recommendation_type, r.intervention_type) AS type, "
        "r.priority AS priority, r.status AS status, "
        "coalesce(r.generated_at, r.timestamp) AS ts "
        "ORDER BY ts DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )


def missed_sessions(user_id: str, expected_gap_days: int = 7) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "WITH u, s ORDER BY s.timestamp ASC "
        "WITH u, collect(s) AS sessions "
        "UNWIND range(0, size(sessions)-2) AS i "
        "WITH sessions[i] AS s1, sessions[i+1] AS s2 "
        "WHERE duration.between(datetime(s1.timestamp), datetime(s2.timestamp)).days > $gap "
        "RETURN s1.session_id AS from_session, s2.session_id AS to_session, "
        "s1.timestamp AS from_ts, s2.timestamp AS to_ts",
        {"user_id": user_id, "gap": expected_gap_days},
    )


def improvement_trend(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "WITH coalesce(p.risk_probability, p.risk_score) AS risk, p.created_at AS ts "
        "ORDER BY ts ASC "
        "WITH collect(risk) AS risks "
        "WHERE size(risks) >= 2 "
        "RETURN risks[0] AS first_risk, risks[-1] AS latest_risk, "
        "risks[-1] - risks[0] AS delta, "
        "CASE WHEN risks[-1] < risks[0] THEN 'improving' ELSE 'declining' END AS trend",
        {"user_id": user_id},
    )


def decline_trend(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "WITH coalesce(p.risk_probability, p.risk_score) AS risk, p.created_at AS ts "
        "ORDER BY ts ASC "
        "WITH collect(risk) AS risks "
        "WHERE size(risks) >= 3 "
        "RETURN risks[-3] AS risk_3_ago, risks[-1] AS latest, "
        "risks[-1] - risks[-3] AS recent_delta, "
        "CASE WHEN risks[-1] > risks[-3] THEN true ELSE false END AS declining",
        {"user_id": user_id},
    )


def user_snapshot_subgraph(user_id: str) -> tuple[str, dict[str, Any]]:
    """Full user memory subgraph for snapshot export."""
    return (
        "MATCH (u:User {user_id: $user_id}) "
        "OPTIONAL MATCH (u)-[r1]->(n1) "
        "OPTIONAL MATCH (n1)-[r2]->(n2) "
        "RETURN u, collect(DISTINCT n1) AS level1, collect(DISTINCT n2) AS level2, "
        "count(DISTINCT n1) + count(DISTINCT n2) AS node_count",
        {"user_id": user_id},
    )


def soft_deleted_nodes(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (n {owner: $user_id}) WHERE n.deleted_at IS NOT NULL "
        "RETURN labels(n) AS labels, n.uuid AS uuid, n.deleted_at AS deleted_at",
        {"user_id": user_id},
    )
