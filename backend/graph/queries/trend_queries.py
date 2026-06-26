"""Trend analysis Cypher queries."""

from __future__ import annotations

from typing import Any


def risk_trend(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(:Session) "
        "-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "WITH coalesce(p.risk_probability, p.risk_score) AS risk, p.created_at AS ts "
        "ORDER BY ts ASC "
        "RETURN collect(risk) AS risks, collect(ts) AS timestamps",
        {"user_id": user_id},
    )


def emotion_trend(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(:Session) "
        "-[:EXPRESSES|IDENTIFIED]->(e:Emotion) "
        "RETURN coalesce(e.emotion_name, e.emotion) AS emotion, "
        "e.probability AS prob, e.timestamp AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def symptom_trend(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(:Session) "
        "-[:INDICATES|IDENTIFIED]->(s:Symptom) "
        "RETURN coalesce(s.symptom_name, s.symptom) AS symptom, "
        "s.severity AS severity, s.confidence AS conf, s.timestamp AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def behaviour_trend(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(:Session) "
        "-[:PRODUCED|PREDICTED]->(:Prediction)-[:IDENTIFIED]->(b:Behaviour) "
        "RETURN b.behaviour_name AS behaviour, b.value AS value, "
        "b.confidence AS conf, b.timestamp AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def modality_quality_trend(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "RETURN s.session_id AS session, s.quality_score AS quality, "
        "s.timestamp AS ts, s.offline_mode AS offline ORDER BY ts ASC",
        {"user_id": user_id},
    )
