"""Subgraph extraction Cypher queries."""

from __future__ import annotations

from typing import Any

RADIUS_LABELS = {
    "symptom": "Symptom",
    "emotion": "Emotion",
    "recommendation": "Recommendation",
    "prediction": "Prediction",
    "behaviour": "Behaviour",
}


def user_neighborhood(user_id: str, hops: int = 2) -> tuple[str, dict[str, Any]]:
    return (
        f"MATCH path=(u:User {{user_id: $user_id}})-[*1..{hops}]-(n) "
        "RETURN path LIMIT 500",
        {"user_id": user_id},
    )


def session_subgraph(session_id: str, hops: int = 2) -> tuple[str, dict[str, Any]]:
    return (
        f"MATCH path=(s:Session {{session_id: $session_id}})-[*1..{hops}]-(n) "
        "RETURN path LIMIT 500",
        {"session_id": session_id},
    )


def radius_subgraph(user_id: str, label: str, hops: int = 2) -> tuple[str, dict[str, Any]]:
    node_label = RADIUS_LABELS.get(label, label.capitalize())
    return (
        f"MATCH (u:User {{user_id: $user_id}})-[:HAS_SESSION]->(:Session)-[*1..{hops}]-(n:{node_label}) "
        "RETURN DISTINCT n LIMIT 200",
        {"user_id": user_id},
    )


def temporal_radius_subgraph(user_id: str, since_iso: str, hops: int = 2) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "WHERE s.timestamp >= $since "
        f"WITH s MATCH path=(s)-[*1..{hops}]-(n) "
        "RETURN path LIMIT 500",
        {"user_id": user_id, "since": since_iso},
    )
