"""Recommendation and feedback Cypher queries."""

from __future__ import annotations

from typing import Any


def current_recommendations(user_id: str, limit: int = 10) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(:Session) "
        "-[:RECOMMENDS|GENERATED_REC]->(r:Recommendation) "
        "WHERE r.status IN ['pending', 'active'] "
        "RETURN r ORDER BY r.generated_at DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )


def recommendation_history(user_id: str, limit: int = 50) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(:Session) "
        "-[:RECOMMENDS|GENERATED_REC]->(r:Recommendation) "
        "RETURN r.recommendation_id AS id, "
        "coalesce(r.recommendation_type, r.intervention_type) AS type, "
        "r.status AS status, r.confidence AS confidence, "
        "coalesce(r.generated_at, r.timestamp) AS ts "
        "ORDER BY ts DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )


def feedback_summary(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(:Session) "
        "-[:RECOMMENDS|GENERATED_REC]->(r:Recommendation)-[:FOLLOWED_BY]->(i:Intervention) "
        "RETURN r.recommendation_id AS rec_id, i.completion_status AS status, "
        "i.type AS intervention_type, count(*) AS cnt",
        {"user_id": user_id},
    )
