"""Graph analytics Cypher queries."""

from __future__ import annotations

from typing import Any


def node_relationship_counts(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id}) "
        "OPTIONAL MATCH (u)-[*1..3]-(n) "
        "WITH u, collect(DISTINCT n) AS nodes "
        "UNWIND nodes AS node "
        "OPTIONAL MATCH (node)-[r]-() "
        "RETURN count(DISTINCT node) AS node_count, count(DISTINCT r) AS rel_count",
        {"user_id": user_id},
    )


def central_nodes(user_id: str, limit: int = 10) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[*1..2]-(n) "
        "WHERE NOT n:User "
        "WITH n, count(*) AS degree "
        "RETURN labels(n)[0] AS label, n.uuid AS uuid, degree "
        "ORDER BY degree DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )


def temporal_density(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "WITH count(s) AS sessions, "
        "min(s.timestamp) AS first_ts, max(s.timestamp) AS last_ts "
        "RETURN sessions, first_ts, last_ts, "
        "CASE WHEN sessions > 1 THEN toFloat(sessions) ELSE 0.0 END AS density",
        {"user_id": user_id},
    )


def connected_components_count(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "OPTIONAL MATCH path=(s)-[:TEMPORALLY_PRECEDES*]->(s2:Session) "
        "RETURN count(DISTINCT s) AS sessions, count(path) AS temporal_links",
        {"user_id": user_id},
    )


def high_risk_users_research(threshold: float = 0.7, limit: int = 100) -> tuple[str, dict[str, Any]]:
    """Research mode only — cross-user aggregate."""
    return (
        "MATCH (u:User)-[:HAS_SESSION]->(:Session)-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "WITH u.user_id AS user_id, max(coalesce(p.risk_probability, p.risk_score)) AS peak_risk "
        "WHERE peak_risk >= $threshold "
        "RETURN user_id, peak_risk ORDER BY peak_risk DESC LIMIT $limit",
        {"threshold": threshold, "limit": limit},
    )
