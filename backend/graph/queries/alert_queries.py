"""Graph alert Cypher queries."""

from __future__ import annotations

from typing import Any


def store_alert_props(props: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return (
        "CREATE (a:GraphAlert) SET a = $props RETURN a",
        {"props": props},
    )


def user_alerts(user_id: str, limit: int = 50) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_ALERT]->(a:GraphAlert) "
        "RETURN a ORDER BY a.created_at DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )


def link_alert_to_user(user_id: str, alert_uuid: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id}) "
        "MATCH (a:GraphAlert {uuid: $alert_uuid}) "
        "MERGE (u)-[:HAS_ALERT]->(a) RETURN a",
        {"user_id": user_id, "alert_uuid": alert_uuid},
    )
