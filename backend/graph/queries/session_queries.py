"""Session graph Cypher queries."""

from __future__ import annotations

from typing import Any


def create_session_props(props: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return ("CREATE (s:Session) SET s = $props RETURN s", {"props": props})


def latest_session(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "WHERE s.deleted_at IS NULL "
        "RETURN s ORDER BY s.timestamp DESC LIMIT 1",
        {"user_id": user_id},
    )


def session_timeline(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "WHERE s.deleted_at IS NULL "
        "RETURN s.session_id AS session_id, s.timestamp AS timestamp, "
        "s.quality_score AS quality ORDER BY s.timestamp ASC",
        {"user_id": user_id},
    )


def session_history(user_id: str, limit: int = 12) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "WHERE s.deleted_at IS NULL "
        "RETURN s ORDER BY s.timestamp DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )


def sessions_in_window(user_id: str, since_iso: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "WHERE s.deleted_at IS NULL AND s.timestamp >= $since "
        "RETURN s ORDER BY s.timestamp ASC",
        {"user_id": user_id, "since": since_iso},
    )
