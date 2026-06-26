"""User graph Cypher queries — parameterized and injection-safe."""

from __future__ import annotations

from typing import Any


def get_user(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id}) WHERE u.deleted_at IS NULL RETURN u LIMIT 1",
        {"user_id": user_id},
    )


def create_user_props(props: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return (
        "CREATE (u:User) SET u = $props RETURN u",
        {"props": props},
    )


def update_user(user_id: str, props: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id}) SET u += $props, u.last_updated = $props.last_updated RETURN u",
        {"user_id": user_id, "props": props},
    )


def delete_user(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id}) DETACH DELETE u",
        {"user_id": user_id},
    )
