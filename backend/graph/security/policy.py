"""Graph security policies."""

from __future__ import annotations


class GraphSecurityPolicy:
    """Enforce owner-scoped access to graph nodes."""

    @staticmethod
    def can_read(requester_id: str, owner: str) -> bool:
        return requester_id == owner or requester_id == "system"

    @staticmethod
    def can_write(requester_id: str, owner: str) -> bool:
        return requester_id == owner

    @staticmethod
    def sanitize_props(props: dict) -> dict:
        blocked = {"password", "token", "secret", "raw_media", "media_bytes"}
        return {k: v for k, v in props.items() if k not in blocked}
