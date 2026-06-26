"""Session graph repository."""

from __future__ import annotations

from typing import Any

from graph.entities.base import GraphEntity
from graph.queries import merge_entity, merge_relationship, session_history
from graph.relationships import GraphRelationship
from graph.repositories.base import BaseGraphRepository


class SessionRepository(BaseGraphRepository):
    async def get_by_id(self, session_id: str) -> dict[str, Any] | None:
        cypher = "MATCH (s:Session {session_id: $session_id}) RETURN s LIMIT 1"
        records, _ = await self._run(cypher, {"session_id": session_id})
        if not records:
            return None
        return self._node_props(records[0], "s")

    async def save(self, entity: GraphEntity) -> dict[str, Any]:
        cypher, params = merge_entity(entity)
        records, _ = await self._write(cypher, params)
        return self._node_props(records[0]) if records else entity.to_neo4j_props()

    async def delete(self, session_id: str) -> None:
        cypher = "MATCH (s:Session {session_id: $session_id}) DETACH DELETE s"
        await self._write(cypher, {"session_id": session_id})

    async def history(self, user_id: str, limit: int = 12) -> list[dict[str, Any]]:
        cypher, params = session_history(user_id, limit)
        records, _ = await self._run(cypher, params)
        return [self._node_props(r, "s") for r in records]

    async def link_to_user(self, user_id: str, session_id: str) -> None:
        rel = GraphRelationship("HAS_SESSION", "user_id", user_id, "session_id", session_id)
        cypher, params = merge_relationship(rel, "User", "Session")
        await self._write(cypher, params)

    async def link_temporal(self, earlier_session_id: str, later_session_id: str) -> None:
        rel = GraphRelationship(
            "TEMPORALLY_PRECEDES",
            "session_id",
            earlier_session_id,
            "session_id",
            later_session_id,
        )
        cypher, params = merge_relationship(rel, "Session", "Session")
        await self._write(cypher, params)
