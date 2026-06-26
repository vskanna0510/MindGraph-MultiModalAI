"""User graph repository."""

from __future__ import annotations

from typing import Any

from graph.entities.base import GraphEntity
from graph.queries import merge_entity, user_timeline
from graph.repositories.base import BaseGraphRepository


class UserRepository(BaseGraphRepository):
    async def get_by_id(self, user_id: str) -> dict[str, Any] | None:
        cypher = "MATCH (u:User {user_id: $user_id}) RETURN u LIMIT 1"
        records, _ = await self._run(cypher, {"user_id": user_id})
        if not records:
            return None
        return self._node_props(records[0], "u")

    async def save(self, entity: GraphEntity) -> dict[str, Any]:
        cypher, params = merge_entity(entity)
        records, _ = await self._write(cypher, params)
        return self._node_props(records[0]) if records else entity.to_neo4j_props()

    async def delete(self, user_id: str) -> None:
        cypher = "MATCH (u:User {user_id: $user_id}) DETACH DELETE u"
        await self._write(cypher, {"user_id": user_id})

    async def timeline(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = user_timeline(user_id)
        records, _ = await self._run(cypher, params)
        return [self._node_props(r, "s") for r in records]
