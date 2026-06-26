"""Emotion graph repository."""

from __future__ import annotations

from typing import Any

from graph.entities.base import GraphEntity
from graph.queries import emotion_evolution, merge_relationship, persist_entity
from graph.relationships import GraphRelationship
from graph.repositories.base import BaseGraphRepository


class EmotionRepository(BaseGraphRepository):
    async def get_by_id(self, uuid: str) -> dict[str, Any] | None:
        cypher = "MATCH (e:Emotion {uuid: $uuid}) RETURN e LIMIT 1"
        records, _ = await self._run(cypher, {"uuid": uuid})
        if not records:
            return None
        return self._node_props(records[0], "e")

    async def save(self, entity: GraphEntity) -> dict[str, Any]:
        cypher, params = persist_entity(entity)
        records, _ = await self._write(cypher, params)
        return self._node_props(records[0]) if records else entity.to_neo4j_props()

    async def delete(self, uuid: str) -> None:
        cypher = "MATCH (e:Emotion {uuid: $uuid}) DETACH DELETE e"
        await self._write(cypher, {"uuid": uuid})

    async def evolution(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = emotion_evolution(user_id)
        records, _ = await self._run(cypher, params)
        return list(records)

    async def link_to_session(self, session_id: str, emotion_uuid: str) -> None:
        rel = GraphRelationship("EXPRESSES", "session_id", session_id, "uuid", emotion_uuid)
        cypher, params = merge_relationship(rel, "Session", "Emotion")
        await self._write(cypher, params)
