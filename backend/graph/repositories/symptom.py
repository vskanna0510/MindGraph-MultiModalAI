"""Symptom graph repository."""

from __future__ import annotations

from typing import Any

from graph.entities.base import GraphEntity
from graph.queries.library import persist_entity, symptom_evolution
from graph.relationships import GraphRelationship
from graph.repositories.base import BaseGraphRepository


class SymptomRepository(BaseGraphRepository):
    async def get_by_id(self, uuid: str) -> dict[str, Any] | None:
        cypher = "MATCH (s:Symptom {uuid: $uuid}) RETURN s LIMIT 1"
        records, _ = await self._run(cypher, {"uuid": uuid})
        if not records:
            return None
        return self._node_props(records[0], "s")

    async def save(self, entity: GraphEntity) -> dict[str, Any]:
        cypher, params = persist_entity(entity)
        records, _ = await self._write(cypher, params)
        return self._node_props(records[0]) if records else entity.to_neo4j_props()

    async def delete(self, uuid: str) -> None:
        cypher = "MATCH (s:Symptom {uuid: $uuid}) DETACH DELETE s"
        await self._write(cypher, {"uuid": uuid})

    async def evolution(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = symptom_evolution(user_id)
        records, _ = await self._run(cypher, params)
        return list(records)

    async def link_to_session(self, session_id: str, symptom_uuid: str) -> None:
        rel = GraphRelationship("INDICATES", "session_id", session_id, "uuid", symptom_uuid)
        cypher, params = merge_relationship(rel, "Session", "Symptom")
        await self._write(cypher, params)
