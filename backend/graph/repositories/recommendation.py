"""Recommendation graph repository."""

from __future__ import annotations

from typing import Any

from graph.entities.base import GraphEntity
from graph.queries.library import merge_relationship, persist_entity, recommendations_for_user
from graph.relationships import GraphRelationship
from graph.repositories.base import BaseGraphRepository


class RecommendationRepository(BaseGraphRepository):
    async def get_by_id(self, recommendation_id: str) -> dict[str, Any] | None:
        cypher = "MATCH (r:Recommendation {recommendation_id: $id}) RETURN r LIMIT 1"
        records, _ = await self._run(cypher, {"id": recommendation_id})
        if not records:
            return None
        return self._node_props(records[0], "r")

    async def save(self, entity: GraphEntity) -> dict[str, Any]:
        cypher, params = persist_entity(entity)
        records, _ = await self._write(cypher, params)
        return self._node_props(records[0]) if records else entity.to_neo4j_props()

    async def delete(self, recommendation_id: str) -> None:
        cypher = "MATCH (r:Recommendation {recommendation_id: $id}) DETACH DELETE r"
        await self._write(cypher, {"id": recommendation_id})

    async def for_user(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        cypher, params = recommendations_for_user(user_id, limit)
        records, _ = await self._run(cypher, params)
        return [self._node_props(r, "r") for r in records]

    async def link_to_session(self, session_id: str, recommendation_id: str) -> None:
        rel = GraphRelationship("RECOMMENDS", "session_id", session_id, "recommendation_id", recommendation_id)
        cypher, params = merge_relationship(rel, "Session", "Recommendation")
        await self._write(cypher, params)
