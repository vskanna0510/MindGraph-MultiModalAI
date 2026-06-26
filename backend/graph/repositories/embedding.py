"""Embedding graph repository."""

from __future__ import annotations

from typing import Any

from graph.entities.base import GraphEntity
from graph.queries import merge_entity, merge_relationship
from graph.relationships import GraphRelationship
from graph.repositories.base import BaseGraphRepository


class EmbeddingRepository(BaseGraphRepository):
    async def get_by_id(self, embedding_id: str) -> dict[str, Any] | None:
        cypher = "MATCH (e:Embedding {embedding_id: $embedding_id}) RETURN e LIMIT 1"
        records, _ = await self._run(cypher, {"embedding_id": embedding_id})
        if not records:
            return None
        return self._node_props(records[0], "e")

    async def save(self, entity: GraphEntity) -> dict[str, Any]:
        cypher, params = merge_entity(entity)
        records, _ = await self._write(cypher, params)
        return self._node_props(records[0]) if records else entity.to_neo4j_props()

    async def delete(self, embedding_id: str) -> None:
        cypher = "MATCH (e:Embedding {embedding_id: $embedding_id}) DETACH DELETE e"
        await self._write(cypher, {"embedding_id": embedding_id})

    async def link_to_session(self, session_id: str, embedding_id: str) -> None:
        rel = GraphRelationship("HAS_EMBEDDING", "session_id", session_id, "embedding_id", embedding_id)
        cypher, params = merge_relationship(rel, "Session", "Embedding")
        await self._write(cypher, params)
