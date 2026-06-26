"""Prediction graph repository."""

from __future__ import annotations

from typing import Any

from graph.entities.base import GraphEntity
from graph.queries.library import merge_relationship, persist_entity, risk_progression
from graph.relationships import GraphRelationship
from graph.repositories.base import BaseGraphRepository


class PredictionRepository(BaseGraphRepository):
    async def get_by_id(self, prediction_id: str) -> dict[str, Any] | None:
        cypher = "MATCH (p:Prediction {prediction_id: $prediction_id}) RETURN p LIMIT 1"
        records, _ = await self._run(cypher, {"prediction_id": prediction_id})
        if not records:
            return None
        return self._node_props(records[0], "p")

    async def save(self, entity: GraphEntity) -> dict[str, Any]:
        cypher, params = persist_entity(entity)
        records, _ = await self._write(cypher, params)
        return self._node_props(records[0]) if records else entity.to_neo4j_props()

    async def delete(self, prediction_id: str) -> None:
        cypher = "MATCH (p:Prediction {prediction_id: $prediction_id}) DETACH DELETE p"
        await self._write(cypher, {"prediction_id": prediction_id})

    async def risk_progression(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = risk_progression(user_id)
        records, _ = await self._run(cypher, params)
        return list(records)

    async def link_to_session(self, session_id: str, prediction_id: str) -> None:
        for rel_type in ("PRODUCED", "PREDICTED"):
            rel = GraphRelationship(rel_type, "session_id", session_id, "prediction_id", prediction_id)
            cypher, params = merge_relationship(rel, "Session", "Prediction")
            await self._write(cypher, params)
