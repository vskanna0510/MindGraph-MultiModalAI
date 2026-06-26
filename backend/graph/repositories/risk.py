"""Risk score graph repository."""

from __future__ import annotations

from typing import Any

from graph.entities.base import GraphEntity
from graph.queries import merge_entity, merge_relationship, high_risk_users
from graph.relationships import GraphRelationship
from graph.repositories.base import BaseGraphRepository


class RiskRepository(BaseGraphRepository):
    async def get_by_id(self, node_id: str) -> dict[str, Any] | None:
        cypher = "MATCH (r:RiskScore {node_id: $node_id}) RETURN r LIMIT 1"
        records, _ = await self._run(cypher, {"node_id": node_id})
        if not records:
            return None
        return self._node_props(records[0], "r")

    async def save(self, entity: GraphEntity) -> dict[str, Any]:
        cypher, params = merge_entity(entity)
        records, _ = await self._write(cypher, params)
        return self._node_props(records[0]) if records else entity.to_neo4j_props()

    async def delete(self, node_id: str) -> None:
        cypher = "MATCH (r:RiskScore {node_id: $node_id}) DETACH DELETE r"
        await self._write(cypher, {"node_id": node_id})

    async def high_risk_sessions(self, threshold: float = 0.7) -> list[dict[str, Any]]:
        cypher, params = high_risk_users(threshold)
        records, _ = await self._run(cypher, params)
        return list(records)

    async def link_to_prediction(self, prediction_id: str, risk_node_id: str) -> None:
        rel = GraphRelationship("HAS_RISK", "prediction_id", prediction_id, "node_id", risk_node_id)
        cypher, params = merge_relationship(rel, "Prediction", "RiskScore")
        await self._write(cypher, params)
