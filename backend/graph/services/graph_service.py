"""Core graph service — business logic over repositories."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession

from app.exceptions.base import GraphError, ValidationError
from graph.audit import AuditService
from graph.cache import GraphCache
from graph.config.loader import get_graph_config
from graph.entities.base import GraphEntity
from graph.entities.factories import (
    PredictionFactory,
    SessionFactory,
    UserFactory,
)
from graph.queries import graph_statistics, subgraph_extraction
from graph.repositories import (
    PredictionRepository,
    SessionRepository,
    UserRepository,
)
from graph.validators import GraphValidator


class GraphService:
    """Orchestrates graph mutations and reads. Never writes Cypher directly."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        cache: GraphCache | None = None,
        audit: AuditService | None = None,
    ) -> None:
        self._session = session
        self._users = UserRepository(session)
        self._sessions = SessionRepository(session)
        self._predictions = PredictionRepository(session)
        self._validator = GraphValidator()
        self._cache = cache or GraphCache()
        self._audit = audit or AuditService()
        self._config = get_graph_config()

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        cache_key = f"graph:user:{user_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        user = await self._users.get_by_id(user_id)
        if user:
            self._cache.set(cache_key, user)
        return user

    async def upsert_user(self, user_id: str, *, owner: str, **props: Any) -> dict[str, Any]:
        entity = UserFactory.create(user_id, owner=owner, **props)
        result = self._validator.validate_entity(entity)
        if not result.valid:
            raise ValidationError("; ".join(result.errors))
        saved = await self._users.save(entity)
        self._cache.invalidate_user(user_id)
        self._audit.log_mutation("upsert_user", entity_type="User", entity_id=user_id)
        return saved

    async def upsert_session(
        self,
        user_id: str,
        session_id: str,
        *,
        owner: str,
        link_previous: str | None = None,
        **props: Any,
    ) -> dict[str, Any]:
        session_entity = SessionFactory.create(session_id, owner=owner, **props)
        result = self._validator.validate_entity(session_entity)
        if not result.valid:
            raise ValidationError("; ".join(result.errors))

        user = await self._users.get_by_id(user_id)
        if not user:
            await self.upsert_user(user_id, owner=owner)

        saved = await self._sessions.save(session_entity)
        await self._sessions.link_to_user(user_id, session_id)

        if link_previous:
            await self._sessions.link_temporal(link_previous, session_id)

        self._cache.invalidate_user(user_id)
        self._cache.invalidate_session(session_id)
        self._audit.log_mutation("upsert_session", entity_type="Session", entity_id=session_id)
        return saved

    async def upsert_prediction(
        self,
        session_id: str,
        *,
        owner: str,
        risk_score: float,
        confidence: float,
        prediction_label: str,
        **props: Any,
    ) -> dict[str, Any]:
        binary = 1 if str(prediction_label).lower() in {"high", "1", "depressed", "positive"} else 0
        entity = PredictionFactory.create(
            owner=owner,
            risk_probability=risk_score,
            confidence=confidence,
            binary_prediction=binary,
            prediction_label=prediction_label,
            **props,
        )
        result = self._validator.validate_entity(entity)
        if not result.valid:
            raise ValidationError("; ".join(result.errors))

        prediction_id = entity.properties["prediction_id"]
        saved = await self._predictions.save(entity)
        await self._predictions.link_to_session(session_id, prediction_id)
        self._cache.invalidate_session(session_id)
        self._audit.log_mutation("append_prediction", entity_type="Prediction", entity_id=prediction_id)
        return saved

    async def user_timeline(self, user_id: str) -> list[dict[str, Any]]:
        cache_key = f"graph:temporal:{user_id}:timeline"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        timeline = await self._users.timeline(user_id)
        self._cache.set(cache_key, timeline)
        return timeline

    async def statistics(self) -> dict[str, Any]:
        cypher, params = graph_statistics()
        records, meta = await self._users._run(cypher, params)
        if not records:
            return {"nodes": 0, "edges": 0, "execution_time_ms": meta.execution_time_ms}
        row = records[0]
        return {
            "nodes": row.get("nodes", 0),
            "edges": row.get("edges", 0),
            "execution_time_ms": meta.execution_time_ms,
        }

    async def subgraph(self, node_uuid: str, depth: int | None = None) -> list[dict[str, Any]]:
        depth = min(depth or 2, self._config.max_subgraph_depth)
        cypher, params = subgraph_extraction(node_uuid, depth)
        records, meta = await self._users._run(cypher, params)
        if len(records) > self._config.max_nodes_per_query:
            raise GraphError(f"Subgraph exceeds max nodes ({self._config.max_nodes_per_query})")
        return {"paths": len(records), "execution_time_ms": meta.execution_time_ms}
