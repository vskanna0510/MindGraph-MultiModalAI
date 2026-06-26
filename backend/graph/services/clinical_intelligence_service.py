"""Clinical intelligence orchestrator — reasoning + analytics + alerts."""

from __future__ import annotations

import uuid
from typing import Any

from neo4j import AsyncSession

from graph.analytics import GraphMetricsEngine, MaterializedViewStore
from graph.cache import GraphCache
from graph.config.loader import get_graph_config
from graph.contracts import GraphServiceResponse, TimedOperation
from graph.queries import analytics_queries as aq
from graph.queries import trend_queries as tq
from graph.reasoning.pipeline import GraphReasoningPipeline
from graph.reasoning.retrieval import GraphRetrievalEngine, TemporalWindow
from graph.repositories.base import BaseGraphRepository


class ClinicalIntelligenceService:
    """Top-level clinical graph intelligence API."""

    def __init__(self, session: AsyncSession, *, cache: GraphCache | None = None) -> None:
        self._session = session
        self._repo = BaseGraphRepository(session)
        self._cache = cache or GraphCache()
        self._pipeline = GraphReasoningPipeline(session, cache=self._cache)
        self._retrieval = GraphRetrievalEngine(session, cache=self._cache)
        self._views = MaterializedViewStore()
        self._config = get_graph_config()

    async def full_reasoning(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        window: str = "last_30_days",
        correlation_id: str | None = None,
    ) -> GraphServiceResponse[dict[str, Any]]:
        tw = TemporalWindow(window)
        return await self._pipeline.reason(user_id, session_id=session_id, window=tw, correlation_id=correlation_id)

    async def risk_analytics(self, user_id: str, *, correlation_id: str | None = None) -> GraphServiceResponse[dict[str, Any]]:
        cid = correlation_id or str(uuid.uuid4())
        cache_key = f"analytics:risk:{user_id}"
        cached = self._cache.get(cache_key)
        if cached:
            return GraphServiceResponse(result=cached, correlation_id=cid, cache_hit=True)

        with TimedOperation() as timer:
            cypher, params = tq.risk_trend(user_id)
            records, _ = await self._repo._run(cypher, params)
            risks = [float(r) for r in (records[0].get("risks", []) if records else [])]

            from graph.reasoning.risk_trend import RiskTrendEngine
            from graph.reasoning.temporal_reasoning import TemporalReasoningEngine

            metrics = RiskTrendEngine.to_dict(RiskTrendEngine.compute(risks))
            temporal = TemporalReasoningEngine().analyze_risk(risks)

            from graph.queries import risk_queries as rq

            cypher_a, params_a = rq.average_risk(user_id)
            avg_records, _ = await self._repo._run(cypher_a, params_a)

            result = {
                "user_id": user_id,
                "metrics": metrics,
                "temporal_state": temporal,
                "db_average": avg_records[0] if avg_records else {},
            }
            self._views.risk_timeline_view(user_id, [{"risk": r} for r in risks])

        self._cache.set(cache_key, result)
        return GraphServiceResponse(
            result=result,
            execution_time_ms=timer.elapsed_ms,
            correlation_id=cid,
            confidence=temporal.get("confidence", 0.5),
        )

    async def graph_analytics(self, user_id: str, *, correlation_id: str | None = None) -> GraphServiceResponse[dict[str, Any]]:
        cid = correlation_id or str(uuid.uuid4())
        with TimedOperation() as timer:
            cypher, params = aq.node_relationship_counts(user_id)
            counts, _ = await self._repo._run(cypher, params)
            row = counts[0] if counts else {}

            cypher_c, params_c = aq.central_nodes(user_id)
            central, _ = await self._repo._run(cypher_c, params_c)

            cypher_d, params_d = aq.temporal_density(user_id)
            density, _ = await self._repo._run(cypher_d, params_d)

            node_count = int(row.get("node_count", 0))
            rel_count = int(row.get("rel_count", 0))
            degree_map = {str(r.get("uuid", i)): int(r.get("degree", 0)) for i, r in enumerate(central)}

            metrics = GraphMetricsEngine.from_degree_map(degree_map, node_count, rel_count)
            importance = GraphMetricsEngine.node_importance(central)

            result = {
                "user_id": user_id,
                "metrics": metrics,
                "node_importance": importance,
                "temporal_density": density[0] if density else {},
            }

        return GraphServiceResponse(
            result=result,
            execution_time_ms=timer.elapsed_ms,
            correlation_id=cid,
            metadata={"targets_ms": self._config.reasoning.get("performance_targets_ms", {})},
        )

    async def emotion_analytics(self, user_id: str) -> GraphServiceResponse[dict[str, Any]]:
        cid = str(uuid.uuid4())
        with TimedOperation() as timer:
            cypher, params = tq.emotion_trend(user_id)
            records, _ = await self._repo._run(cypher, params)
            cooccurrence = GraphMetricsEngine.emotion_cooccurrence(records)
            from graph.reasoning.temporal_reasoning import TemporalReasoningEngine
            probs = [float(r.get("prob", 0)) for r in records]
            stability = TemporalReasoningEngine().analyze_emotion_stability(probs)
            self._views.emotion_timeline_view(user_id, list(records))
            result = {"timeline": list(records), "cooccurrence": cooccurrence, "stability": stability}

        return GraphServiceResponse(result=result, execution_time_ms=timer.elapsed_ms, correlation_id=cid)

    async def get_materialized_view(self, user_id: str, view_type: str) -> GraphServiceResponse[dict[str, Any] | None]:
        view = self._views.get_latest(user_id, view_type)
        return GraphServiceResponse(
            result=view.to_dict() if view else None,
            correlation_id=str(uuid.uuid4()),
        )
