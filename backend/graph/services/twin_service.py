"""Digital Twin and Graph Intelligence service."""

from __future__ import annotations

import uuid
from typing import Any

from neo4j import AsyncSession

from graph.cache import GraphCache
from graph.config.loader import get_graph_config
from graph.contracts import GraphServiceResponse, TimedOperation
from graph.evaluation import ResearchEvaluator
from graph.explainability import GraphExplainabilityEngine
from graph.export.formats import GraphExportEngine
from graph.recommendation.prioritizer import RecommendationPrioritizer
from graph.twin import GraphIntelligenceEngine


class TwinService:
    """Dashboard API for Digital Cognitive Twin."""

    def __init__(self, session: AsyncSession, *, cache: GraphCache | None = None) -> None:
        self._engine = GraphIntelligenceEngine(session, cache=cache)
        self._prioritizer = RecommendationPrioritizer()
        self._config = get_graph_config()

    async def twin_summary(self, user_id: str, *, requester_id: str | None = None) -> GraphServiceResponse[dict[str, Any]]:
        req = requester_id or user_id
        return await self._engine.get_twin_summary(user_id, requester_id=req)

    async def personalized_recommendations(
        self,
        user_id: str,
        *,
        requester_id: str | None = None,
    ) -> GraphServiceResponse[dict[str, Any]]:
        req = requester_id or user_id
        with TimedOperation() as timer:
            summary = await self._engine.get_twin_summary(user_id, requester_id=req)
            twin_data = summary.result["twin"]
            risk = summary.result["risk_evolution"]
            current_risk = float(risk.get("current_risk", 0.5))

            twin_obj = self._engine._twins.get(user_id)
            recs = self._prioritizer.generate(
                current_risk=current_risk,
                twin_summary=twin_data,
                compliance_history=twin_data.get("recovery_profile", {}).get("recommendation_compliance", 0.5),
                language=twin_data.get("language_profile", {}).get("preferred_language", "en"),
                previous_types=[r.get("type", "") for r in summary.result.get("recommendation_timeline", [])],
            )

            explanations = []
            if twin_obj:
                for rec in recs[:3]:
                    explanations.append(
                        GraphExplainabilityEngine.explain_recommendation(rec, twin_obj)
                    )

        return GraphServiceResponse(
            result={
                "recommendations": recs,
                "explanations": explanations,
                "current_risk": current_risk,
                "guidance_only": True,
            },
            execution_time_ms=timer.elapsed_ms,
            correlation_id=summary.correlation_id,
            confidence=risk.get("prediction_confidence", 0.5),
        )

    async def dashboard(self, user_id: str, *, requester_id: str | None = None) -> GraphServiceResponse[dict[str, Any]]:
        req = requester_id or user_id
        timelines = await self._engine.dashboard_timelines(user_id, requester_id=req)
        summary = await self._engine.get_twin_summary(user_id, requester_id=req)
        return GraphServiceResponse(
            result={
                "timelines": timelines.result,
                "twin_summary": summary.result,
            },
            execution_time_ms=timelines.execution_time_ms + summary.execution_time_ms,
            correlation_id=str(uuid.uuid4()),
        )

    async def export_twin(
        self,
        user_id: str,
        fmt: str = "json",
        *,
        requester_id: str | None = None,
    ) -> GraphServiceResponse[dict[str, Any]]:
        req = requester_id or user_id
        update = await self._engine.update_twin(user_id, requester_id=req)
        twin_data = update.result["twin"]
        exported = GraphExportEngine.export_twin(twin_data, fmt)
        return GraphServiceResponse(
            result={"format": fmt, "data": exported},
            correlation_id=update.correlation_id,
        )

    async def research_evaluation(self, user_id: str, *, requester_id: str | None = None) -> GraphServiceResponse[dict[str, Any]]:
        req = requester_id or user_id
        summary = await self._engine.get_twin_summary(user_id, requester_id=req)
        twin = self._engine._twins.get(user_id)
        if not twin:
            return GraphServiceResponse(result={"error": "Twin not built"}, correlation_id=str(uuid.uuid4()))
        evaluation = ResearchEvaluator.evaluate(
            twin=twin,
            evolution_score=summary.result["evolution_score"],
            reasoning_latency_ms=summary.execution_time_ms,
            recommendation_count=len(summary.result.get("recommendation_timeline", [])),
        )
        return GraphServiceResponse(result=evaluation, correlation_id=summary.correlation_id)

    async def snapshot(self, user_id: str, granularity: str = "daily", *, requester_id: str | None = None) -> GraphServiceResponse[dict[str, Any]]:
        req = requester_id or user_id
        return await self._engine.capture_snapshot(user_id, requester_id=req, granularity=granularity)
