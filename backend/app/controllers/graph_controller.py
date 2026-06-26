"""Graph API controller."""

from __future__ import annotations

from neo4j import AsyncSession

from app.schemas.common import ApiResponse
from app.schemas.graph import (
    ClinicalTrendsResponse,
    ExplainResponse,
    GraphStatisticsResponse,
    LongitudinalReportResponse,
    MemoryRecordRequest,
    PredictionUpsertRequest,
    RiskProgressionResponse,
    SessionUpsertRequest,
    TimelineResponse,
)
from graph.services import (
    AnalyticsService,
    ClinicalIntelligenceService,
    ExplainabilityService,
    GraphService,
    LongitudinalService,
    MemoryService,
    RecommendationService,
    TemporalService,
    TwinService,
)


class GraphController:
    """HTTP adapter for graph endpoints."""

    def __init__(self, session: AsyncSession) -> None:
        self._graph = GraphService(session)
        self._temporal = TemporalService(session)
        self._analytics = AnalyticsService(session)
        self._explain = ExplainabilityService(session)
        self._recommendations = RecommendationService(session)
        self._memory = MemoryService(session)
        self._longitudinal = LongitudinalService(session)
        self._clinical = ClinicalIntelligenceService(session)
        self._twin = TwinService(session)

    async def twin_summary(self, user_id: str) -> ApiResponse[dict]:
        response = await self._twin.twin_summary(user_id)
        return ApiResponse.success(message="Digital twin summary", data=response.to_dict())

    async def twin_dashboard(self, user_id: str) -> ApiResponse[dict]:
        response = await self._twin.dashboard(user_id)
        return ApiResponse.success(message="Twin dashboard", data=response.to_dict())

    async def twin_recommendations(self, user_id: str) -> ApiResponse[dict]:
        response = await self._twin.personalized_recommendations(user_id)
        return ApiResponse.success(message="Personalized recommendations", data=response.to_dict())

    async def twin_export(self, user_id: str, fmt: str = "json") -> ApiResponse[dict]:
        response = await self._twin.export_twin(user_id, fmt)
        return ApiResponse.success(message="Twin exported", data=response.to_dict())

    async def twin_evaluation(self, user_id: str) -> ApiResponse[dict]:
        response = await self._twin.research_evaluation(user_id)
        return ApiResponse.success(message="Research evaluation", data=response.to_dict())

    async def twin_snapshot(self, user_id: str, granularity: str = "daily") -> ApiResponse[dict]:
        response = await self._twin.snapshot(user_id, granularity)
        return ApiResponse.success(message="Twin snapshot captured", data=response.to_dict())

    async def get_user(self, user_id: str) -> ApiResponse[dict]:
        data = await self._graph.get_user(user_id)
        return ApiResponse.success(message="User retrieved", data=data)

    async def timeline(self, user_id: str) -> ApiResponse[TimelineResponse]:
        sessions = await self._temporal.timeline(user_id)
        return ApiResponse.success(
            message="Timeline retrieved",
            data=TimelineResponse(user_id=user_id, sessions=sessions),
        )

    async def risk_progression(self, user_id: str) -> ApiResponse[RiskProgressionResponse]:
        points = await self._analytics.risk_progression(user_id)
        return ApiResponse.success(
            message="Risk progression retrieved",
            data=RiskProgressionResponse(user_id=user_id, points=points),
        )

    async def clinical_trends(self, user_id: str) -> ApiResponse[ClinicalTrendsResponse]:
        trends = await self._analytics.clinical_trends(user_id)
        return ApiResponse.success(
            message="Clinical trends retrieved",
            data=ClinicalTrendsResponse(
                user_id=user_id,
                trend_available=trends["trend_available"],
                risk=trends["risk"],
                emotions=trends["emotions"],
                symptoms=trends["symptoms"],
            ),
        )

    async def explain(self, user_id: str) -> ApiResponse[ExplainResponse]:
        result = await self._explain.explain_user_state(user_id)
        return ApiResponse.success(
            message="Explanation generated",
            data=ExplainResponse(
                user_id=user_id,
                reasoning_path=result["reasoning_path"],
                explanation=result["explanation"],
                latest_risk=result.get("latest_risk"),
            ),
        )

    async def statistics(self) -> ApiResponse[GraphStatisticsResponse]:
        stats = await self._graph.statistics()
        return ApiResponse.success(
            message="Graph statistics retrieved",
            data=GraphStatisticsResponse(
                nodes=stats["nodes"],
                edges=stats["edges"],
                execution_time_ms=stats.get("execution_time_ms", 0.0),
            ),
        )

    async def recommendations(self, user_id: str) -> ApiResponse[list]:
        recs = await self._recommendations.for_user(user_id)
        return ApiResponse.success(message="Recommendations retrieved", data=recs)

    async def upsert_session(self, user_id: str, body: SessionUpsertRequest) -> ApiResponse[dict]:
        saved = await self._graph.upsert_session(
            user_id,
            body.session_id,
            owner=body.owner,
            timestamp=body.timestamp,
            quality_score=body.quality_score,
            dataset=body.dataset,
            link_previous=body.previous_session_id,
        )
        return ApiResponse.success(message="Session upserted", data=saved)

    async def upsert_prediction(self, body: PredictionUpsertRequest) -> ApiResponse[dict]:
        saved = await self._graph.upsert_prediction(
            body.session_id,
            owner=body.owner,
            risk_score=body.risk_score,
            confidence=body.confidence,
            prediction_label=body.prediction_label,
            model_version=body.model_version,
        )
        return ApiResponse.success(message="Prediction appended", data=saved)

    async def record_memory(self, user_id: str, body: MemoryRecordRequest) -> ApiResponse[dict]:
        result = await self._memory.record_session_memory(
            user_id,
            body.session_id,
            owner=body.owner,
            previous_session_id=body.previous_session_id,
            risk_probability=body.risk_probability,
            confidence=body.confidence,
            binary_prediction=body.binary_prediction,
            emotions=body.emotions,
            symptoms=body.symptoms,
            behaviours=body.behaviours,
            recommendations=body.recommendations,
        )
        return ApiResponse.success(
            message="Memory recorded",
            data={
                "user_id": result.user_id,
                "session_id": result.session_id,
                "prediction_id": result.prediction_id,
                "risk_id": result.risk_id,
                "emotions": result.emotions,
                "symptoms": result.symptoms,
                "behaviours": result.behaviours,
                "recommendations": result.recommendations,
            },
        )

    async def longitudinal_report(self, user_id: str) -> ApiResponse[LongitudinalReportResponse]:
        report = await self._longitudinal.full_longitudinal_report(user_id)
        return ApiResponse.success(
            message="Longitudinal report generated",
            data=LongitudinalReportResponse(**report),
        )

    async def snapshot(self, user_id: str, granularity: str = "daily") -> ApiResponse[dict]:
        from graph.temporal.snapshots import SnapshotGranularity

        gran = SnapshotGranularity(granularity)
        data = await self._memory.snapshot(user_id, gran)
        return ApiResponse.success(message="Snapshot captured", data=data)

    async def reasoning(self, user_id: str, session_id: str | None = None, window: str = "last_30_days") -> ApiResponse[dict]:
        response = await self._clinical.full_reasoning(user_id, session_id=session_id, window=window)
        return ApiResponse.success(message="Reasoning complete", data=response.to_dict())

    async def risk_analytics(self, user_id: str) -> ApiResponse[dict]:
        response = await self._clinical.risk_analytics(user_id)
        return ApiResponse.success(message="Risk analytics computed", data=response.to_dict())

    async def graph_analytics(self, user_id: str) -> ApiResponse[dict]:
        response = await self._clinical.graph_analytics(user_id)
        return ApiResponse.success(message="Graph analytics computed", data=response.to_dict())

    async def emotion_analytics(self, user_id: str) -> ApiResponse[dict]:
        response = await self._clinical.emotion_analytics(user_id)
        return ApiResponse.success(message="Emotion analytics computed", data=response.to_dict())
