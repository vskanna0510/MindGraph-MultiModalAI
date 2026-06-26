"""Knowledge graph API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from neo4j import AsyncSession

from app.controllers.graph_controller import GraphController
from app.schemas.common import ApiResponse
from app.schemas.graph import (
    ClinicalTrendsResponse,
    ExplainResponse,
    GraphStatisticsResponse,
    MemoryRecordRequest,
    PredictionUpsertRequest,
    RiskProgressionResponse,
    SessionUpsertRequest,
    TimelineResponse,
    LongitudinalReportResponse,
)
from graph.connection import get_neo4j_session

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])


def _controller(session: Annotated[AsyncSession, Depends(get_neo4j_session)]) -> GraphController:
    return GraphController(session)


@router.get("/statistics", response_model=ApiResponse[GraphStatisticsResponse])
async def graph_statistics(ctrl: Annotated[GraphController, Depends(_controller)]) -> ApiResponse[GraphStatisticsResponse]:
    return await ctrl.statistics()


@router.get("/users/{user_id}", response_model=ApiResponse[dict])
async def get_user(user_id: str, ctrl: Annotated[GraphController, Depends(_controller)]) -> ApiResponse[dict]:
    return await ctrl.get_user(user_id)


@router.get("/users/{user_id}/timeline", response_model=ApiResponse[TimelineResponse])
async def user_timeline(user_id: str, ctrl: Annotated[GraphController, Depends(_controller)]) -> ApiResponse[TimelineResponse]:
    return await ctrl.timeline(user_id)


@router.get("/users/{user_id}/risk", response_model=ApiResponse[RiskProgressionResponse])
async def risk_progression(user_id: str, ctrl: Annotated[GraphController, Depends(_controller)]) -> ApiResponse[RiskProgressionResponse]:
    return await ctrl.risk_progression(user_id)


@router.get("/users/{user_id}/trends", response_model=ApiResponse[ClinicalTrendsResponse])
async def clinical_trends(user_id: str, ctrl: Annotated[GraphController, Depends(_controller)]) -> ApiResponse[ClinicalTrendsResponse]:
    return await ctrl.clinical_trends(user_id)


@router.get("/users/{user_id}/explain", response_model=ApiResponse[ExplainResponse])
async def explain_user(user_id: str, ctrl: Annotated[GraphController, Depends(_controller)]) -> ApiResponse[ExplainResponse]:
    return await ctrl.explain(user_id)


@router.get("/users/{user_id}/recommendations", response_model=ApiResponse[list])
async def user_recommendations(user_id: str, ctrl: Annotated[GraphController, Depends(_controller)]) -> ApiResponse[list]:
    return await ctrl.recommendations(user_id)


@router.post("/users/{user_id}/sessions", response_model=ApiResponse[dict])
async def create_session(
    user_id: str,
    body: SessionUpsertRequest,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.upsert_session(user_id, body)


@router.post("/predictions", response_model=ApiResponse[dict])
async def create_prediction(
    body: PredictionUpsertRequest,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.upsert_prediction(body)


@router.post("/users/{user_id}/memory", response_model=ApiResponse[dict])
async def record_memory(
    user_id: str,
    body: MemoryRecordRequest,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.record_memory(user_id, body)


@router.get("/users/{user_id}/longitudinal", response_model=ApiResponse[LongitudinalReportResponse])
async def longitudinal_report(
    user_id: str,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[LongitudinalReportResponse]:
    return await ctrl.longitudinal_report(user_id)


@router.post("/users/{user_id}/snapshots", response_model=ApiResponse[dict])
async def capture_snapshot(
    user_id: str,
    granularity: str = "daily",
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.snapshot(user_id, granularity)


@router.post("/users/{user_id}/reasoning", response_model=ApiResponse[dict])
async def run_reasoning(
    user_id: str,
    session_id: str | None = None,
    window: str = "last_30_days",
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.reasoning(user_id, session_id, window)


@router.get("/users/{user_id}/analytics/risk", response_model=ApiResponse[dict])
async def risk_analytics(
    user_id: str,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.risk_analytics(user_id)


@router.get("/users/{user_id}/analytics/graph", response_model=ApiResponse[dict])
async def graph_analytics(
    user_id: str,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.graph_analytics(user_id)


@router.get("/users/{user_id}/analytics/emotions", response_model=ApiResponse[dict])
async def emotion_analytics(
    user_id: str,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.emotion_analytics(user_id)


@router.get("/users/{user_id}/twin", response_model=ApiResponse[dict])
async def twin_summary(
    user_id: str,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.twin_summary(user_id)


@router.get("/users/{user_id}/twin/dashboard", response_model=ApiResponse[dict])
async def twin_dashboard(
    user_id: str,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.twin_dashboard(user_id)


@router.get("/users/{user_id}/twin/recommendations", response_model=ApiResponse[dict])
async def twin_recommendations(
    user_id: str,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.twin_recommendations(user_id)


@router.get("/users/{user_id}/twin/export", response_model=ApiResponse[dict])
async def twin_export(
    user_id: str,
    fmt: str = "json",
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.twin_export(user_id, fmt)


@router.get("/users/{user_id}/twin/evaluation", response_model=ApiResponse[dict])
async def twin_evaluation(
    user_id: str,
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.twin_evaluation(user_id)


@router.post("/users/{user_id}/twin/snapshots", response_model=ApiResponse[dict])
async def twin_snapshot(
    user_id: str,
    granularity: str = "daily",
    ctrl: Annotated[GraphController, Depends(_controller)],
) -> ApiResponse[dict]:
    return await ctrl.twin_snapshot(user_id, granularity)
