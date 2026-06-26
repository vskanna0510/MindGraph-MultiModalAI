"""Graph API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GraphUserResponse(BaseModel):
    user_id: str
    data: dict[str, Any] | None = None


class TimelineResponse(BaseModel):
    user_id: str
    sessions: list[dict[str, Any]] = Field(default_factory=list)


class RiskProgressionResponse(BaseModel):
    user_id: str
    points: list[dict[str, Any]] = Field(default_factory=list)


class ClinicalTrendsResponse(BaseModel):
    user_id: str
    trend_available: bool
    risk: list[dict[str, Any]] = Field(default_factory=list)
    emotions: list[dict[str, Any]] = Field(default_factory=list)
    symptoms: list[dict[str, Any]] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    user_id: str
    reasoning_path: list[str]
    explanation: str
    latest_risk: dict[str, Any] | None = None


class GraphStatisticsResponse(BaseModel):
    nodes: int
    edges: int
    execution_time_ms: float = 0.0


class SessionUpsertRequest(BaseModel):
    session_id: str
    owner: str
    timestamp: str | None = None
    quality_score: float = 0.0
    dataset: str = "daic"
    previous_session_id: str | None = None


class PredictionUpsertRequest(BaseModel):
    session_id: str
    owner: str
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    prediction_label: str
    model_version: str = "MODEL_010"


class MemoryRecordRequest(BaseModel):
    session_id: str
    owner: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    previous_session_id: str | None = None
    binary_prediction: int = 0
    emotions: list[dict] = Field(default_factory=list)
    symptoms: list[dict] = Field(default_factory=list)
    behaviours: list[dict] = Field(default_factory=list)
    recommendations: list[dict] = Field(default_factory=list)


class LongitudinalReportResponse(BaseModel):
    user_id: str
    risk: list[dict] = Field(default_factory=list)
    emotions: list[dict] = Field(default_factory=list)
    symptoms: list[dict] = Field(default_factory=list)
    behaviours: list[dict] = Field(default_factory=list)
    recommendations: list[dict] = Field(default_factory=list)
    improvement: dict = Field(default_factory=dict)
    decline: dict = Field(default_factory=dict)
