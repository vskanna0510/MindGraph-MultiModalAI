"""MP4 Part 4 — Digital Twin, recommendations, explainability, export tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from graph.evaluation.research import ResearchEvaluator
from graph.explainability import GraphExplainabilityEngine
from graph.export.formats import GraphExportEngine
from graph.recommendation import RecommendationPrioritizer
from graph.twin.attention import AttentionVisualizer
from graph.twin.digital_twin import DigitalTwin, DigitalTwinBuilder
from graph.twin.evolution_score import EvolutionScoreCalculator
from graph.twin.intelligence import GraphIntelligenceEngine
from graph.twin.risk_evolution import RiskEvolutionModel
from graph.twin.snapshots import TwinSnapshotGranularity, TwinSnapshotStore
from graph.twin.story import GraphStoryGenerator
from graph.services.twin_service import TwinService
from graph.tests.conftest import MockAsyncSession


class TestDigitalTwinBuilder:
    def test_build_profiles_from_history(self) -> None:
        twin = DigitalTwinBuilder.build(
            "u1",
            risk_scores=[0.6, 0.5, 0.4],
            emotions=[
                {"emotion": "Sad", "prob": 0.7},
                {"emotion": "Neutral", "prob": 0.3},
            ],
            symptoms=[{"symptom": "Low Energy", "conf": 0.8}] * 3,
            behaviours=[{"behaviour": "speech rate", "value": 0.6}],
            sessions=[{"session_id": "s1", "duration": 120, "quality_score": 0.8}],
            recommendations=[{"type": "Journaling", "status": "completed"}],
        )
        assert twin.user_id == "u1"
        assert twin.emotion.dominant_emotion == "Sad"
        assert "Low Energy" in twin.symptom.recurring
        assert twin.recovery.risk_reduction == pytest.approx(0.2, abs=0.01)
        assert twin.interaction.total_sessions == 1

    def test_incremental_update_preserves_twin_id(self) -> None:
        existing = DigitalTwin(user_id="u1", twin_id="fixed-id")
        twin = DigitalTwinBuilder.build(
            "u1",
            risk_scores=[0.5],
            emotions=[],
            symptoms=[],
            behaviours=[],
            sessions=[],
            recommendations=[],
            existing=existing,
        )
        assert twin.twin_id == "fixed-id"


class TestRiskEvolution:
    def test_forecasts_with_data(self) -> None:
        model = RiskEvolutionModel(min_sessions=2)
        result = model.analyze([0.5, 0.55, 0.6, 0.65])
        assert 0.0 <= result["current_risk"] <= 1.0
        assert "forecast_7d" in result
        assert "forecast_30d" in result
        assert "forecast_90d" in result
        assert result["sufficient_data"]

    def test_empty_scores(self) -> None:
        result = RiskEvolutionModel().analyze([])
        assert result["current_risk"] == 0.0
        assert not result["sufficient_data"]


class TestEvolutionScore:
    def test_maturity_levels(self) -> None:
        twin = DigitalTwin(user_id="u1", temporal_memory_sessions=20)
        twin.emotion.variability = 0.1
        twin.recovery.stability = 0.8
        twin.recovery.recommendation_compliance = 0.7
        score = EvolutionScoreCalculator.compute(twin)
        assert score["overall_score"] > 0
        assert score["maturity"] in {"nascent", "developing", "mature"}


class TestGraphStory:
    def test_non_diagnostic_narrative(self) -> None:
        twin = DigitalTwinBuilder.build(
            "u1",
            risk_scores=[0.7, 0.5, 0.4],
            emotions=[{"emotion": "Happy", "prob": 0.8}],
            symptoms=[],
            behaviours=[],
            sessions=[{"session_id": "s1"}],
            recommendations=[],
        )
        story = GraphStoryGenerator.generate(twin, window_days=30)
        assert story["is_diagnostic"] is False
        assert "not a clinical diagnosis" in story["disclaimers"][0]
        assert "improvement" in story["narrative"].lower() or "recorded" in story["narrative"].lower()

    def test_empty_history(self) -> None:
        twin = DigitalTwin(user_id="u1")
        story = GraphStoryGenerator.generate(twin)
        assert "No session history" in story["narrative"]


class TestAttention:
    def test_heatmap_generation(self) -> None:
        result = AttentionVisualizer.compute(
            emotions=[{"emotion": "Sad", "prob": 0.9}],
            symptoms=[{"symptom": "Fatigue", "conf": 0.7}],
            behaviours=[{"behaviour": "low speech", "value": 0.5}],
            sessions=[{"session_id": "s1", "quality_score": 0.8}],
            recommendations=[{"type": "Mindfulness", "confidence": 0.6}],
        )
        assert "node_attention" in result
        assert "heatmap" in result
        assert len(result["node_attention"]) >= 1


class TestRecommendationPrioritizer:
    def test_low_risk_recommendations(self) -> None:
        prioritizer = RecommendationPrioritizer()
        recs = prioritizer.generate(current_risk=0.25, compliance_history=0.6)
        assert len(recs) > 0
        assert all(r["guidance_only"] for r in recs)
        assert all(not r["is_diagnosis"] for r in recs)
        assert all(r["tier"] == "low" for r in recs[:2])

    def test_high_risk_includes_urgent(self) -> None:
        recs = RecommendationPrioritizer().generate(current_risk=0.85)
        types = {r["type"] for r in recs}
        assert "Emergency Helpline" in types or "Immediate Mental Health Resource" in types

    def test_diversity_avoids_duplicates(self) -> None:
        recs = RecommendationPrioritizer().generate(
            current_risk=0.5,
            previous_types=["Breathing Exercise", "Mood Tracking"],
        )
        assert len(recs) <= 8


class TestExplainability:
    def test_explanation_path(self) -> None:
        twin = DigitalTwinBuilder.build(
            "u1",
            risk_scores=[0.6],
            emotions=[{"emotion": "Anxious", "prob": 0.7}],
            symptoms=[{"symptom": "Low Energy", "conf": 0.8}] * 3,
            behaviours=[{"behaviour": "speech rate", "value": 0.7}],
            sessions=[{"session_id": "s1"}],
            recommendations=[],
        )
        rec = {"type": "Breathing Exercise", "tier": "medium", "confidence": 0.75}
        explanation = GraphExplainabilityEngine.explain_recommendation(
            rec, twin, supporting_sessions=[{"session_id": "s1"}]
        )
        assert explanation["recommendation"] == "Breathing Exercise"
        assert "why" in explanation
        assert "explanation_path" in explanation
        assert "not a clinical diagnosis" in explanation["guidance_disclaimer"]


class TestGraphExport:
    def test_json_export(self) -> None:
        data = {"user_id": "u1", "version": "2.0.0"}
        exported = GraphExportEngine.export_twin(data, "json")
        assert '"user_id": "u1"' in exported

    def test_networkx_export(self) -> None:
        data = DigitalTwin(user_id="u1").to_dict()
        nx = GraphExportEngine.export_twin(data, "networkx")
        assert "nodes" in nx
        assert "links" in nx

    def test_graphml_export(self) -> None:
        data = DigitalTwin(user_id="u1").to_dict()
        graphml = GraphExportEngine.export_twin(data, "graphml")
        assert "<graphml" in graphml


class TestTwinSnapshots:
    def test_capture_and_compare(self) -> None:
        store = TwinSnapshotStore()
        twin = DigitalTwin(user_id="u1", temporal_memory_sessions=5)
        score_a = {"overall_score": 0.4}
        score_b = {"overall_score": 0.6}
        snap_a = store.capture(twin, score_a, TwinSnapshotGranularity.DAILY)
        twin.temporal_memory_sessions = 8
        snap_b = store.capture(twin, score_b, TwinSnapshotGranularity.WEEKLY)
        comparison = store.compare("u1", snap_a.snapshot_id, snap_b.snapshot_id)
        assert comparison["evolution_delta"] == pytest.approx(0.2)
        assert comparison["sessions_delta"] == 3


class TestResearchEvaluator:
    def test_evaluation_metrics(self) -> None:
        twin = DigitalTwin(user_id="u1", temporal_memory_sessions=10)
        evolution = EvolutionScoreCalculator.compute(twin)
        metrics = ResearchEvaluator.evaluate(
            twin=twin,
            evolution_score=evolution,
            reasoning_latency_ms=45.0,
            recommendation_count=3,
        )
        assert metrics["graph_growth_sessions"] == 10
        assert metrics["reasoning_latency_ms"] == 45.0
        assert metrics["research_mode"] is True


@pytest.mark.asyncio
class TestGraphIntelligenceEngine:
    async def test_update_twin_with_mocked_repo(self, mock_neo4j_session: MockAsyncSession) -> None:
        engine = GraphIntelligenceEngine(mock_neo4j_session)  # type: ignore[arg-type]
        mock_data = {
            "risk_scores": [0.4, 0.5],
            "emotions": [{"emotion": "Neutral", "prob": 0.5}],
            "symptoms": [],
            "behaviours": [],
            "sessions": [{"session_id": "s1"}],
            "recommendations": [],
            "user_props": {},
        }
        with patch.object(engine, "_fetch_user_data", AsyncMock(return_value=mock_data)):
            response = await engine.update_twin("u1", requester_id="u1")
        assert response.result["twin"]["user_id"] == "u1"
        assert "risk_evolution" in response.result
        assert "evolution_score" in response.result

    async def test_unauthorized_access_raises(self, mock_neo4j_session: MockAsyncSession) -> None:
        engine = GraphIntelligenceEngine(mock_neo4j_session)  # type: ignore[arg-type]
        with pytest.raises(PermissionError):
            await engine.update_twin("u1", requester_id="other-user")


@pytest.mark.asyncio
class TestTwinService:
    async def test_personalized_recommendations(self, mock_neo4j_session: MockAsyncSession) -> None:
        service = TwinService(mock_neo4j_session)  # type: ignore[arg-type]
        mock_data = {
            "risk_scores": [0.55],
            "emotions": [{"emotion": "Sad", "prob": 0.6}],
            "symptoms": [],
            "behaviours": [],
            "sessions": [{"session_id": "s1"}],
            "recommendations": [],
            "user_props": {},
        }
        with patch.object(service._engine, "_fetch_user_data", AsyncMock(return_value=mock_data)):
            with patch.object(service._engine, "_fetch_emotions", AsyncMock(return_value=[])):
                with patch.object(service._engine, "_fetch_symptoms", AsyncMock(return_value=[])):
                    with patch.object(service._engine, "_fetch_behaviours", AsyncMock(return_value=[])):
                        with patch.object(service._engine, "_fetch_sessions", AsyncMock(return_value=[])):
                            with patch.object(service._engine, "_fetch_recommendations", AsyncMock(return_value=[])):
                                with patch.object(
                                    service._engine, "_longitudinal_summaries", AsyncMock(return_value={})
                                ):
                                    response = await service.personalized_recommendations("u1")
        assert response.result["guidance_only"] is True
        assert len(response.result["recommendations"]) > 0
        assert response.execution_time_ms >= 0

    async def test_export_twin_json(self, mock_neo4j_session: MockAsyncSession) -> None:
        service = TwinService(mock_neo4j_session)  # type: ignore[arg-type]
        mock_data = {
            "risk_scores": [0.3],
            "emotions": [],
            "symptoms": [],
            "behaviours": [],
            "sessions": [],
            "recommendations": [],
            "user_props": {},
        }
        with patch.object(service._engine, "_fetch_user_data", AsyncMock(return_value=mock_data)):
            response = await service.export_twin("u1", "json")
        assert response.result["format"] == "json"
        assert "u1" in response.result["data"]
