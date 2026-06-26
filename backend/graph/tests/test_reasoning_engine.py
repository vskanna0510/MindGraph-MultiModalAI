"""MP4 Part 3 — reasoning engine, analytics, clinical intelligence tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graph.analytics import GraphMetricsEngine, MaterializedViewStore
from graph.contracts import GraphServiceResponse, TimedOperation
from graph.reasoning.anomaly import AnomalyDetector
from graph.reasoning.patterns import PatternDetector
from graph.reasoning.risk_trend import RiskTrendEngine
from graph.reasoning.similarity import GraphSimilarityEngine
from graph.reasoning.temporal_reasoning import TemporalReasoningEngine, TemporalState


class TestRiskTrendEngine:
    def test_compute_metrics(self) -> None:
        scores = [0.3, 0.4, 0.5, 0.6, 0.7]
        m = RiskTrendEngine.compute(scores)
        assert m.current_risk == 0.7
        assert m.peak_risk == 0.7
        assert m.slope > 0

    def test_empty_scores(self) -> None:
        m = RiskTrendEngine.compute([])
        assert m.n_points == 0

    def test_to_dict(self) -> None:
        d = RiskTrendEngine.to_dict(RiskTrendEngine.compute([0.5, 0.6]))
        assert "slope" in d
        assert "confidence_interval" in d


class TestTemporalReasoning:
    def test_deterioration(self) -> None:
        engine = TemporalReasoningEngine()
        result = engine.analyze_risk([0.3, 0.4, 0.5, 0.7, 0.8])
        assert result["state"] in {
            TemporalState.DETERIORATION.value,
            TemporalState.SUDDEN_CHANGE.value,
            TemporalState.RELAPSE.value,
        }

    def test_recovery(self) -> None:
        engine = TemporalReasoningEngine()
        result = engine.analyze_risk([0.8, 0.7, 0.5, 0.4, 0.3])
        assert result["state"] in {TemporalState.RECOVERY.value, TemporalState.IMPROVEMENT.value, TemporalState.STABLE.value}

    def test_insufficient_data(self) -> None:
        engine = TemporalReasoningEngine()
        result = engine.analyze_risk([0.5])
        assert result["state"] == TemporalState.INSUFFICIENT_DATA.value

    def test_emotion_stability(self) -> None:
        engine = TemporalReasoningEngine()
        stable = engine.analyze_emotion_stability([0.5, 0.51, 0.49, 0.5])
        volatile = engine.analyze_emotion_stability([0.1, 0.9, 0.2, 0.8])
        assert stable["stability"] > volatile["stability"]


class TestPatternDetection:
    def test_repeated_items(self) -> None:
        patterns = PatternDetector.repeated_items(["a", "b", "a", "a"], min_count=2)
        assert len(patterns) == 1
        assert patterns[0]["item"] == "a"

    def test_persistent_symptoms(self) -> None:
        records = [{"symptom": "Low Energy", "conf": 0.7}] * 4
        persistent = PatternDetector.persistent_symptoms(records, min_sessions=3)
        assert len(persistent) == 1

    def test_escalation(self) -> None:
        assert PatternDetector.escalation_pattern([0.3, 0.4, 0.5, 0.6])
        assert not PatternDetector.escalation_pattern([0.6, 0.5, 0.4])


class TestAnomalyDetection:
    def test_unexpected_risk_increase(self) -> None:
        anomaly = AnomalyDetector.unexpected_risk_increase([0.3, 0.3, 0.3, 0.9])
        assert anomaly is not None
        assert anomaly["type"] == "unexpected_risk_increase"

    def test_sudden_mood_shift(self) -> None:
        shift = AnomalyDetector.sudden_mood_shift([0.2, 0.8])
        assert shift is not None

    def test_scan_multiple(self) -> None:
        anomalies = AnomalyDetector.scan([0.3, 0.3, 0.9], [0.2, 0.8], [10])
        assert len(anomalies) >= 2


class TestSimilarity:
    def test_cosine(self) -> None:
        sim = GraphSimilarityEngine.cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert sim == pytest.approx(1.0)

    def test_jaccard(self) -> None:
        sim = GraphSimilarityEngine.jaccard_similarity({"a", "b"}, {"b", "c"})
        assert sim == pytest.approx(1 / 3, abs=0.01)

    def test_cross_user_blocked(self) -> None:
        with pytest.raises(PermissionError):
            GraphSimilarityEngine.user_similarity_research({}, {}, research_mode=False)

    def test_session_similarity(self) -> None:
        result = GraphSimilarityEngine.session_similarity(
            ["Low Energy"], ["Low Energy", "Sleep Issues"],
            ["Sadness"], ["Sadness"],
        )
        assert "combined" in result


class TestGraphMetrics:
    def test_density_computation(self) -> None:
        metrics = GraphMetricsEngine.from_degree_map({"a": 3, "b": 1}, node_count=3, edge_count=4)
        assert metrics["node_count"] == 3
        assert metrics["avg_degree"] > 0

    def test_node_importance(self) -> None:
        records = [{"label": "Symptom", "uuid": "u1", "degree": 5}, {"label": "Emotion", "uuid": "u2", "degree": 2}]
        ranked = GraphMetricsEngine.node_importance(records)
        assert ranked[0]["degree"] == 5


class TestMaterializedViews:
    def test_store_and_retrieve(self) -> None:
        store = MaterializedViewStore()
        view = store.risk_timeline_view("u1", [{"risk": 0.5}])
        latest = store.get_latest("u1", "risk_timeline")
        assert latest is not None
        assert latest.view_id == view.view_id

    def test_weekly_summary(self) -> None:
        store = MaterializedViewStore()
        view = store.weekly_summary("u1", [{"risk": 0.5}, {"risk": 0.7}], [])
        assert view.data["risk_points"] == 2


class TestGraphServiceResponse:
    def test_to_dict(self) -> None:
        resp = GraphServiceResponse(result={"ok": True}, execution_time_ms=12.5, confidence=0.9)
        d = resp.to_dict()
        assert d["result"]["ok"]
        assert d["execution_time_ms"] == 12.5
        assert "correlation_id" in d

    def test_timed_operation(self) -> None:
        with TimedOperation() as t:
            pass
        assert t.elapsed_ms >= 0


class TestCypherQueries:
    def test_user_queries_parameterized(self) -> None:
        from graph.queries.user_queries import get_user
        cypher, params = get_user("u1")
        assert "$user_id" in cypher
        assert params["user_id"] == "u1"
        assert "u1" not in cypher  # injection safe

    def test_trend_queries(self) -> None:
        from graph.queries.trend_queries import risk_trend
        cypher, params = risk_trend("u1")
        assert "$user_id" in cypher

    def test_subgraph_queries(self) -> None:
        from graph.queries.subgraph_queries import radius_subgraph
        cypher, params = radius_subgraph("u1", "emotion", 2)
        assert "Emotion" in cypher
