"""MP4 Part 2 — temporal schema, memory engine, longitudinal tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from graph.entities.factories import (
    AssessmentFactory,
    BehaviourFactory,
    ObservationFactory,
    PredictionFactory,
    ResearchFactory,
    RiskFactory,
    SessionFactory,
    SymptomFactory,
    UserFactory,
)
from graph.entities.schemas import (
    APPEND_ONLY_LABELS,
    SUPPORTED_BEHAVIOURS,
    SUPPORTED_EMOTIONS,
    SUPPORTED_SYMPTOMS,
)
from graph.queries.library import create_entity, merge_entity, persist_entity
from graph.relationships import GraphRelationship, edge_properties
from graph.retention import DataRetentionPolicy, RetentionAction
from graph.schema.nodes.definitions import NODE_PROPERTY_SCHEMAS
from graph.schema.relationships.definitions import RELATIONSHIP_TYPES
from graph.temporal.memory_engine import TemporalMemoryEngine
from graph.temporal.snapshots import SnapshotGranularity, SnapshotManager
from graph.validators import GraphValidator


class TestEntitySchemas:
    def test_all_core_labels_have_schemas(self) -> None:
        for label in ("User", "Session", "Observation", "Prediction", "Risk", "Emotion", "Symptom", "Behaviour"):
            assert label in NODE_PROPERTY_SCHEMAS

    def test_append_only_labels(self) -> None:
        assert "Prediction" in APPEND_ONLY_LABELS
        assert "User" not in APPEND_ONLY_LABELS

    def test_supported_catalogs(self) -> None:
        assert "Joy" in SUPPORTED_EMOTIONS
        assert "Sleep Issues" in SUPPORTED_SYMPTOMS
        assert "Reduced Eye Contact" in SUPPORTED_BEHAVIOURS


class TestV2Factories:
    def test_user_v2_properties(self) -> None:
        u = UserFactory.create("u1", owner="u1", timezone="Asia/Kolkata")
        assert u.properties["graph_version"] == "2.0.0"
        assert u.properties["timezone"] == "Asia/Kolkata"

    def test_observation_factory(self) -> None:
        o = ObservationFactory.create(owner="u1", source_modality="audio", confidence=0.9)
        assert o.label == "Observation"
        assert o.properties["observation_id"].startswith("obs_")

    def test_prediction_append_properties(self) -> None:
        p = PredictionFactory.create(owner="u1", risk_probability=0.7, confidence=0.8)
        assert p.properties["risk_probability"] == 0.7
        assert p.properties["uncertainty"] == pytest.approx(0.2)

    def test_risk_factory(self) -> None:
        r = RiskFactory.create(owner="u1", risk_score=0.75, confidence=0.9)
        assert r.properties["risk_level"] == "high"

    def test_behaviour_factory(self) -> None:
        b = BehaviourFactory.create("Speech Pause", owner="u1", value=0.6, confidence=0.7)
        assert b.properties["behaviour_name"] == "Speech Pause"

    def test_research_factory(self) -> None:
        r = ResearchFactory.create("exp_001", owner="system")
        assert r.properties["experiment_id"] == "exp_001"


class TestPersistRouting:
    def test_stable_entity_uses_merge(self) -> None:
        entity = UserFactory.create("u1", owner="u1")
        cypher, _ = persist_entity(entity)
        assert "MERGE" in cypher

    def test_append_entity_uses_create(self) -> None:
        entity = PredictionFactory.create(owner="u1", risk_probability=0.5, confidence=0.6)
        cypher, _ = persist_entity(entity)
        assert "CREATE" in cypher

    def test_create_entity_explicit(self) -> None:
        entity = RiskFactory.create(owner="u1", risk_score=0.4, confidence=0.5)
        cypher, params = create_entity(entity)
        assert "CREATE" in cypher
        assert "props" in params


class TestRelationshipsV2:
    def test_all_structural_types_registered(self) -> None:
        for rel in ("HAS_ASSESSMENT", "PRODUCED", "ESTIMATES", "IDENTIFIED", "MAY_INFLUENCE"):
            assert rel in RELATIONSHIP_TYPES

    def test_edge_properties_schema(self) -> None:
        props = edge_properties(confidence=0.8, weight=0.5, reason="inference")
        assert "uuid" in props
        assert props["confidence"] == 0.8

    def test_causal_relationship(self) -> None:
        rel = GraphRelationship(
            "MAY_INFLUENCE", "symptom_id", "s1", "symptom_id", "s2",
            properties={"evidence_source": "clinical_rules"},
        )
        result = GraphValidator().validate_relationship(rel)
        assert result.valid


class TestValidatorV2:
    def test_future_timestamp_rejected(self) -> None:
        future = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        entity = SessionFactory.create("s1", owner="u1", timestamp=future)
        result = GraphValidator().validate_entity(entity)
        assert not result.valid

    def test_temporal_order_valid(self) -> None:
        now = datetime.now(UTC).isoformat()
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        result = GraphValidator().validate_temporal_order([past, now])
        assert result.valid

    def test_temporal_order_invalid(self) -> None:
        now = datetime.now(UTC).isoformat()
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        result = GraphValidator().validate_temporal_order([now, past])
        assert not result.valid

    def test_session_connectivity(self) -> None:
        result = GraphValidator().validate_session_connectivity(
            ["s1", "s2", "s3"],
            [("s1", "s2"), ("s2", "s3")],
        )
        assert result.valid

    def test_disconnected_sessions(self) -> None:
        result = GraphValidator().validate_session_connectivity(["s1", "s2"], [])
        assert not result.valid


class TestSnapshots:
    @pytest.mark.asyncio
    async def test_capture_snapshot(self, mock_neo4j_session) -> None:
        mock_neo4j_session._responses = [{"node_count": 5}]
        mgr = SnapshotManager(mock_neo4j_session)  # type: ignore[arg-type]
        snap = await mgr.capture("u1", SnapshotGranularity.WEEKLY)
        assert snap.user_id == "u1"
        assert snap.granularity == "weekly"
        assert snap.node_count == 5

    def test_rollback_reference(self) -> None:
        from graph.temporal.snapshots import GraphSnapshot

        mgr = SnapshotManager(AsyncMock())  # type: ignore[arg-type]
        s = GraphSnapshot(user_id="u1", snapshot_id="snap-1")
        mgr._snapshots["u1"] = [s]
        assert mgr.rollback_reference("u1", "snap-1") is not None
        assert mgr.rollback_reference("u1", "missing") is None


class TestMemoryEngine:
    @pytest.mark.asyncio
    async def test_append_inference_structure(self, mock_neo4j_session) -> None:
        mock_neo4j_session._responses = [{"n": {"prediction_id": "p1", "risk_id": "r1"}}]
        engine = TemporalMemoryEngine(mock_neo4j_session)  # type: ignore[arg-type]
        result = await engine.append_inference(
            "u1",
            "sess1",
            risk_probability=0.6,
            confidence=0.8,
            emotions=[{"emotion_name": "Sadness", "probability": 0.7}],
            symptoms=[{"symptom_name": "Low Energy", "confidence": 0.6}],
        )
        assert result.prediction_id is not None
        assert result.risk_id is not None


class TestRetention:
    @pytest.mark.asyncio
    async def test_consent_withdrawal(self, mock_neo4j_session) -> None:
        mock_neo4j_session.execute_write = AsyncMock(return_value=[{"u": {}}])
        policy = DataRetentionPolicy(mock_neo4j_session)  # type: ignore[arg-type]
        result = await policy.consent_withdrawal("u1")
        assert result.action == RetentionAction.CONSENT_WITHDRAWAL.value
