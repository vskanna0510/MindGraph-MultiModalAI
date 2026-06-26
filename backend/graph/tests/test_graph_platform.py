"""Graph platform unit tests."""

from __future__ import annotations

import pytest

from graph.audit import AuditService
from graph.cache import GraphCache
from graph.entities.factories import (
    EmotionFactory,
    PredictionFactory,
    SessionFactory,
    UserFactory,
)
from graph.export import GraphExporter
from graph.migration.runner import MigrationRunner
from graph.monitoring import GraphMonitor
from graph.relationships import GraphRelationship
from graph.restore import GraphRestore
from graph.security import GraphSecurityPolicy
from graph.temporal import TimelineUnit, parse_timeline_window
from graph.validators import GraphValidator


class TestFactories:
    def test_user_factory_creates_identity(self) -> None:
        entity = UserFactory.create("u1", owner="u1", language="en")
        assert entity.label == "User"
        assert entity.properties["user_id"] == "u1"
        assert entity.identity.uuid
        assert entity.identity.checksum

    def test_session_factory(self) -> None:
        entity = SessionFactory.create("s1", owner="u1", quality_score=0.9)
        assert entity.properties["session_id"] == "s1"
        assert entity.properties["quality_score"] == 0.9

    def test_prediction_factory(self) -> None:
        entity = PredictionFactory.create(
            owner="u1", risk_probability=0.8, confidence=0.9, binary_prediction=1
        )
        assert entity.properties["risk_probability"] == 0.8

    def test_emotion_factory(self) -> None:
        entity = EmotionFactory.create("Sadness", owner="u1", probability=0.7)
        assert entity.properties["emotion_name"] == "Sadness"


class TestValidator:
    def test_valid_user(self) -> None:
        entity = UserFactory.create("u1", owner="u1")
        result = GraphValidator().validate_entity(entity)
        assert result.valid

    def test_invalid_confidence(self) -> None:
        entity = UserFactory.create("u1", owner="u1")
        bad = type(entity)(
            entity.label,
            type(entity.identity)(
                **{**entity.identity.__dict__, "confidence": 1.5}
            ),
            entity.properties,
        )
        result = GraphValidator().validate_entity(bad)
        assert not result.valid

    def test_circular_temporal(self) -> None:
        rel = GraphRelationship("TEMPORALLY_PRECEDES", "session_id", "s1", "session_id", "s1")
        result = GraphValidator().validate_relationship(rel)
        assert not result.valid

    def test_batch_duplicate_detection(self) -> None:
        e1 = UserFactory.create("u1", owner="u1")
        e2 = UserFactory.create("u1", owner="u1")
        result = GraphValidator().validate_batch([e1, e2], [])
        assert not result.valid


class TestCache:
    def test_set_and_get(self) -> None:
        cache = GraphCache(ttl_seconds=60)
        cache.set("key", {"a": 1})
        assert cache.get("key") == {"a": 1}

    def test_invalidate_user(self) -> None:
        cache = GraphCache()
        cache.set("graph:user:u1", {})
        cache.set("graph:temporal:u1:full", [])
        cache.invalidate_user("u1")
        assert cache.get("graph:user:u1") is None


class TestAudit:
    def test_log_mutation(self) -> None:
        audit = AuditService()
        record = audit.log_mutation("upsert", entity_type="User", entity_id="u1")
        assert record.operation == "upsert"
        assert len(audit.recent()) == 1


class TestMigrationParser:
    def test_parse_statements(self) -> None:
        content = "// comment\nCREATE INDEX x IF NOT EXISTS FOR (n:User) ON (n.uuid);\n\nCREATE INDEX y IF NOT EXISTS FOR (n:Session) ON (n.uuid);"
        stmts = MigrationRunner._parse_statements(content)
        assert len(stmts) == 2
        assert "CREATE INDEX x" in stmts[0]


class TestSecurity:
    def test_owner_read(self) -> None:
        assert GraphSecurityPolicy.can_read("u1", "u1")
        assert not GraphSecurityPolicy.can_read("u2", "u1")

    def test_sanitize_props(self) -> None:
        props = GraphSecurityPolicy.sanitize_props({"user_id": "u1", "password": "secret"})
        assert "password" not in props


class TestExport:
    def test_to_json(self) -> None:
        out = GraphExporter.to_json({"nodes": 1})
        assert '"nodes": 1' in out

    def test_to_csv(self) -> None:
        out = GraphExporter.to_csv_rows([{"a": 1, "b": 2}])
        assert "a,b" in out


class TestTemporal:
    def test_timeline_window(self) -> None:
        start, end = parse_timeline_window(7)
        assert start < end

    def test_timeline_units(self) -> None:
        assert TimelineUnit.DAYS.value == "days"


class TestMonitor:
    def test_record_and_summary(self) -> None:
        monitor = GraphMonitor()
        monitor.record("lookup", 5.0)
        summary = monitor.summary()
        assert summary["count"] == 1
        assert monitor.check_sla("lookup", 5.0)


class TestRestore:
    def test_parse_manifest(self) -> None:
        manifest = '{"backup_version": "2.0.0", "includes_raw_media": false}'
        data = GraphRestore.parse_manifest(manifest)
        assert data["backup_version"] == "2.0.0"

    def test_reject_raw_media(self) -> None:
        with pytest.raises(ValueError):
            GraphRestore.parse_manifest('{"backup_version": "1", "includes_raw_media": true}')
