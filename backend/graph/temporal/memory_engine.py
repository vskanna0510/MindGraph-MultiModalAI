"""Temporal memory engine — append-only longitudinal graph updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from neo4j import AsyncSession

from app.exceptions.base import ValidationError
from graph.audit import AuditService
from graph.cache import GraphCache
from graph.entities.factories import (
    BehaviourFactory,
    EmotionFactory,
    ObservationFactory,
    PredictionFactory,
    RecommendationFactory,
    RiskFactory,
    SessionFactory,
    SymptomFactory,
    UserFactory,
)
from graph.queries.library import merge_relationship, persist_entity
from graph.relationships import GraphRelationship
from graph.repositories.base import BaseGraphRepository
from graph.validators import GraphValidator


@dataclass
class MemoryUpdateResult:
    """Result of a single memory append operation."""

    user_id: str
    session_id: str
    prediction_id: str | None = None
    risk_id: str | None = None
    observations: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    behaviours: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    temporal_linked: bool = False


class TemporalMemoryEngine:
    """Evolving digital memory — nothing deleted, everything appended.

    Flow:
        Current Session → Graph Update → Historical Memory → Future Inference
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        cache: GraphCache | None = None,
        audit: AuditService | None = None,
    ) -> None:
        self._repo = BaseGraphRepository(session)
        self._validator = GraphValidator()
        self._cache = cache or GraphCache()
        self._audit = audit or AuditService()

    async def _persist(self, entity: Any) -> dict[str, Any]:
        result = self._validator.validate_entity(entity)
        if not result.valid:
            raise ValidationError("; ".join(result.errors))
        cypher, params = persist_entity(entity)
        records, meta = await self._repo._write(cypher, params)
        self._audit.log_mutation(
            "append_memory",
            entity_type=entity.label,
            entity_id=entity.properties.get(
                list(entity.properties.keys())[0] if entity.properties else "uuid",
                entity.identity.uuid,
            ),
            execution_time_ms=meta.execution_time_ms,
        )
        return self._repo._node_props(records[0]) if records else entity.to_neo4j_props()

    async def _link(
        self,
        rel: GraphRelationship,
        source_label: str,
        target_label: str,
    ) -> None:
        result = self._validator.validate_relationship(rel)
        if not result.valid:
            raise ValidationError("; ".join(result.errors))
        cypher, params = merge_relationship(rel, source_label, target_label)
        await self._repo._write(cypher, params)

    async def ensure_user(self, user_id: str, **props: Any) -> dict[str, Any]:
        entity = UserFactory.create(user_id, owner=user_id, **props)
        return await self._persist(entity)

    async def record_session(
        self,
        user_id: str,
        session_id: str,
        *,
        previous_session_id: str | None = None,
        **session_props: Any,
    ) -> dict[str, Any]:
        session_entity = SessionFactory.create(session_id, owner=user_id, **session_props)
        saved = await self._persist(session_entity)
        await self._link(
            GraphRelationship("HAS_SESSION", "user_id", user_id, "session_id", session_id),
            "User",
            "Session",
        )
        if previous_session_id:
            await self._link(
                GraphRelationship(
                    "TEMPORALLY_PRECEDES",
                    "session_id",
                    previous_session_id,
                    "session_id",
                    session_id,
                ),
                "Session",
                "Session",
            )
        self._cache.invalidate_user(user_id)
        return saved

    async def append_observation(
        self,
        session_id: str,
        user_id: str,
        *,
        source_modality: str,
        confidence: float,
        **props: Any,
    ) -> dict[str, Any]:
        entity = ObservationFactory.create(
            owner=user_id,
            source_modality=source_modality,
            confidence=confidence,
            **props,
        )
        saved = await self._persist(entity)
        obs_id = entity.properties["observation_id"]
        # Link observation to session via temporal event pattern
        await self._link(
            GraphRelationship("HAS_OBSERVATION", "session_id", session_id, "observation_id", obs_id),
            "Session",
            "Observation",
        )
        return saved

    async def append_inference(
        self,
        user_id: str,
        session_id: str,
        *,
        risk_probability: float,
        confidence: float,
        binary_prediction: int = 0,
        emotions: list[dict[str, Any]] | None = None,
        symptoms: list[dict[str, Any]] | None = None,
        behaviours: list[dict[str, Any]] | None = None,
        recommendations: list[dict[str, Any]] | None = None,
        **prediction_props: Any,
    ) -> MemoryUpdateResult:
        """Append prediction + risk + identified entities — never overwrites."""
        pred_entity = PredictionFactory.create(
            owner=user_id,
            risk_probability=risk_probability,
            confidence=confidence,
            binary_prediction=binary_prediction,
            **prediction_props,
        )
        pred_saved = await self._persist(pred_entity)
        pred_id = pred_entity.properties["prediction_id"]

        await self._link(
            GraphRelationship("PRODUCED", "session_id", session_id, "prediction_id", pred_id),
            "Session",
            "Prediction",
        )

        risk_entity = RiskFactory.create(
            owner=user_id,
            risk_score=risk_probability,
            confidence=confidence,
            trend=prediction_props.get("trend", "stable"),
            change_rate=prediction_props.get("change_rate", 0.0),
            baseline=prediction_props.get("baseline", risk_probability),
        )
        risk_saved = await self._persist(risk_entity)
        risk_id = risk_entity.properties["risk_id"]

        await self._link(
            GraphRelationship("ESTIMATES", "prediction_id", pred_id, "risk_id", risk_id),
            "Prediction",
            "Risk",
        )

        result = MemoryUpdateResult(
            user_id=user_id,
            session_id=session_id,
            prediction_id=pred_id,
            risk_id=risk_id,
        )

        for emo in emotions or []:
            entity = EmotionFactory.create(
                str(emo.get("emotion_name", emo.get("emotion", "Neutral"))),
                owner=user_id,
                probability=float(emo.get("probability", 0.5)),
                confidence=float(emo.get("confidence", 0.5)),
            )
            await self._persist(entity)
            eid = entity.properties["emotion_id"]
            await self._link(
                GraphRelationship("IDENTIFIED", "prediction_id", pred_id, "emotion_id", eid),
                "Prediction",
                "Emotion",
            )
            result.emotions.append(eid)

        for sym in symptoms or []:
            entity = SymptomFactory.create(
                str(sym.get("symptom_name", sym.get("symptom", "Low Energy"))),
                owner=user_id,
                confidence=float(sym.get("confidence", 0.5)),
                severity=float(sym.get("severity", 0.5)),
            )
            await self._persist(entity)
            sid = entity.properties["symptom_id"]
            await self._link(
                GraphRelationship("IDENTIFIED", "prediction_id", pred_id, "symptom_id", sid),
                "Prediction",
                "Symptom",
            )
            result.symptoms.append(sid)

        for beh in behaviours or []:
            entity = BehaviourFactory.create(
                str(beh.get("behaviour_name", "Reduced Expression")),
                owner=user_id,
                value=float(beh.get("value", 0.5)),
                confidence=float(beh.get("confidence", 0.5)),
            )
            await self._persist(entity)
            bid = entity.properties["behaviour_id"]
            await self._link(
                GraphRelationship("IDENTIFIED", "prediction_id", pred_id, "behaviour_id", bid),
                "Prediction",
                "Behaviour",
            )
            result.behaviours.append(bid)

        for rec in recommendations or []:
            entity = RecommendationFactory.create(
                str(rec.get("recommendation_type", rec.get("type", "Mindfulness"))),
                owner=user_id,
                confidence=float(rec.get("confidence", 0.6)),
                priority=rec.get("priority", "medium"),
            )
            await self._persist(entity)
            rid = entity.properties["recommendation_id"]
            await self._link(
                GraphRelationship("GENERATED_REC", "prediction_id", pred_id, "recommendation_id", rid),
                "Prediction",
                "Recommendation",
            )
            result.recommendations.append(rid)

        self._cache.invalidate_user(user_id)
        return result

    async def link_causal(
        self,
        source_symptom_id: str,
        target_symptom_id: str,
        *,
        weight: float = 0.5,
        confidence: float = 0.6,
        evidence_source: str = "clinical_rules",
    ) -> None:
        rel = GraphRelationship(
            "MAY_INFLUENCE",
            "symptom_id",
            source_symptom_id,
            "symptom_id",
            target_symptom_id,
            properties={"weight": weight, "confidence": confidence, "evidence_source": evidence_source},
        )
        await self._link(rel, "Symptom", "Symptom")

    async def link_semantic(
        self,
        emotion_id: str,
        symptom_id: str,
        *,
        confidence: float = 0.7,
    ) -> None:
        rel = GraphRelationship(
            "CORRELATES_WITH",
            "emotion_id",
            emotion_id,
            "symptom_id",
            symptom_id,
            properties={"confidence": confidence},
        )
        await self._link(rel, "Emotion", "Symptom")
