"""Session graph builder — bridges ML pipeline data to graph entities."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession

from graph.entities.factories import EmotionFactory, FeatureFactory, SymptomFactory
from graph.repositories import EmotionRepository, SessionRepository, SymptomRepository, UserRepository
from graph.services.graph_service import GraphService
from graph.validators import GraphValidator


class SessionGraphBuilder:
    """Build a complete session subgraph from extraction output."""

    def __init__(self, session: AsyncSession) -> None:
        self._graph = GraphService(session)
        self._users = UserRepository(session)
        self._sessions = SessionRepository(session)
        self._emotions = EmotionRepository(session)
        self._symptoms = SymptomRepository(session)
        self._validator = GraphValidator()

    async def build_from_extraction(self, data: dict[str, Any]) -> dict[str, Any]:
        user_id = str(data["participant_id"])
        session_id = str(data.get("session_id", user_id))
        owner = str(data.get("owner", user_id))

        saved_session = await self._graph.upsert_session(
            user_id,
            session_id,
            owner=owner,
            timestamp=data.get("timestamp"),
            quality_score=float(data.get("quality_score", 0.0)),
            dataset=str(data.get("dataset", "daic")),
            link_previous=data.get("previous_session_id"),
        )

        features_created = 0
        for mod, label in (
            ("audio", "AudioFeature"),
            ("visual", "VisualFeature"),
            ("text", "TextFeature"),
            ("image", "ImageFeature"),
        ):
            if not data.get("modalities", {}).get(mod, data.get(f"has_{mod}", False)):
                continue
            feat_id = f"{session_id}_{mod}"
            entity = FeatureFactory.create(
                label,
                feat_id,
                owner=owner,
                dimension=int(data.get("embedding_dims", {}).get(mod, 128)),
                quality_score=float(data.get("quality", {}).get(mod, 0.8)),
            )
            result = self._validator.validate_entity(entity)
            if result.valid:
                from graph.repositories.base import BaseGraphRepository

                repo = BaseGraphRepository(self._sessions._session)
                from graph.queries import merge_entity, merge_relationship
                from graph.relationships import GraphRelationship

                cypher, params = merge_entity(entity)
                await repo._write(cypher, params)
                rel = GraphRelationship("HAS_FEATURE", "session_id", session_id, "feature_id", feat_id)
                cypher, params = merge_relationship(rel, "Session", label)
                await repo._write(cypher, params)
                features_created += 1

        if "label" in data:
            await self._graph.upsert_prediction(
                session_id,
                owner=owner,
                risk_score=float(data.get("risk_score", 0.5)),
                confidence=float(data.get("prediction_confidence", 0.5)),
                prediction_label=str(data["label"]),
                model_version=data.get("model_version", "MODEL_010"),
            )

        emotions_created = 0
        for emo in data.get("emotions", []):
            entity = EmotionFactory.create(
                str(emo.get("emotion", "Neutral")),
                owner=owner,
                probability=float(emo.get("probability", 0.5)),
                source_modality=str(emo.get("source", "multimodal")),
            )
            if self._validator.validate_entity(entity).valid:
                saved = await self._emotions.save(entity)
                await self._emotions.link_to_session(session_id, saved.get("uuid", entity.identity.uuid))
                emotions_created += 1

        symptoms_created = 0
        for sym in data.get("symptoms", []):
            entity = SymptomFactory.create(
                str(sym.get("symptom", "Low Mood")),
                owner=owner,
                confidence=float(sym.get("confidence", 0.5)),
            )
            if self._validator.validate_entity(entity).valid:
                saved = await self._symptoms.save(entity)
                await self._symptoms.link_to_session(session_id, saved.get("uuid", entity.identity.uuid))
                symptoms_created += 1

        return {
            "user_id": user_id,
            "session_id": session_id,
            "session": saved_session,
            "features_created": features_created,
            "emotions_created": emotions_created,
            "symptoms_created": symptoms_created,
        }
