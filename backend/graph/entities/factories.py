"""Entity factories — create immutable graph entities (v2.0)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from graph.entities.base import GraphEntity

def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class UserFactory:
    @staticmethod
    def create(
        user_id: str,
        *,
        owner: str,
        external_id: str | None = None,
        preferred_language: str = "en",
        timezone: str = "UTC",
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "User",
            owner=owner,
            user_id=user_id,
            external_id=external_id or user_id,
            preferred_language=preferred_language,
            timezone=timezone,
            consent_status=extra.get("consent_status", "granted"),
            privacy_mode=extra.get("privacy_mode", "standard"),
            edge_enabled=extra.get("edge_enabled", True),
            cloud_enabled=extra.get("cloud_enabled", True),
            notification_preferences=extra.get("notification_preferences", {}),
            graph_version=extra.get("graph_version", "2.0.0"),
            reasoning_version=extra.get("reasoning_version", "1.0.0"),
            country=extra.get("country", ""),
            **{k: v for k, v in extra.items() if k not in {
                "consent_status", "privacy_mode", "edge_enabled", "cloud_enabled",
                "notification_preferences", "graph_version", "reasoning_version", "country",
            }},
        )


class AssessmentFactory:
    @staticmethod
    def create(
        assessment_id: str,
        *,
        owner: str,
        assessment_type: str = "phq9",
        assessment_version: str = "1.0",
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Assessment",
            owner=owner,
            assessment_id=assessment_id,
            assessment_type=assessment_type,
            assessment_version=assessment_version,
            completed_at=extra.get("completed_at", _iso_now()),
            assessment_source=extra.get("assessment_source", "app"),
            assessment_duration=extra.get("assessment_duration", 0),
            language=extra.get("language", "en"),
            **{k: v for k, v in extra.items() if k not in {
                "completed_at", "assessment_source", "assessment_duration", "language",
            }},
        )


class SessionFactory:
    @staticmethod
    def create(
        session_id: str,
        *,
        owner: str,
        timestamp: str | None = None,
        quality_score: float = 0.0,
        duration: int = 0,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Session",
            owner=owner,
            session_id=session_id,
            timestamp=timestamp or _iso_now(),
            duration=duration,
            quality_score=quality_score,
            device=extra.get("device", "unknown"),
            platform=extra.get("platform", "mobile"),
            network_type=extra.get("network_type", "wifi"),
            processing_time=extra.get("processing_time", 0.0),
            language=extra.get("language", "en"),
            battery_level=extra.get("battery_level", 1.0),
            offline_mode=extra.get("offline_mode", False),
            edge_processing=extra.get("edge_processing", False),
            cloud_processing=extra.get("cloud_processing", True),
            **{k: v for k, v in extra.items() if k not in {
                "device", "platform", "network_type", "processing_time", "language",
                "battery_level", "offline_mode", "edge_processing", "cloud_processing",
            }},
        )


class ObservationFactory:
    @staticmethod
    def create(
        *,
        owner: str,
        source_modality: str,
        confidence: float,
        observation_id: str | None = None,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Observation",
            owner=owner,
            observation_id=observation_id or _new_id("obs"),
            timestamp=extra.get("timestamp", _iso_now()),
            confidence=confidence,
            source_modality=source_modality,
            quality=extra.get("quality", 0.8),
            processing_version=extra.get("processing_version", "1.0.0"),
            **{k: v for k, v in extra.items() if k not in {
                "timestamp", "quality", "processing_version",
            }},
        )


class PredictionFactory:
    @staticmethod
    def create(
        *,
        owner: str,
        risk_probability: float,
        confidence: float,
        binary_prediction: int = 0,
        prediction_id: str | None = None,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Prediction",
            owner=owner,
            prediction_id=prediction_id or _new_id("pred"),
            risk_probability=risk_probability,
            binary_prediction=binary_prediction,
            severity_score=extra.get("severity_score", risk_probability),
            confidence=confidence,
            uncertainty=extra.get("uncertainty", 1.0 - confidence),
            temperature_scaled_probability=extra.get(
                "temperature_scaled_probability", risk_probability
            ),
            model_version=extra.get("model_version", "MODEL_010"),
            created_at=extra.get("created_at", _iso_now()),
            # Legacy alias for Part 1 queries
            risk_score=risk_probability,
            prediction_label=str(extra.get("prediction_label", binary_prediction)),
            **{k: v for k, v in extra.items() if k not in {
                "severity_score", "uncertainty", "temperature_scaled_probability",
                "model_version", "created_at", "prediction_label",
            }},
        )


class RiskFactory:
    @staticmethod
    def create(
        *,
        owner: str,
        risk_score: float,
        confidence: float,
        risk_id: str | None = None,
        **extra: Any,
    ) -> GraphEntity:
        level = "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.4 else "low"
        return GraphEntity.build(
            "Risk",
            owner=owner,
            risk_id=risk_id or _new_id("risk"),
            risk_level=extra.get("risk_level", level),
            risk_score=risk_score,
            trend=extra.get("trend", "stable"),
            change_rate=extra.get("change_rate", 0.0),
            baseline=extra.get("baseline", risk_score),
            confidence=confidence,
            **{k: v for k, v in extra.items() if k not in {
                "risk_level", "trend", "change_rate", "baseline",
            }},
        )


class EmotionFactory:
    @staticmethod
    def create(
        emotion_name: str,
        *,
        owner: str,
        probability: float,
        emotion_id: str | None = None,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Emotion",
            owner=owner,
            emotion_id=emotion_id or _new_id("emo"),
            emotion_name=emotion_name,
            emotion=emotion_name,  # legacy
            probability=probability,
            confidence=extra.get("confidence", probability),
            timestamp=extra.get("timestamp", _iso_now()),
            source_modality=extra.get("source_modality", "multimodal"),
            **{k: v for k, v in extra.items() if k not in {
                "confidence", "timestamp", "source_modality",
            }},
        )


class SymptomFactory:
    @staticmethod
    def create(
        symptom_name: str,
        *,
        owner: str,
        confidence: float,
        severity: float = 0.5,
        symptom_id: str | None = None,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Symptom",
            owner=owner,
            symptom_id=symptom_id or _new_id("sym"),
            symptom_name=symptom_name,
            symptom=symptom_name,  # legacy
            severity=severity,
            confidence=confidence,
            source=extra.get("source", "inference"),
            timestamp=extra.get("timestamp", _iso_now()),
            **{k: v for k, v in extra.items() if k not in {"source", "timestamp"}},
        )


class BehaviourFactory:
    @staticmethod
    def create(
        behaviour_name: str,
        *,
        owner: str,
        value: float,
        confidence: float,
        behaviour_id: str | None = None,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Behaviour",
            owner=owner,
            behaviour_id=behaviour_id or _new_id("beh"),
            behaviour_name=behaviour_name,
            value=value,
            confidence=confidence,
            duration=extra.get("duration", 0.0),
            timestamp=extra.get("timestamp", _iso_now()),
            **{k: v for k, v in extra.items() if k not in {"duration", "timestamp"}},
        )


class RecommendationFactory:
    @staticmethod
    def create(
        recommendation_type: str,
        *,
        owner: str,
        confidence: float,
        recommendation_id: str | None = None,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Recommendation",
            owner=owner,
            recommendation_id=recommendation_id or _new_id("rec"),
            recommendation_type=recommendation_type,
            intervention_type=recommendation_type,  # legacy
            priority=extra.get("priority", "medium"),
            generated_at=extra.get("generated_at", _iso_now()),
            confidence=confidence,
            status=extra.get("status", "pending"),
            timestamp=extra.get("generated_at", _iso_now()),
            **{k: v for k, v in extra.items() if k not in {
                "priority", "generated_at", "status",
            }},
        )


class InterventionFactory:
    @staticmethod
    def create(
        intervention_type: str,
        *,
        owner: str,
        intervention_id: str | None = None,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Intervention",
            owner=owner,
            intervention_id=intervention_id or _new_id("int"),
            type=intervention_type,
            status=extra.get("status", "scheduled"),
            start_time=extra.get("start_time", _iso_now()),
            end_time=extra.get("end_time"),
            completion_status=extra.get("completion_status", "pending"),
            **{k: v for k, v in extra.items() if k not in {
                "status", "start_time", "end_time", "completion_status",
            }},
        )


class EmbeddingFactory:
    @staticmethod
    def create(
        embedding_id: str,
        *,
        owner: str,
        dimension: int,
        embedding_type: str = "multimodal",
        storage_uri: str = "",
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Embedding",
            owner=owner,
            embedding_id=embedding_id,
            embedding_type=embedding_type,
            dimension=dimension,
            model_name=extra.get("model_name", "default"),
            model_version=extra.get("model_version", "emb_v1"),
            storage_uri=storage_uri,
            checksum=extra.get("checksum", ""),
            created_at=extra.get("created_at", _iso_now()),
            **{k: v for k, v in extra.items() if k not in {
                "model_name", "model_version", "checksum", "created_at",
            }},
        )


class TemporalEventFactory:
    @staticmethod
    def create(
        event_type: str,
        *,
        owner: str,
        event_id: str | None = None,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "TemporalEvent",
            owner=owner,
            event_id=event_id or _new_id("evt"),
            event_type=event_type,
            timestamp=extra.get("timestamp", _iso_now()),
            relative_time=extra.get("relative_time", 0.0),
            absolute_time=extra.get("absolute_time", _iso_now()),
            duration=extra.get("duration", 0.0),
            **{k: v for k, v in extra.items() if k not in {
                "timestamp", "relative_time", "absolute_time", "duration",
            }},
        )


class ModelFactory:
    @staticmethod
    def create(
        model_id: str,
        *,
        owner: str,
        architecture: str,
        version: str,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Model",
            owner=owner,
            model_id=model_id,
            architecture=architecture,
            version=version,
            training_dataset=extra.get("training_dataset", "daic"),
            training_date=extra.get("training_date", _iso_now()),
            checkpoint_hash=extra.get("checkpoint_hash", ""),
            **{k: v for k, v in extra.items() if k not in {
                "training_dataset", "training_date", "checkpoint_hash",
            }},
        )


class ResearchFactory:
    @staticmethod
    def create(
        experiment_id: str,
        *,
        owner: str,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            "Research",
            owner=owner,
            experiment_id=experiment_id,
            paper_version=extra.get("paper_version", "1.0"),
            dataset_version=extra.get("dataset_version", "1.0"),
            feature_version=extra.get("feature_version", "1.0"),
            git_commit=extra.get("git_commit", ""),
            pipeline_version=extra.get("pipeline_version", "1.0.0"),
            **{k: v for k, v in extra.items() if k not in {
                "paper_version", "dataset_version", "feature_version",
                "git_commit", "pipeline_version",
            }},
        )


class FeatureFactory:
    """Legacy modality feature factory (MP4 Part 1)."""

    @staticmethod
    def create(
        label: str,
        feature_id: str,
        *,
        owner: str,
        dimension: int = 128,
        quality_score: float = 0.8,
        **extra: Any,
    ) -> GraphEntity:
        return GraphEntity.build(
            label,
            owner=owner,
            feature_id=feature_id,
            dimension=dimension,
            quality_score=quality_score,
            feature_version=extra.get("feature_version", "v1"),
            **{k: v for k, v in extra.items() if k != "feature_version"},
        )
