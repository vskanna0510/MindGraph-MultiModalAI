"""Digital Cognitive Twin — evolving user model."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from graph.twin.profiles import (
    BehaviourProfile,
    EmotionProfile,
    InteractionProfile,
    LanguageProfile,
    RecoveryProfile,
    SymptomProfile,
)


@dataclass
class DigitalTwin:
    """Unique per-user cognitive twin assembled from graph history."""

    user_id: str
    twin_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "2.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    behaviour: BehaviourProfile = field(default_factory=BehaviourProfile)
    emotion: EmotionProfile = field(default_factory=EmotionProfile)
    symptom: SymptomProfile = field(default_factory=SymptomProfile)
    recovery: RecoveryProfile = field(default_factory=RecoveryProfile)
    language: LanguageProfile = field(default_factory=LanguageProfile)
    interaction: InteractionProfile = field(default_factory=InteractionProfile)
    risk_scores: list[float] = field(default_factory=list)
    recommendation_history: list[dict[str, Any]] = field(default_factory=list)
    embedding_refs: list[str] = field(default_factory=list)
    temporal_memory_sessions: int = 0
    model_version: str = "MODEL_010"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "twin_id": self.twin_id,
            "version": self.version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "behaviour_profile": self.behaviour.to_dict(),
            "emotion_profile": self.emotion.to_dict(),
            "symptom_profile": self.symptom.to_dict(),
            "recovery_profile": self.recovery.to_dict(),
            "language_profile": self.language.to_dict(),
            "interaction_profile": self.interaction.to_dict(),
            "risk_history_length": len(self.risk_scores),
            "latest_risk": self.risk_scores[-1] if self.risk_scores else None,
            "recommendation_count": len(self.recommendation_history),
            "embedding_count": len(self.embedding_refs),
            "temporal_memory_sessions": self.temporal_memory_sessions,
            "model_version": self.model_version,
        }


class DigitalTwinBuilder:
    """Build/update digital twin from graph query results."""

    @staticmethod
    def build(
        user_id: str,
        *,
        risk_scores: list[float],
        emotions: list[dict[str, Any]],
        symptoms: list[dict[str, Any]],
        behaviours: list[dict[str, Any]],
        sessions: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
        user_props: dict[str, Any] | None = None,
        existing: DigitalTwin | None = None,
    ) -> DigitalTwin:
        twin = existing or DigitalTwin(user_id=user_id)
        twin.last_updated = datetime.now(UTC).isoformat()
        twin.risk_scores = risk_scores
        twin.recommendation_history = recommendations
        twin.temporal_memory_sessions = len(sessions)

        DigitalTwinBuilder._build_emotion(twin, emotions)
        DigitalTwinBuilder._build_symptom(twin, symptoms)
        DigitalTwinBuilder._build_behaviour(twin, behaviours)
        DigitalTwinBuilder._build_recovery(twin, risk_scores, recommendations)
        DigitalTwinBuilder._build_interaction(twin, sessions)
        DigitalTwinBuilder._build_language(twin, user_props or {}, sessions)
        return twin

    @staticmethod
    def _build_emotion(twin: DigitalTwin, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        probs = [float(r.get("prob", 0)) for r in records]
        twin.emotion.average_emotion_score = round(sum(probs) / len(probs), 4)
        mean = twin.emotion.average_emotion_score
        twin.emotion.variability = round(
            (sum((p - mean) ** 2 for p in probs) / len(probs)) ** 0.5, 4
        )
        twin.emotion.stability = round(max(0.0, 1.0 - twin.emotion.variability), 4)
        emotion_counts: dict[str, int] = {}
        for r in records:
            emo = str(r.get("emotion", "Neutral"))
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        if emotion_counts:
            twin.emotion.dominant_emotion = max(emotion_counts, key=emotion_counts.get)  # type: ignore[arg-type]
            rare = [e for e, c in emotion_counts.items() if c == 1]
            twin.emotion.rare_emotions = rare[:5]

    @staticmethod
    def _build_symptom(twin: DigitalTwin, records: list[dict[str, Any]]) -> None:
        for r in records:
            name = str(r.get("symptom", r.get("symptom_name", "")))
            if not name:
                continue
            twin.symptom.frequency[name] = twin.symptom.frequency.get(name, 0) + 1
            twin.symptom.severity_avg[name] = float(r.get("severity", r.get("conf", 0.5)))
        twin.symptom.recurring = [s for s, c in twin.symptom.frequency.items() if c >= 3]
        twin.symptom.persistence = dict(twin.symptom.frequency)

    @staticmethod
    def _build_behaviour(twin: DigitalTwin, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        values = [float(r.get("value", 0)) for r in records]
        twin.behaviour.facial_activity = round(sum(values) / len(values), 4)
        twin.behaviour.session_count = len(records)
        name_map = {
            "speech rate": "speech_rate",
            "low voice energy": "voice_energy",
            "reduced eye contact": "eye_contact",
            "reduced facial activity": "facial_activity",
            "slow speaking rate": "speech_rate",
            "speech pause": "pause_behaviour",
        }
        for r in records:
            bname = str(r.get("behaviour", r.get("behaviour_name", ""))).lower()
            attr = name_map.get(bname)
            if attr and hasattr(twin.behaviour, attr):
                setattr(twin.behaviour, attr, float(r.get("value", 0.5)))

    @staticmethod
    def _build_recovery(
        twin: DigitalTwin,
        risk_scores: list[float],
        recommendations: list[dict[str, Any]],
    ) -> None:
        if len(risk_scores) >= 2:
            twin.recovery.risk_reduction = round(risk_scores[0] - risk_scores[-1], 4)
            twin.recovery.improvement_speed = round(
                (risk_scores[0] - risk_scores[-1]) / len(risk_scores), 4
            )
        completed = sum(1 for r in recommendations if r.get("status") == "completed")
        twin.recovery.recommendation_compliance = round(
            completed / max(len(recommendations), 1), 4
        )
        twin.recovery.recovery_rate = max(0.0, twin.recovery.risk_reduction)
        twin.recovery.stability = twin.emotion.stability

    @staticmethod
    def _build_interaction(twin: DigitalTwin, sessions: list[dict[str, Any]]) -> None:
        twin.interaction.total_sessions = len(sessions)
        if not sessions:
            return
        durations = [float(s.get("duration", 0)) for s in sessions]
        twin.interaction.avg_session_duration = round(sum(durations) / len(durations), 2)
        twin.interaction.offline_sessions = sum(1 for s in sessions if s.get("offline_mode"))
        twin.interaction.edge_sessions = sum(1 for s in sessions if s.get("edge_processing"))
        twin.interaction.engagement_score = round(min(1.0, len(sessions) / 12), 4)

    @staticmethod
    def _build_language(
        twin: DigitalTwin,
        user_props: dict[str, Any],
        sessions: list[dict[str, Any]],
    ) -> None:
        twin.language.preferred_language = str(
            user_props.get("preferred_language", user_props.get("language", "en"))
        )
        langs = {str(s.get("language", twin.language.preferred_language)) for s in sessions}
        twin.language.language_switching_count = max(0, len(langs) - 1)
