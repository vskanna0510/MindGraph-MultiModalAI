"""Digital twin profile dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class BehaviourProfile:
    speech_rate: float = 0.0
    voice_energy: float = 0.0
    pause_behaviour: float = 0.0
    eye_contact: float = 0.0
    blink_rate: float = 0.0
    head_movement: float = 0.0
    facial_activity: float = 0.0
    conversation_length: float = 0.0
    response_delay: float = 0.0
    interaction_frequency: float = 0.0
    session_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class EmotionProfile:
    average_emotion_score: float = 0.0
    variability: float = 0.0
    stability: float = 1.0
    dominant_emotion: str = "Neutral"
    rare_emotions: list[str] = field(default_factory=list)
    weekly_trend: str = "stable"
    monthly_trend: str = "stable"
    recovery_rate: float = 0.0
    escalation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class SymptomProfile:
    frequency: dict[str, int] = field(default_factory=dict)
    persistence: dict[str, int] = field(default_factory=dict)
    severity_avg: dict[str, float] = field(default_factory=dict)
    co_occurrence: list[tuple[str, str]] = field(default_factory=list)
    improving: list[str] = field(default_factory=list)
    recurring: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency": self.frequency,
            "persistence": self.persistence,
            "severity_avg": self.severity_avg,
            "co_occurrence": [list(p) for p in self.co_occurrence],
            "improving": self.improving,
            "recurring": self.recurring,
        }


@dataclass
class RecoveryProfile:
    recovery_rate: float = 0.0
    stability: float = 0.0
    improvement_speed: float = 0.0
    recommendation_compliance: float = 0.0
    risk_reduction: float = 0.0
    positive_behaviour_changes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class LanguageProfile:
    preferred_language: str = "en"
    language_switching_count: int = 0
    vocabulary_complexity: float = 0.0
    sentence_length_avg: float = 0.0
    negative_language_frequency: float = 0.0
    positive_language_frequency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class InteractionProfile:
    total_sessions: int = 0
    avg_session_duration: float = 0.0
    engagement_score: float = 0.0
    offline_sessions: int = 0
    edge_sessions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
