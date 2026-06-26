"""Dataset pipeline domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path


class LabelClass(str, Enum):
    """Normalized depression screening labels."""

    DEPRESSION = "depression"
    NORMAL = "normal"
    UNKNOWN = "unknown"
    EXCLUDED = "excluded"


class LanguageCode(str, Enum):
    """Detected or declared sample language."""

    ENGLISH = "en"
    TAMIL = "ta"
    HINDI = "hi"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class SampleRecord:
    """Single discoverable dataset sample."""

    sample_id: str
    participant_id: str
    dataset: str
    session_id: str
    modality: str
    file_path: Path
    label: LabelClass = LabelClass.UNKNOWN
    language: LanguageCode = LanguageCode.UNKNOWN
    duration_seconds: float | None = None
    split: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class QualityScores:
    """Per-sample quality assessment."""

    overall: float
    audio: float | None = None
    video: float | None = None
    transcript: float | None = None
    synchronization: float | None = None
    missing_value: float | None = None
    corruption: float | None = None
    confidence: float | None = None


@dataclass
class ProcessingProvenance:
    """Reproducibility metadata attached to every processed artifact."""

    dataset_version: str
    processing_version: str
    git_commit: str
    created_at: str
    updated_at: str
    config_hash: str
