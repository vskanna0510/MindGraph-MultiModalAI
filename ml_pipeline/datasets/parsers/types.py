"""Parser domain types for dataset engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_pipeline.datasets.types import LabelClass, LanguageCode


@dataclass
class FileInfo:
    """Discovered file with modality classification."""

    path: Path
    modality: str
    exists: bool = True
    readable: bool = True
    size_bytes: int = 0
    validation_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Utterance:
    """Single transcript utterance."""

    speaker: str
    start_time: float
    stop_time: float
    text: str
    order: int
    duration: float = 0.0


@dataclass
class TranscriptData:
    """Parsed and cleaned transcript."""

    utterances: list[Utterance]
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    vocabulary_size: int = 0
    unknown_tokens: int = 0
    quality_score: float = 0.0
    json_path: Path | None = None
    csv_path: Path | None = None


@dataclass
class ParticipantMetadata:
    """Per-participant extracted metadata."""

    participant_id: str
    session_id: str
    dataset: str
    gender: str | None = None
    age: int | None = None
    interview_duration: float | None = None
    transcript_length: int = 0
    audio_length: float | None = None
    video_length: float | None = None
    frame_count: int | None = None
    sample_rate: int | None = None
    label: LabelClass = LabelClass.UNKNOWN
    split: str | None = None
    language: LanguageCode = LanguageCode.ENGLISH
    missing_files: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    processing_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    files: dict[str, Path | None] = field(default_factory=dict)
    sync_offset: float | None = None
    sync_score: float | None = None
    corruption_flags: list[str] = field(default_factory=list)


@dataclass
class ValidationIssue:
    """Validation issue with severity."""

    participant_id: str
    check: str
    severity: str
    message: str


@dataclass
class SplitInfo:
    """Official split assignment."""

    participant_id: str
    split: str
    label: LabelClass
    gender: str | None = None
    source_file: str = ""


@dataclass
class ParserResult:
    """Complete parser output."""

    dataset: str
    participants: list[ParticipantMetadata]
    samples: list[dict[str, Any]]
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    split_records: list[SplitInfo] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
