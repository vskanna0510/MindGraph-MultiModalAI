"""Audio pipeline domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class AudioMetadata:
    """Extracted audio file metadata."""

    participant_id: str
    session_id: str
    source_path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int | None
    codec: str
    bitrate: int | None
    peak_amplitude: float
    rms_energy: float
    silence_ratio: float
    speech_ratio: float
    original_channels: int = 1
    snr_original: float | None = None
    snr_processed: float | None = None
    quality_score: float = 0.0
    processing_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    language: str = "en"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeechSegment:
    """Single speech segment with original timestamps."""

    start_seconds: float
    end_seconds: float
    segment_index: int
    source_start: float
    source_end: float


@dataclass
class VADResult:
    """Voice activity detection output."""

    speech_segments: list[SpeechSegment]
    silence_segments: list[SpeechSegment]
    speech_ratio: float
    avg_segment_duration: float
    method: str


@dataclass
class PauseStatistics:
    """Silence / pause metrics."""

    leading_silence: float
    trailing_silence: float
    internal_pause_count: int
    avg_pause_duration: float
    silence_percentage: float


@dataclass
class FeatureBundle:
    """Extracted acoustic features."""

    participant_id: str
    feature_version: str
    mfcc: Any | None = None
    mel: Any | None = None
    spectral: dict[str, Any] = field(default_factory=dict)
    prosody: dict[str, Any] = field(default_factory=dict)
    pitch: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class EmbeddingRecord:
    """Deep speech embedding artifact."""

    participant_id: str
    session_id: str
    model_name: str
    model_version: str
    embedding_dim: int
    vector_path: Path
    audio_hash: str
    config_version: str
    extraction_timestamp: str
    language: str = "en"
    checksum: str = ""


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage."""

    stage: str
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class AudioPipelineResult:
    """Full audio pipeline output for one recording."""

    participant_id: str
    source_path: Path
    metadata: AudioMetadata | None = None
    processed_path: Path | None = None
    vad: VADResult | None = None
    pauses: PauseStatistics | None = None
    features: FeatureBundle | None = None
    embedding: EmbeddingRecord | None = None
    stages: list[PipelineStageResult] = field(default_factory=list)
    validation_passed: bool = True
