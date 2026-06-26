"""Feature store domain types — immutable session model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class FeatureType(str, Enum):
    AUDIO = "audio"
    VISUAL = "visual"
    TEXT = "text"
    IMAGE = "image"
    GRAPH = "graph"
    METADATA = "metadata"
    CLINICAL = "clinical"


class QualityCategory(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


@dataclass(frozen=True)
class FeatureRecord:
    """Traceable feature artifact — never anonymous tensors."""

    feature_id: str
    participant_id: str
    session_id: str
    feature_type: FeatureType
    feature_version: str
    extraction_version: str
    model_version: str
    dataset: str
    language: str
    split: str
    embedding_shape: tuple[int, ...]
    embedding_dim: int
    normalization_method: str
    storage_path: Path
    checksum: str
    quality_score: float
    extraction_time: str
    processing_duration_ms: float
    cache_status: str = "miss"
    storage_format: str = "numpy"


@dataclass(frozen=True)
class QualityMetrics:
    audio: float = 0.0
    video: float = 0.0
    text: float = 0.0
    synchronization: float = 0.0
    embedding: float = 0.0
    metadata: float = 0.0
    overall: float = 0.0
    category: QualityCategory = QualityCategory.ACCEPTABLE


@dataclass(frozen=True)
class BaseSession:
    participant_id: str
    session_id: str
    dataset: str
    split: str
    language: str
    label: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class AudioSession(BaseSession):
    audio_path: Path | None = None
    audio_feature: FeatureRecord | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class VisualSession(BaseSession):
    visual_path: Path | None = None
    visual_feature: FeatureRecord | None = None
    frame_count: int = 0


@dataclass(frozen=True)
class TextSession(BaseSession):
    transcript_path: Path | None = None
    text_feature: FeatureRecord | None = None
    transcript_length: int = 0


@dataclass(frozen=True)
class MultimodalSession(BaseSession):
    """Primary training unit — aggregates all modalities."""

    audio: FeatureRecord | None = None
    visual: FeatureRecord | None = None
    text: FeatureRecord | None = None
    image: FeatureRecord | None = None
    graph: FeatureRecord | None = None
    clinical: dict[str, Any] = field(default_factory=dict)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    modality_mask: dict[str, bool] = field(default_factory=dict)
    sync_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "dataset": self.dataset,
            "split": self.split,
            "language": self.language,
            "label": self.label,
            "modality_mask": self.modality_mask,
            "quality": {
                "overall": self.quality.overall,
                "category": self.quality.category.value,
            },
            "sync_score": self.sync_score,
        }


@dataclass(frozen=True)
class GraphSession(MultimodalSession):
  graph_refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SessionBatch:
    """Loaded tensors for a single session."""

    participant_id: str
    session_id: str
    audio: np.ndarray | None
    visual: np.ndarray | None
    text: np.ndarray | None
    image: np.ndarray | None
    graph: np.ndarray | None
    clinical: dict[str, Any]
    label: int | str
    quality: QualityMetrics
    language: str
    split: str
    timestamp: str
    modality_mask: dict[str, bool]


@dataclass
class FeatureRegistryEntry:
    name: str
    version: str
    owner: str
    description: str
    dependencies: list[str]
    extraction_pipeline: str
    storage_location: str
    validation_status: str = "pending"
    deprecated: bool = False


@dataclass
class ReproducibilityManifest:
    dataset_version: str
    feature_version: str
    git_commit: str
    random_seed: int
    config_hash: str
    processing_timestamp: str
    pipeline_version: str
    content_hash: str
