"""Video pipeline domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VideoMetadata:
    participant_id: str
    session_id: str
    source_path: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_count: int
    codec: str
    quality_score: float = 0.0
    face_detection_rate: float = 0.0
    landmark_success_rate: float = 0.0
    processing_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class FrameRecord:
    index: int
    timestamp: float
    path: Path | None
    frame_hash: str = ""
    scene_number: int = 0


@dataclass
class FaceDetection:
    frame_index: int
    bbox: tuple[float, float, float, float]
    confidence: float
    face_count: int = 1


@dataclass
class LandmarkFrame:
    frame_index: int
    timestamp: float
    landmarks: np.ndarray  # (N, 3) xyz
    visibility: np.ndarray | None = None
    confidence: float = 1.0


@dataclass
class TrackState:
    track_id: int
    landmarks_sequence: list[LandmarkFrame] = field(default_factory=list)
    dropped_frames: int = 0


@dataclass
class FacialFeatures:
    ear: np.ndarray  # eye aspect ratio per frame
    mar: np.ndarray  # mouth aspect ratio
    blink_rate: float
    smile_intensity: float
    head_pose_yaw: np.ndarray
    head_pose_pitch: np.ndarray
    head_pose_roll: np.ndarray
    symmetry_score: float
    gaze_forward_ratio: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualEmbedding:
    participant_id: str
    session_id: str
    model_name: str
    model_version: str
    embedding_dim: int
    vector_path: Path
    video_hash: str
    config_version: str
    extraction_timestamp: str


@dataclass
class PipelineStageResult:
    stage: str
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class VideoPipelineResult:
    participant_id: str
    source_path: Path
    metadata: VideoMetadata | None = None
    frames: list[FrameRecord] = field(default_factory=list)
    detections: list[FaceDetection] = field(default_factory=list)
    landmarks: list[LandmarkFrame] = field(default_factory=list)
    tracks: list[TrackState] = field(default_factory=list)
    facial_features: FacialFeatures | None = None
    embedding: VisualEmbedding | None = None
    stages: list[PipelineStageResult] = field(default_factory=list)
    validation_passed: bool = True
