"""Feature validation for video pipeline."""

from __future__ import annotations

import numpy as np

from ml_pipeline.video.types import FacialFeatures, LandmarkFrame, VisualEmbedding


def validate_visual_pipeline(
    landmarks: list[LandmarkFrame],
    features: FacialFeatures | None,
    embedding: VisualEmbedding | None,
    expected_landmarks: int = 468,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for lm in landmarks:
        if lm.landmarks.shape[0] != expected_landmarks:
            errors.append(f"landmark_count_{lm.frame_index}")
        if np.isnan(lm.landmarks).any():
            errors.append(f"landmark_nan_{lm.frame_index}")
    if features and len(features.ear) == 0:
        errors.append("empty_features")
    if embedding and not embedding.vector_path.exists():
        errors.append("embedding_missing")
    return len(errors) == 0, errors


def write_validation_report(issues: list[dict], output) -> None:
    lines = ["# Video Validation Report", "", f"Issues: {len(issues)}", ""]
    for item in issues:
        lines.append(f"- {item.get('participant_id')}: {item.get('errors')}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
