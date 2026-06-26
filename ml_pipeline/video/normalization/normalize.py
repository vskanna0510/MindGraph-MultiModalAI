"""Video feature normalization."""

from __future__ import annotations

import numpy as np

from ml_pipeline.video.config import video_config


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    if landmarks.size == 0:
        return landmarks
    mean = landmarks.mean(axis=0, keepdims=True)
    std = landmarks.std(axis=0, keepdims=True) + 1e-8
    return ((landmarks - mean) / std).astype(np.float32)


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    method = video_config().get("normalization", {}).get("embedding_method", "zscore")
    if method == "minmax":
        return ((vector - vector.min()) / (vector.ptp() + 1e-8)).astype(np.float32)
    std = float(vector.std()) + 1e-8
    return ((vector - vector.mean()) / std).astype(np.float32)
