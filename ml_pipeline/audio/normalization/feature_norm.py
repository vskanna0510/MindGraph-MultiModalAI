"""Stage 13 — Feature normalization."""

from __future__ import annotations

import numpy as np

from ml_pipeline.audio.config import audio_config


def normalize_features(features: np.ndarray, method: str | None = None) -> tuple[np.ndarray, dict]:
    """Z-score, min-max, robust, or quantile normalization."""
    method = method or audio_config().get("normalization", {}).get("method", "zscore")
    stats: dict = {}
    if features.size == 0:
        return features, stats
    if method == "minmax":
        vmin, vmax = float(features.min()), float(features.max())
        normed = (features - vmin) / (vmax - vmin + 1e-8)
        stats = {"min": vmin, "max": vmax}
    elif method == "robust":
        med = float(np.median(features))
        iqr = float(np.percentile(features, 75) - np.percentile(features, 25)) + 1e-8
        normed = (features - med) / iqr
        stats = {"median": med, "iqr": iqr}
    elif method == "quantile":
        q_low, q_high = np.percentile(features, [5, 95])
        normed = np.clip((features - q_low) / (q_high - q_low + 1e-8), 0, 1)
        stats = {"q05": float(q_low), "q95": float(q_high)}
    else:
        mean = float(features.mean())
        std = float(features.std()) + 1e-8
        normed = (features - mean) / std
        stats = {"mean": mean, "std": std}
    return normed.astype(np.float32), stats
