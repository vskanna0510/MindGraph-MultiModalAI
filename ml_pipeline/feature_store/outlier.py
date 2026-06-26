"""Outlier detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def detect_outliers_zscore(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    if len(values) < 3:
        return np.zeros(len(values), dtype=bool)
    z = np.abs((values - np.mean(values)) / (np.std(values) + 1e-8))
    return z > threshold


def detect_embedding_outliers(embeddings: list[np.ndarray], method: str = "zscore", threshold: float = 3.0) -> list[bool]:
    if not embeddings:
        return []
    norms = np.array([float(np.linalg.norm(e.ravel())) for e in embeddings])
    if method == "zscore":
        return detect_outliers_zscore(norms, threshold).tolist()
    try:
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(random_state=42, contamination=0.05)
        preds = model.fit_predict(norms.reshape(-1, 1))
        return (preds == -1).tolist()
    except ImportError:
        return detect_outliers_zscore(norms, threshold).tolist()


def write_outlier_report(outliers: list[dict[str, Any]], output_path: Path) -> Path:
    lines = ["# Outlier Report\n", f"Total outliers: {len(outliers)}\n\n"]
    for o in outliers:
        lines.append(f"- {o.get('participant_id')}: {o.get('reason', 'unknown')}\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
