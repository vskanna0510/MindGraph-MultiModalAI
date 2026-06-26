"""Outlier and correlation analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.data_quality.config import data_quality_config


def detect_outliers(series: pd.Series, method: str = "zscore") -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().values
    if len(values) < 5:
        return np.zeros(len(values), dtype=bool)
    if method == "zscore":
        z = np.abs((values - np.mean(values)) / (np.std(values) + 1e-8))
        threshold = float(data_quality_config().get("outliers", {}).get("zscore_threshold", 3.0))
        return z > threshold
    try:
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(random_state=42, contamination=0.05)
        preds = model.fit_predict(values.reshape(-1, 1))
        return preds == -1
    except ImportError:
        z = np.abs((values - np.mean(values)) / (np.std(values) + 1e-8))
        return z > 3.0


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="pearson")


def modality_correlation(sessions_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("audio_length", "transcript_length", "quality_score", "sync_score") if c in sessions_df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return correlation_matrix(sessions_df, cols)
