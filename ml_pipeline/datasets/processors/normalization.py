"""Step 10 — Normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_pipeline.datasets.config import dataset_paths
from ml_pipeline.datasets.logging_utils import processing_logger
from ml_pipeline.datasets.types import SampleRecord


def run_normalization(records: list[SampleRecord]) -> dict:
    """Z-score normalize fusion features per participant batch."""
    paths = dataset_paths()
    fusion_dir = paths["processed"] / "fusion"
    norm_dir = paths["processed"] / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    feature_files = list(fusion_dir.glob("*_features_v1.npy"))
    if not feature_files:
        processing_logger.warning("normalization_skipped reason=no_features")
        return {"normalized": 0}

    arrays = [np.load(path) for path in feature_files]
    stacked = np.vstack([a.reshape(1, -1) if a.ndim == 1 else a for a in arrays])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-8

    count = 0
    for path, arr in zip(feature_files, arrays, strict=False):
        normalized = (arr - mean) / std
        participant_id = path.name.split("_")[0]
        out_path = norm_dir / f"{participant_id}_normalized_v1.npy"
        np.save(out_path, normalized)
        count += 1

    stats_path = norm_dir / "normalization_stats.json"
    stats_path.write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist(), "count": count}),
        encoding="utf-8",
    )
    processing_logger.info("normalization_complete count=%d", count)
    return {"normalized": count, "stats": str(stats_path)}
