"""Step 8 — Feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_pipeline.datasets.config import dataset_paths
from ml_pipeline.datasets.logging_utils import processing_logger
from ml_pipeline.datasets.parallel import map_parallel
from ml_pipeline.datasets.types import SampleRecord


def _extract_features(record: SampleRecord) -> dict:
    paths = dataset_paths()
    fusion_dir = paths["processed"] / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    modality_dirs = {
        "audio": paths["processed"] / "audio_features",
        "video": paths["processed"] / "visual_features",
        "text": paths["processed"] / "text_embeddings",
    }
    features: list[np.ndarray] = []
    for modality, directory in modality_dirs.items():
        if record.modality != modality:
            continue
        for npy in directory.glob(f"{record.participant_id}_*.npy"):
            features.append(np.load(npy))

    if not features and record.file_path.suffix == ".npy":
        features.append(np.load(record.file_path).astype(np.float32).ravel()[:64])

    if not features:
        features.append(np.zeros(8, dtype=np.float32))

    fused = np.concatenate([f.ravel() for f in features])
    out_path = fusion_dir / f"{record.participant_id}_features_v1.npy"
    np.save(out_path, fused)
    return {"sample_id": record.sample_id, "output": str(out_path), "dim": int(fused.size)}


def run_feature_extraction(records: list[SampleRecord]) -> list[dict]:
    by_participant: dict[str, SampleRecord] = {}
    for record in records:
        by_participant.setdefault(record.participant_id, record)
    items = list(by_participant.values())
    results, failures = map_parallel(_extract_features, items, "feature_extraction")
    processing_logger.info("features_complete processed=%d failed=%d", len(results), len(failures))
    return results
