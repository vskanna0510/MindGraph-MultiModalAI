"""Step 6 — Video processing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_pipeline.datasets.config import dataset_paths, load_config
from ml_pipeline.datasets.logging_utils import processing_logger
from ml_pipeline.datasets.parallel import map_parallel
from ml_pipeline.datasets.types import SampleRecord


def _process_video_sample(record: SampleRecord) -> dict:
    cfg = load_config("video.yaml")
    out_cfg = cfg.get("output", {})
    version = out_cfg.get("version", 1)
    paths = dataset_paths()
    out_dir = paths["processed"] / out_cfg.get("processed_subdir", "visual_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = out_cfg.get("naming_pattern", "{participant_id}_visual_v{version}.npy")
    out_name = pattern.format(participant_id=record.participant_id, version=version)
    out_path = out_dir / out_name

    path = record.file_path
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        features = np.array([arr.size, np.mean(arr), np.std(arr)], dtype=np.float32)
    else:
        features = np.array([path.stat().st_size], dtype=np.float32)

    np.save(out_path, features)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps({"sample_id": record.sample_id, "frames": int(features[0])}), encoding="utf-8")
    return {"sample_id": record.sample_id, "output": str(out_path)}


def run_video_processing(records: list[SampleRecord]) -> list[dict]:
    video_records = [r for r in records if r.modality == "video"]
    results, failures = map_parallel(_process_video_sample, video_records, "video_processing")
    processing_logger.info("video_complete processed=%d failed=%d", len(results), len(failures))
    return results
