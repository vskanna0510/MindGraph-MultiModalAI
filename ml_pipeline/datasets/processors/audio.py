"""Step 5 — Audio processing."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from ml_pipeline.datasets.config import dataset_paths, load_config
from ml_pipeline.datasets.logging_utils import processing_logger
from ml_pipeline.datasets.parallel import map_parallel
from ml_pipeline.datasets.types import SampleRecord


def _process_audio_sample(record: SampleRecord) -> dict:
    cfg = load_config("audio.yaml")
    out_cfg = cfg.get("output", {})
    version = out_cfg.get("version", 1)
    paths = dataset_paths()
    out_dir = paths["processed"] / out_cfg.get("processed_subdir", "audio_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = out_cfg.get("naming_pattern", "{participant_id}_audio_v{version}.npy")
    out_name = pattern.format(participant_id=record.participant_id, version=version)
    out_path = out_dir / out_name

    features: list[float] = []
    sample_rate = 0
    duration = 0.0
    path = record.file_path
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            duration = frames / float(sample_rate) if sample_rate else 0.0
            features = [float(sample_rate), duration, float(wav.getnchannels())]
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
        features = [float(arr.size), float(np.mean(arr)), float(np.std(arr))]
        duration = record.duration_seconds or 0.0
    else:
        features = [float(path.stat().st_size)]
        duration = record.duration_seconds or 0.0

    np.save(out_path, np.array(features, dtype=np.float32))
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps({"sample_id": record.sample_id, "duration": duration, "sample_rate": sample_rate}),
        encoding="utf-8",
    )
    return {"sample_id": record.sample_id, "output": str(out_path), "duration": duration}


def run_audio_processing(records: list[SampleRecord]) -> list[dict]:
    audio_records = [r for r in records if r.modality == "audio"]
    results, failures = map_parallel(_process_audio_sample, audio_records, "audio_processing")
    processing_logger.info("audio_complete processed=%d failed=%d", len(results), len(failures))
    return results
