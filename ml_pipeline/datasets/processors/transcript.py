"""Step 7 — Transcript processing."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ml_pipeline.datasets.config import dataset_paths, load_config
from ml_pipeline.datasets.logging_utils import processing_logger
from ml_pipeline.datasets.parallel import map_parallel
from ml_pipeline.datasets.types import LanguageCode, SampleRecord


def _detect_language(text: str) -> LanguageCode:
    if re.search(r"[\u0B80-\u0BFF]", text):
        return LanguageCode.TAMIL
    if re.search(r"[\u0900-\u097F]", text):
        return LanguageCode.HINDI
    if re.search(r"[a-zA-Z]", text):
        return LanguageCode.ENGLISH
    return LanguageCode.UNKNOWN


def _process_transcript(record: SampleRecord) -> dict:
    cfg = load_config("text.yaml")
    out_cfg = cfg.get("output", {})
    version = out_cfg.get("version", 1)
    paths = dataset_paths()
    cache_dir = paths["cache"] / "transcripts"
    out_dir = paths["processed"] / out_cfg.get("processed_subdir", "text_embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = ""
    if record.file_path.suffix.lower() in {".txt", ".transcript"}:
        text = record.file_path.read_text(encoding="utf-8", errors="replace")
    elif record.file_path.suffix.lower() == ".npy":
        text = ""

    language = _detect_language(text) if text else record.language
    words = len(text.split()) if text else 0
    chars = len(text)

    cache_path = cache_dir / f"{record.participant_id}_transcript_v{version}.txt"
    if text:
        cache_path.write_text(text, encoding="utf-8")

    pattern = out_cfg.get("naming_pattern", "{participant_id}_text_v{version}.npy")
    out_path = out_dir / pattern.format(participant_id=record.participant_id, version=version)
    np.save(out_path, np.array([words, chars], dtype=np.float32))

    return {
        "sample_id": record.sample_id,
        "output": str(out_path),
        "language": language.value,
        "word_count": words,
    }


def run_transcript_processing(records: list[SampleRecord]) -> list[dict]:
    text_records = [r for r in records if r.modality == "text"]
    results, failures = map_parallel(_process_transcript, text_records, "transcript_processing")
    processing_logger.info("transcript_complete processed=%d failed=%d", len(results), len(failures))
    return results
