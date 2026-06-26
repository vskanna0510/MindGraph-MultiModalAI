"""Structured logging for dataset pipeline operations."""

from __future__ import annotations

import logging
from pathlib import Path

from ml_pipeline.datasets.config import dataset_paths


def _build_logger(name: str, filename: str) -> logging.Logger:
    paths = dataset_paths()
    log_dir = paths["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"mindgraph.datasets.{name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


dataset_logger = _build_logger("dataset", "dataset.log")
validation_logger = _build_logger("validation", "validation.log")
processing_logger = _build_logger("processing", "processing.log")
quality_logger = _build_logger("quality", "quality.log")
