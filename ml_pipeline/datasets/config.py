"""Configuration loader for dataset pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(name: str) -> dict[str, Any]:
    """Load YAML config from configs/ directory."""
    path = project_root() / "configs" / name
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache
def dataset_config() -> dict[str, Any]:
    return load_config("dataset.yaml")


def dataset_paths() -> dict[str, Path]:
    """Resolve standard dataset directory paths."""
    cfg = dataset_config().get("dataset", {})
    env_root = os.environ.get("MINDGRAPH_DATASETS_ROOT")
    root = Path(env_root) if env_root else project_root() / cfg.get("root", "datasets")
    return {
        "root": root,
        "raw": root / "raw" if env_root else project_root() / cfg.get("raw_dir", "datasets/raw"),
        "processed": root / "processed" if env_root else project_root() / cfg.get("processed_dir", "datasets/processed"),
        "cache": root / "cache" if env_root else project_root() / cfg.get("cache_dir", "datasets/cache"),
        "metadata": root / "metadata" if env_root else project_root() / cfg.get("metadata_dir", "datasets/metadata"),
        "exports": root / "exports" if env_root else project_root() / cfg.get("exports_dir", "datasets/exports"),
        "logs": root / "logs" if env_root else project_root() / cfg.get("logs_dir", "datasets/logs"),
        "validation": root / "validation" if env_root else project_root() / cfg.get("validation_dir", "datasets/validation"),
        "splits": root / "splits" if env_root else project_root() / cfg.get("splits_dir", "datasets/splits"),
    }


def label_mapping() -> dict[str, int]:
    """Return label class to integer mapping from config."""
    return dataset_config().get("labels", {}).get("mapping", {})
