"""Feature store configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_feature_store_config() -> dict[str, Any]:
    path = project_root() / "configs" / "feature_store.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache
def feature_store_config() -> dict[str, Any]:
    return load_feature_store_config()


def feature_store_paths() -> dict[str, Path]:
    root = project_root()
    cfg = feature_store_config()
    paths_cfg = cfg.get("paths", {})
    cache_cfg = cfg.get("cache", {})
    source = cfg.get("source_paths", {})
    store_root = root / paths_cfg.get("root", "datasets/processed/feature_store")
    return {
        "root": store_root,
        "audio": store_root / "audio",
        "visual": store_root / "visual",
        "text": store_root / "text",
        "image": store_root / "image",
        "graph": store_root / "graph",
        "metadata": store_root / "metadata",
        "clinical": store_root / "clinical",
        "quality": store_root / "quality",
        "fusion": store_root / "fusion",
        "statistics": store_root / "statistics",
        "cache": root / cache_cfg.get("root", "datasets/cache/feature_store"),
        "exports": store_root / "exports",
        "logs": store_root / "logs",
        "dataset_index": root / source.get("dataset_index", "datasets/metadata/dataset_index.csv"),
        "source_audio": root / source.get("audio_features", "datasets/processed/audio_features"),
        "source_visual": root / source.get("visual_features", "datasets/processed/visual"),
        "source_text": root / source.get("text_features", "datasets/processed/text"),
        "source_transcripts": root / source.get("transcripts", "datasets/processed/transcripts"),
    }
