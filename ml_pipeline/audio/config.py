"""Audio pipeline configuration loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_audio_config() -> dict[str, Any]:
    path = project_root() / "configs" / "audio_pipeline.yaml"
    base = project_root() / "configs" / "audio.yaml"
    cfg: dict[str, Any] = {}
    if base.exists():
        with base.open(encoding="utf-8") as handle:
            cfg.update(yaml.safe_load(handle) or {})
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            pipeline_cfg = yaml.safe_load(handle) or {}
            cfg.update(pipeline_cfg)
    return cfg


@lru_cache
def audio_config() -> dict[str, Any]:
    return load_audio_config()


def audio_paths() -> dict[str, Path]:
    cfg = audio_config()
    root = project_root()
    paths_cfg = cfg.get("paths", {})
    cache_cfg = cfg.get("cache", {})
    return {
        "processed": root / paths_cfg.get("processed_root", "datasets/processed/audio_features"),
        "reports": root / paths_cfg.get("reports_root", "datasets/reports/audio"),
        "cache": root / cache_cfg.get("root", "datasets/cache"),
    }
