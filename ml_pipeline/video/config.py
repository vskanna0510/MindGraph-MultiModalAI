"""Video pipeline configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_video_config() -> dict[str, Any]:
    root = project_root()
    cfg: dict[str, Any] = {}
    for name in ("video.yaml", "video_pipeline.yaml"):
        path = root / "configs" / name
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                cfg.update(yaml.safe_load(handle) or {})
    return cfg


@lru_cache
def video_config() -> dict[str, Any]:
    return load_video_config()


def video_paths() -> dict[str, Path]:
    cfg = video_config()
    root = project_root()
    paths_cfg = cfg.get("paths", {})
    cache_cfg = cfg.get("cache", {})
    return {
        "processed": root / paths_cfg.get("processed_root", "datasets/processed/visual"),
        "reports": root / paths_cfg.get("reports_root", "datasets/reports/video"),
        "cache": root / cache_cfg.get("root", "datasets/cache"),
    }
