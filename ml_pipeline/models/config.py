"""Model configuration loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_model_config() -> dict[str, Any]:
    root = project_root() / "configs" / "model"
    cfg: dict[str, Any] = {}
    for name in (
        "architecture",
        "audio",
        "visual",
        "text",
        "image",
        "fusion",
        "graph",
        "training",
        "evaluation",
        "deployment",
    ):
        data = load_yaml(root / f"{name}.yaml")
        if name == "architecture":
            cfg.update(data)
        else:
            cfg[name] = data
    return cfg


@lru_cache
def model_config() -> dict[str, Any]:
    return load_model_config()


def model_paths() -> dict[str, Path]:
    root = project_root()
    cfg = model_config()
    return {
        "weights": root / cfg.get("checkpoint", {}).get("directory", "ml_pipeline/weights"),
        "experiments": root / "ml_pipeline/experiments",
        "registry": root / "ml_pipeline/models/registry.json",
    }
