"""Training configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def training_config() -> dict[str, Any]:
    pipeline = _load(ROOT / "configs" / "training_pipeline.yaml")
    top = _load(ROOT / "configs" / "training.yaml")
    model_train = _load(ROOT / "configs" / "model" / "training.yaml")
    return {
        "training": {**top.get("training", {}), **pipeline.get("training", {}), **model_train},
        "loss": pipeline.get("loss", {}),
        "optimization": {**pipeline.get("optimization", {}), **model_train},
        "checkpoint": {**top.get("checkpoints", {}), **pipeline.get("checkpoint", {}), **model_train.get("checkpoint", {})},
        "cross_validation": pipeline.get("cross_validation", {}),
        "hpo": pipeline.get("hpo", {}),
        "logging": pipeline.get("logging", {}),
        "stages": pipeline.get("stages", {}),
        "evaluation": top.get("evaluation", {}),
        "datasets": top.get("datasets", {}),
    }
