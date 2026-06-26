"""Evaluation configuration."""

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
def evaluation_config() -> dict[str, Any]:
    pipe = _load(ROOT / "configs" / "evaluation_pipeline.yaml")
    ml = _load(ROOT / "configs" / "ml" / "evaluation.yaml")
    model_eval = _load(ROOT / "configs" / "model" / "evaluation.yaml")
    return {
        "evaluation": {**ml.get("evaluation", {}), **pipe.get("evaluation", {})},
        "metrics": pipe.get("metrics", ml.get("evaluation", {}).get("metrics", {})),
        "ablation": {**ml.get("comparison", {}), **pipe.get("ablation", {})},
        "reports": {**ml.get("reports", {}), **pipe.get("reports", {})},
        "publication": pipe.get("publication", {}),
        "comparison": ml.get("comparison", {}),
    }


def evaluation_paths() -> dict[str, Path]:
    cfg = evaluation_config()
    root = ROOT / "reports" / "evaluation"
    return {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "exports": root / "exports",
        "calibration": root / "calibration",
    }
