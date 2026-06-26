"""Data quality engine configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_data_quality_config() -> dict[str, Any]:
    path = project_root() / "configs" / "data_quality.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache
def data_quality_config() -> dict[str, Any]:
    return load_data_quality_config()


def data_quality_paths() -> dict[str, Path]:
    root = project_root()
    cfg = data_quality_config()
    paths = cfg.get("paths", {})
    reports = root / paths.get("reports_root", "datasets/reports/data_quality")
    return {
        "reports": reports,
        "figures": reports / paths.get("figures_dir", "figures").split("/")[-1],
        "audit_log": root / paths.get("audit_log", "datasets/reports/data_quality/audit_log.jsonl"),
        "passport": root / paths.get("passport_file", "datasets/reports/data_quality/dataset_passport.json"),
        "ci_report": root / cfg.get("ci", {}).get("machine_report", "datasets/reports/data_quality/ci_validation_report.json"),
        "dataset_index": root / "datasets/metadata/dataset_index.csv",
        "feature_sessions": root / "datasets/processed/feature_store/metadata/sessions.csv",
    }
