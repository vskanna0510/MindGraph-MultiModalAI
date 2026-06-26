"""Reproducibility manifest generation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime

from ml_pipeline.feature_store.config import feature_store_config
from ml_pipeline.feature_store.types import ReproducibilityManifest


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_hash(cfg: dict | None = None) -> str:
    cfg = cfg or feature_store_config()
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_reproducibility_manifest(content_hash: str = "") -> ReproducibilityManifest:
    cfg = feature_store_config()
    fs = cfg.get("feature_store", {})
    repro = cfg.get("reproducibility", {})
    return ReproducibilityManifest(
        dataset_version=fs.get("dataset_version", "v1"),
        feature_version=fs.get("version", "1.0.0"),
        git_commit=get_git_commit() if repro.get("record_git_commit", True) else "n/a",
        random_seed=int(cfg.get("sampling", {}).get("seed", 42)),
        config_hash=config_hash() if repro.get("record_config_hash", True) else "n/a",
        processing_timestamp=datetime.now(UTC).isoformat(),
        pipeline_version=fs.get("version", "1.0.0"),
        content_hash=content_hash or config_hash(),
    )
