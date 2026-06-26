"""Reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess

from ml_pipeline.data_quality.config import data_quality_config


def get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_hash() -> str:
    return hashlib.sha256(json.dumps(data_quality_config(), sort_keys=True, default=str).encode()).hexdigest()[:16]
