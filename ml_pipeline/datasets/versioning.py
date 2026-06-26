"""Dataset versioning and provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ml_pipeline.datasets.config import dataset_config, dataset_paths


def git_commit_hash() -> str:
    """Return current git commit or 'unknown'."""
    if not dataset_config().get("versioning", {}).get("track_git_commit", True):
        return "disabled"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def directory_hash(path: Path, max_files: int = 100) -> str:
    """Compute aggregate hash of directory file listing (not content)."""
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    files = sorted(path.rglob("*"))[:max_files]
    for file_path in files:
        if file_path.is_file():
            digest.update(str(file_path.relative_to(path)).encode())
            digest.update(str(file_path.stat().st_size).encode())
    return digest.hexdigest()


def build_version_record(dataset_name: str, source_path: Path) -> dict:
    """Build versioning metadata for a dataset source."""
    cfg = dataset_config().get("dataset", {})
    return {
        "version": cfg.get("version", "1.0.0"),
        "processing_version": cfg.get("processing_version", "1.0.0"),
        "creation_date": datetime.now(UTC).isoformat(),
        "dataset_source": dataset_name,
        "dataset_hash": directory_hash(source_path) if dataset_config().get("versioning", {}).get("compute_hashes") else "skipped",
        "git_commit": git_commit_hash(),
        "validation_status": "pending",
        "modification_history": [],
    }


def write_version_manifest(dataset_name: str) -> Path:
    """Persist version record under datasets/metadata/."""
    paths = dataset_paths()
    source = paths["raw"] / dataset_name
    record = build_version_record(dataset_name, source)
    out = paths["metadata"] / f"{dataset_name}_version.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return out
