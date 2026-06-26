"""Dataset manifest generation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ml_pipeline.datasets.config import dataset_config, dataset_paths
from ml_pipeline.datasets.types import SampleRecord
from ml_pipeline.datasets.versioning import git_commit_hash


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    records: list[SampleRecord],
    missing_files: list[str] | None = None,
    corrupted_files: list[str] | None = None,
) -> dict:
    """Build manifest.json content."""
    cfg = dataset_config().get("dataset", {})
    labels = Counter(r.label.value for r in records)
    languages = Counter(r.language.value for r in records)
    hashes: dict[str, str] = {}
    for record in records[:50]:
        if record.file_path.exists():
            try:
                hashes[record.sample_id] = _file_hash(record.file_path)
            except OSError:
                pass

    return {
        "dataset_version": cfg.get("version", "1.0.0"),
        "processing_version": cfg.get("processing_version", "1.0.0"),
        "sample_count": len(records),
        "class_distribution": dict(labels),
        "languages": dict(languages),
        "missing_files": missing_files or [],
        "corrupted_files": corrupted_files or [],
        "hash_values": hashes,
        "git_commit": git_commit_hash(),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def write_manifest(
    records: list[SampleRecord],
    missing_files: list[str] | None = None,
    corrupted_files: list[str] | None = None,
) -> Path:
    """Write datasets/metadata/manifest.json."""
    paths = dataset_paths()
    manifest = build_manifest(records, missing_files, corrupted_files)
    filename = dataset_config().get("manifest", {}).get("filename", "manifest.json")
    out = paths["metadata"] / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return out
