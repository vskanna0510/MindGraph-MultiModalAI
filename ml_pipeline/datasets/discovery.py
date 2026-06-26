"""Step 1 — Dataset discovery."""

from __future__ import annotations

import csv
from pathlib import Path

from ml_pipeline.datasets import registry
from ml_pipeline.datasets.config import dataset_paths
from ml_pipeline.datasets.logging_utils import dataset_logger
from ml_pipeline.datasets.types import SampleRecord


def discover_all(datasets: list[str] | None = None) -> list[SampleRecord]:
    """Discover samples across enabled dataset adapters."""
    names = datasets or registry.list_adapters()
    records: list[SampleRecord] = []
    for name in names:
        adapter = registry.get_adapter(name)
        records.extend(adapter.discover())
    dataset_logger.info("discovery_complete total=%d datasets=%s", len(records), names)
    return records


def write_dataset_index(records: list[SampleRecord], output: Path | None = None) -> Path:
    """Generate datasets/metadata/dataset_index.csv."""
    paths = dataset_paths()
    out = output or (paths["metadata"] / "dataset_index.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "participant_id",
        "dataset",
        "session_id",
        "modality",
        "file_path",
        "label",
        "language",
        "duration_seconds",
        "split",
        "created_at",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "participant_id": record.participant_id,
                    "dataset": record.dataset,
                    "session_id": record.session_id,
                    "modality": record.modality,
                    "file_path": str(record.file_path),
                    "label": record.label.value,
                    "language": record.language.value,
                    "duration_seconds": record.duration_seconds or "",
                    "split": record.split or "",
                    "created_at": record.created_at,
                }
            )
    dataset_logger.info("dataset_index_written path=%s rows=%d", out, len(records))
    return out


def run_discovery(datasets: list[str] | None = None) -> Path:
    """Execute discovery step and persist index."""
    records = discover_all(datasets)
    return write_dataset_index(records)
