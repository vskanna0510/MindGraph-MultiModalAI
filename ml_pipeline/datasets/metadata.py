"""Step 3 — Metadata generation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ml_pipeline.datasets.config import dataset_paths
from ml_pipeline.datasets.logging_utils import dataset_logger
from ml_pipeline.datasets.types import SampleRecord


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_metadata(records: list[SampleRecord]) -> dict[str, Path]:
    """Generate all metadata CSV files from discovered records."""
    paths = dataset_paths()
    meta_dir = paths["metadata"]
    participants: dict[str, dict] = {}
    sessions: dict[str, dict] = {}
    by_modality: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        participants.setdefault(
            record.participant_id,
            {
                "participant_id": record.participant_id,
                "dataset": record.dataset,
                "label": record.label.value,
                "language": record.language.value,
            },
        )
        sessions.setdefault(
            record.session_id,
            {
                "session_id": record.session_id,
                "participant_id": record.participant_id,
                "dataset": record.dataset,
            },
        )
        by_modality[record.modality].append(
            {
                "sample_id": record.sample_id,
                "participant_id": record.participant_id,
                "file_path": str(record.file_path),
                "duration_seconds": record.duration_seconds or "",
            }
        )

    outputs: dict[str, Path] = {}
    outputs["participants"] = meta_dir / "participants.csv"
    _write_csv(
        outputs["participants"],
        ["participant_id", "dataset", "label", "language"],
        list(participants.values()),
    )

    outputs["sessions"] = meta_dir / "sessions.csv"
    _write_csv(
        outputs["sessions"],
        ["session_id", "participant_id", "dataset"],
        list(sessions.values()),
    )

    modality_files = {
        "audio": "audio.csv",
        "video": "video.csv",
        "text": "text.csv",
        "image": "image.csv",
    }
    for modality, filename in modality_files.items():
        if modality in by_modality:
            path = meta_dir / filename
            _write_csv(
                path,
                ["sample_id", "participant_id", "file_path", "duration_seconds"],
                by_modality[modality],
            )
            outputs[modality] = path

    labels_rows = [
        {"participant_id": p["participant_id"], "label": p["label"]}
        for p in participants.values()
    ]
    outputs["labels"] = meta_dir / "labels.csv"
    _write_csv(outputs["labels"], ["participant_id", "label"], labels_rows)

    lang_rows = [
        {"participant_id": p["participant_id"], "language": p["language"]}
        for p in participants.values()
    ]
    outputs["languages"] = meta_dir / "languages.csv"
    _write_csv(outputs["languages"], ["participant_id", "language"], lang_rows)

    dataset_logger.info("metadata_generated files=%d", len(outputs))
    return outputs


def run_metadata(records: list[SampleRecord]) -> dict[str, Path]:
    return generate_metadata(records)
