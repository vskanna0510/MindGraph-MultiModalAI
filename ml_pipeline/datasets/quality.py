"""Dataset quality scoring."""

from __future__ import annotations

import csv
from pathlib import Path

from ml_pipeline.datasets.config import dataset_config, dataset_paths
from ml_pipeline.datasets.logging_utils import quality_logger
from ml_pipeline.datasets.types import QualityScores, SampleRecord


def score_sample(record: SampleRecord, validation_errors: int = 0) -> QualityScores:
    """Compute per-sample quality scores (0.0–1.0)."""
    exists = 1.0 if record.file_path.exists() else 0.0
    label_score = 1.0 if record.label.value not in {"unknown", "excluded"} else 0.3
    corruption = max(0.0, 1.0 - validation_errors * 0.25)
    missing = exists

    modality_scores: dict[str, float | None] = {
        "audio": None,
        "video": None,
        "transcript": None,
    }
    if record.modality == "audio":
        modality_scores["audio"] = exists * corruption
    elif record.modality == "video":
        modality_scores["video"] = exists * corruption
    elif record.modality == "text":
        modality_scores["transcript"] = exists * corruption

    overall = (exists + label_score + corruption + missing) / 4.0
    return QualityScores(
        overall=round(overall, 4),
        audio=modality_scores["audio"],
        video=modality_scores["video"],
        transcript=modality_scores["transcript"],
        synchronization=1.0 if exists else 0.0,
        missing_value=missing,
        corruption=corruption,
        confidence=label_score,
    )


def write_quality_csv(records: list[SampleRecord], error_counts: dict[str, int] | None = None) -> Path:
    """Write datasets/metadata/quality.csv."""
    paths = dataset_paths()
    out = paths["metadata"] / "quality.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    error_counts = error_counts or {}
    fieldnames = [
        "sample_id",
        "overall",
        "audio",
        "video",
        "transcript",
        "synchronization",
        "missing_value",
        "corruption",
        "confidence",
    ]
    rows: list[dict] = []
    for record in records:
        scores = score_sample(record, error_counts.get(record.sample_id, 0))
        rows.append(
            {
                "sample_id": record.sample_id,
                "overall": scores.overall,
                "audio": scores.audio if scores.audio is not None else "",
                "video": scores.video if scores.video is not None else "",
                "transcript": scores.transcript if scores.transcript is not None else "",
                "synchronization": scores.synchronization,
                "missing_value": scores.missing_value,
                "corruption": scores.corruption,
                "confidence": scores.confidence,
            }
        )
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cfg = dataset_config().get("quality", {})
    threshold = float(cfg.get("min_overall_score", 0.5))
    low_quality = [r for r in rows if float(r["overall"]) < threshold]
    if low_quality and cfg.get("log_rejected", True):
        quality_logger.warning("low_quality_samples count=%d threshold=%s", len(low_quality), threshold)

    quality_logger.info("quality_scores_written count=%d", len(rows))
    return out


def run_quality(records: list[SampleRecord], error_counts: dict[str, int] | None = None) -> Path:
    return write_quality_csv(records, error_counts)
