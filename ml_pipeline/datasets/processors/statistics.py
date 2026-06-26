"""Step 11 — Statistics generation."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ml_pipeline.datasets.config import dataset_paths
from ml_pipeline.datasets.logging_utils import processing_logger
from ml_pipeline.datasets.types import SampleRecord


def run_statistics(records: list[SampleRecord]) -> Path:
    paths = dataset_paths()
    stats_dir = paths["root"] / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_samples": len(records),
        "participants": len({r.participant_id for r in records}),
        "by_dataset": dict(Counter(r.dataset for r in records)),
        "by_modality": dict(Counter(r.modality for r in records)),
        "by_label": dict(Counter(r.label.value for r in records)),
        "by_language": dict(Counter(r.language.value for r in records)),
        "by_split": dict(Counter(r.split or "unassigned" for r in records)),
    }

    out = stats_dir / "dataset_statistics.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    processing_logger.info("statistics_written path=%s", out)
    return out
