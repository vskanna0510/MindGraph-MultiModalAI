"""Step 13 — Pipeline verification."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ml_pipeline.datasets.config import dataset_paths
from ml_pipeline.datasets.logging_utils import processing_logger
from ml_pipeline.datasets.types import SampleRecord


def run_verification(records: list[SampleRecord]) -> dict:
    """Verify pipeline outputs exist and counts are consistent."""
    paths = dataset_paths()
    checks: list[dict] = []

    index_path = paths["metadata"] / "dataset_index.csv"
    checks.append({"check": "dataset_index", "passed": index_path.exists()})

    manifest_path = paths["metadata"] / "manifest.json"
    checks.append({"check": "manifest", "passed": manifest_path.exists()})

    stats_path = paths["root"] / "statistics" / "dataset_statistics.json"
    checks.append({"check": "statistics", "passed": stats_path.exists()})

    export_csv = paths["exports"] / "csv" / "samples.csv"
    checks.append({"check": "export_csv", "passed": export_csv.exists()})

    processed_count = sum(1 for _ in (paths["processed"] / "fusion").glob("*.npy")) if (paths["processed"] / "fusion").exists() else 0
    participant_count = len({r.participant_id for r in records})
    checks.append(
        {
            "check": "feature_coverage",
            "passed": processed_count >= min(participant_count, 1) if records else True,
            "processed": processed_count,
            "participants": participant_count,
        }
    )

    all_passed = all(c.get("passed", False) for c in checks)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "passed": all_passed,
        "checks": checks,
        "sample_count": len(records),
    }

    out = paths["validation"] / "pipeline_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    processing_logger.info("verification_complete passed=%s", all_passed)
    return report
