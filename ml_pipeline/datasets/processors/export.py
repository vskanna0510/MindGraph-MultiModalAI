"""Step 12 — Export."""

from __future__ import annotations

import csv
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ml_pipeline.datasets.config import dataset_paths
from ml_pipeline.datasets.logging_utils import processing_logger
from ml_pipeline.datasets.types import SampleRecord


def run_export(records: list[SampleRecord]) -> dict[str, Path]:
    paths = dataset_paths()
    exports = paths["exports"]
    outputs: dict[str, Path] = {}

    for sub in ["csv", "json", "pickle", "torch", "parquet"]:
        (exports / sub).mkdir(parents=True, exist_ok=True)

    csv_path = exports / "csv" / "samples.csv"
    fieldnames = ["sample_id", "participant_id", "dataset", "modality", "label", "split", "file_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "participant_id": record.participant_id,
                    "dataset": record.dataset,
                    "modality": record.modality,
                    "label": record.label.value,
                    "split": record.split or "",
                    "file_path": str(record.file_path),
                }
            )
    outputs["csv"] = csv_path

    json_path = exports / "json" / "samples.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": r.sample_id,
                    "participant_id": r.participant_id,
                    "dataset": r.dataset,
                    "modality": r.modality,
                    "label": r.label.value,
                }
                for r in records
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs["json"] = json_path

    pickle_path = exports / "pickle" / "samples.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(records, handle)
    outputs["pickle"] = pickle_path

    # torch export — numpy fallback when torch unavailable
    torch_path = exports / "torch" / "samples.pt"
    try:
        import torch

        torch.save([r.__dict__ for r in records], torch_path)
    except ImportError:
        np.save(exports / "torch" / "samples.npy", np.array([len(records)], dtype=np.int32))
        torch_path = exports / "torch" / "samples.npy"
    outputs["torch"] = torch_path

    report_path = exports / "reports" / f"export_report_{datetime.now(UTC).strftime('%Y%m%d')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"formats": list(outputs.keys()), "count": len(records)}), encoding="utf-8")
    outputs["report"] = report_path

    processing_logger.info("export_complete formats=%s", list(outputs.keys()))
    return outputs
