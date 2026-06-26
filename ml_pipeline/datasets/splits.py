"""Step 4 — Split verification."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ml_pipeline.datasets.config import dataset_paths
from ml_pipeline.datasets.logging_utils import dataset_logger
from ml_pipeline.datasets.types import SampleRecord


def _load_split_files(raw_root: Path) -> dict[str, set[str]]:
    """Load train/dev/test participant IDs from split CSV files."""
    splits: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    splits_dir = raw_root / "splits"
    if not splits_dir.exists():
        return splits
    for split_name in splits:
        for csv_path in splits_dir.glob(f"*{split_name}*"):
            with csv_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    pid = row.get("Participant_ID") or row.get("participant_id") or row.get("id")
                    if pid:
                        splits[split_name].add(str(pid).replace("_P", ""))
    return splits


def verify_splits(records: list[SampleRecord], dataset_name: str = "daic_woz") -> dict:
    """Verify no participant appears in multiple splits."""
    paths = dataset_paths()
    raw_root = paths["raw"] / dataset_name
    splits = _load_split_files(raw_root)
    all_assigned = splits["train"] | splits["dev"] | splits["test"]
    participants = {r.participant_id for r in records if r.dataset == dataset_name}

    overlap_issues: list[str] = []
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        overlap = splits[a] & splits[b]
        if overlap:
            overlap_issues.append(f"{a}/{b}: {sorted(overlap)}")

    unassigned = participants - all_assigned
    report = {
        "dataset": dataset_name,
        "train_count": len(splits["train"]),
        "dev_count": len(splits["dev"]),
        "test_count": len(splits["test"]),
        "unassigned_participants": sorted(unassigned),
        "overlap_issues": overlap_issues,
        "valid": not overlap_issues,
    }
    dataset_logger.info("split_verification dataset=%s valid=%s", dataset_name, report["valid"])
    return report


def assign_splits(records: list[SampleRecord], dataset_name: str = "daic_woz") -> list[SampleRecord]:
    """Attach split field to records based on split files."""
    paths = dataset_paths()
    splits = _load_split_files(paths["raw"] / dataset_name)
    pid_to_split: dict[str, str] = {}
    for split_name, pids in splits.items():
        for pid in pids:
            pid_to_split[pid] = split_name
    for record in records:
        if record.dataset == dataset_name:
            record.split = pid_to_split.get(record.participant_id)
    return records


def write_splits_csv(records: list[SampleRecord]) -> Path:
    """Write datasets/metadata/splits.csv."""
    paths = dataset_paths()
    out = paths["metadata"] / "splits.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    for record in records:
        if record.split:
            seen[record.participant_id] = record.split
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["participant_id", "split", "dataset"])
        writer.writeheader()
        for pid, split in sorted(seen.items()):
            writer.writerow({"participant_id": pid, "split": split, "dataset": "daic_woz"})
    return out


def run_split_verification(records: list[SampleRecord]) -> dict:
    report = verify_splits(records)
    paths = dataset_paths()
    report_path = paths["validation"] / "split_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    assign_splits(records)
    write_splits_csv(records)
    return report
