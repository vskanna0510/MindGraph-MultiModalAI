"""Step 2 — File validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ml_pipeline.datasets.config import dataset_paths, load_config
from ml_pipeline.datasets.logging_utils import validation_logger
from ml_pipeline.datasets.types import SampleRecord


@dataclass
class ValidationIssue:
    sample_id: str
    file_path: str
    check: str
    severity: str
    message: str


@dataclass
class ValidationReport:
    timestamp: str
    total_samples: int
    passed: int
    failed: int
    issues: list[ValidationIssue]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_samples": self.total_samples,
            "passed": self.passed,
            "failed": self.failed,
            "issues": [asdict(i) for i in self.issues],
        }


def _file_checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample(record: SampleRecord, cfg: dict | None = None) -> list[ValidationIssue]:
    """Validate existence, size, extension, and checksum for one sample."""
    cfg = cfg or load_config("validation.yaml").get("validation", {})
    issues: list[ValidationIssue] = []
    path = record.file_path

    if not path.exists():
        issues.append(
            ValidationIssue(record.sample_id, str(path), "existence", "error", "File missing")
        )
        return issues

    size = path.stat().st_size
    min_size = int(cfg.get("min_file_size_bytes", 1))
    if size < min_size:
        issues.append(
            ValidationIssue(record.sample_id, str(path), "size", "error", f"Too small: {size}")
        )

    ext = path.suffix.lower()
    allowed = cfg.get("allowed_extensions", {})
    modality_key = record.modality if record.modality in allowed else "numpy"
    valid_exts = allowed.get(modality_key, allowed.get("numpy", []))
    if valid_exts and ext not in valid_exts:
        issues.append(
            ValidationIssue(
                record.sample_id,
                str(path),
                "extension",
                "warning",
                f"Unexpected extension {ext}",
            )
        )

    if record.label.value == "unknown":
        issues.append(
            ValidationIssue(
                record.sample_id,
                str(path),
                "label",
                "warning",
                "Label unavailable",
            )
        )

    try:
        _file_checksum(path, cfg.get("checksum_algorithm", "sha256"))
    except OSError as exc:
        issues.append(
            ValidationIssue(record.sample_id, str(path), "checksum", "error", str(exc))
        )

    return issues


def validate_records(records: list[SampleRecord]) -> ValidationReport:
    """Validate all discovered samples."""
    issues: list[ValidationIssue] = []
    for record in records:
        issues.extend(validate_sample(record))

    errors = [i for i in issues if i.severity == "error"]
    passed = len(records) - len({i.sample_id for i in errors})
    report = ValidationReport(
        timestamp=datetime.now(UTC).isoformat(),
        total_samples=len(records),
        passed=max(passed, 0),
        failed=len({i.sample_id for i in errors}),
        issues=issues,
    )
    validation_logger.info(
        "validation_complete total=%d passed=%d failed=%d",
        report.total_samples,
        report.passed,
        report.failed,
    )
    return report


def write_validation_report(report: ValidationReport, output: Path | None = None) -> Path:
    """Persist validation report to datasets/validation/."""
    paths = dataset_paths()
    out_dir = output.parent if output else paths["validation"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = output or (out_dir / "validation_report.json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)

    md_path = out_dir / "validation_report.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Validation Report\n\n")
        handle.write(f"- Timestamp: {report.timestamp}\n")
        handle.write(f"- Total: {report.total_samples}\n")
        handle.write(f"- Passed: {report.passed}\n")
        handle.write(f"- Failed: {report.failed}\n\n")
        if report.issues:
            handle.write("## Issues\n\n")
            for issue in report.issues:
                handle.write(f"- [{issue.severity}] {issue.sample_id}: {issue.message}\n")
    return json_path


def run_validation(records: list[SampleRecord]) -> ValidationReport:
    report = validate_records(records)
    write_validation_report(report)
    return report
