"""Dataset validation for research reproducibility."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationIssue:
    """Single dataset validation issue."""

    severity: str
    path: str
    message: str


@dataclass
class ValidationReport:
    """Aggregated dataset validation report."""

    dataset_path: str
    total_files: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "total_files": self.total_files,
            "passed": self.passed,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def validate_dataset_root(dataset_root: Path) -> ValidationReport:
    """Validate dataset directory structure and basic file integrity."""
    report = ValidationReport(dataset_path=str(dataset_root))
    if not dataset_root.exists():
        report.issues.append(
            ValidationIssue("error", str(dataset_root), "Dataset directory does not exist")
        )
        return report

    allowed_extensions = {".wav", ".mp4", ".txt", ".json", ".csv", ".yaml", ".md"}
    seen_hashes: set[str] = set()

    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        report.total_files += 1
        if path.suffix.lower() not in allowed_extensions and path.name != ".gitkeep":
            report.issues.append(
                ValidationIssue("warning", str(path), f"Unexpected file extension: {path.suffix}")
            )
        if path.stat().st_size == 0 and path.name != ".gitkeep":
            report.issues.append(ValidationIssue("error", str(path), "Empty file detected"))
        key = f"{path.name}:{path.stat().st_size}"
        if key in seen_hashes:
            report.issues.append(ValidationIssue("warning", str(path), "Potential duplicate file"))
        seen_hashes.add(key)

    required_dirs = ["metadata"]
    for dirname in required_dirs:
        if not (dataset_root / dirname).exists():
            report.issues.append(
                ValidationIssue("warning", dirname, f"Recommended directory missing: {dirname}")
            )

    return report


def write_validation_report(report: ValidationReport, output_path: Path) -> None:
    """Write validation report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
