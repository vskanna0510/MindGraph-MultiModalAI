"""Base dataset parser interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ml_pipeline.datasets.config import dataset_paths
from ml_pipeline.datasets.parsers.corruption import detect_corruption, write_corruption_report
from ml_pipeline.datasets.parsers.export_index import export_index
from ml_pipeline.datasets.parsers.report import generate_report
from ml_pipeline.datasets.parsers.statistics import compute_statistics
from ml_pipeline.datasets.parsers.types import ParserResult, ValidationIssue
from ml_pipeline.datasets.parsers.visualizations import generate_visualizations


class BaseDatasetParser(ABC):
    """Unified parser interface for all datasets."""

    name: str
    version: str

    def __init__(self, raw_root: Path | None = None) -> None:
        paths = dataset_paths()
        self.raw_root = raw_root or (paths["raw"] / self.name)
        self.output_root = paths["metadata"]
        self.reports_root = paths["root"] / "reports" / "statistics"
        self._result: ParserResult | None = None

    @abstractmethod
    def discover(self) -> ParserResult:
        """Discover all participants, files, and samples."""

    @abstractmethod
    def validate(self, result: ParserResult) -> list[ValidationIssue]:
        """Validate discovered data."""

    @abstractmethod
    def parse(self) -> ParserResult:
        """Full parse: discover + validate + extract metadata."""

    def export(self, result: ParserResult | None = None) -> dict[str, Path]:
        """Export dataset index in multiple formats."""
        result = result or self._result or self.parse()
        return export_index(result.samples, self.output_root)

    def statistics(self, result: ParserResult | None = None) -> dict[str, Any]:
        """Compute dataset statistics."""
        result = result or self._result or self.parse()
        stats = compute_statistics(result)
        result.statistics = stats
        return stats

    def report(self, result: ParserResult | None = None) -> Path:
        """Generate markdown dataset report."""
        result = result or self._result or self.parse()
        stats = result.statistics or self.statistics(result)
        report_path = self.output_root.parent / "reports" / f"{self.name}_dataset_report.md"
        return generate_report(result, stats, report_path)

    def verify(self, result: ParserResult | None = None) -> dict[str, Any]:
        """Final verification of parser outputs."""
        result = result or self._result or self.parse()
        exports = self.export(result)
        stats = self.statistics(result)
        viz = generate_visualizations(stats, self.reports_root)
        report_path = self.report(result)
        corruption_path = write_corruption_report(
            result.validation_issues,
            self.output_root.parent / "validation" / "corruption_report.md",
        )
        passed = not any(i.severity == "error" for i in result.validation_issues)
        return {
            "passed": passed,
            "exports": {k: str(v) for k, v in exports.items()},
            "report": str(report_path),
            "corruption_report": str(corruption_path),
            "visualizations": [str(p) for p in viz],
            "participant_count": len(result.participants),
            "sample_count": len(result.samples),
        }

    def run(self) -> dict[str, Any]:
        """Execute full parser workflow."""
        self._result = self.parse()
        return self.verify(self._result)
