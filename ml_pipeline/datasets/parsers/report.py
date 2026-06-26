"""Dataset markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_pipeline.datasets.parsers.types import ParserResult


def generate_report(result: ParserResult, stats: dict[str, Any], output: Path) -> Path:
    """Generate comprehensive dataset report markdown."""
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = [i for i in result.validation_issues if i.severity == "error"]
    warnings = [i for i in result.validation_issues if i.severity == "warning"]

    lines = [
        "# Dataset Engineering Report",
        "",
        f"**Dataset:** {result.dataset}",
        "",
        "## Summary",
        "",
        f"- Participants: {stats.get('participant_count', 0)}",
        f"- Sessions: {stats.get('session_count', 0)}",
        f"- Total hours: {stats.get('total_hours', 0)}",
        f"- Samples indexed: {len(result.samples)}",
        "",
        "## Labels",
        "",
    ]
    for label, count in stats.get("label_distribution", {}).items():
        lines.append(f"- {label}: {count}")

    lines.extend(["", "## Splits", ""])
    for split, count in stats.get("split_distribution", {}).items():
        lines.append(f"- {split}: {count}")

    lines.extend(
        [
            "",
            "## Quality",
            "",
            f"- Validation errors: {len(errors)}",
            f"- Validation warnings: {len(warnings)}",
            "",
            "## Statistics",
            "",
            f"- Average duration: {stats.get('average_duration', 0)}s",
            f"- Median duration: {stats.get('median_duration', 0)}s",
            f"- Avg audio length: {stats.get('avg_audio_length', 0)}s",
            f"- Avg transcript length: {stats.get('avg_transcript_length', 0)} words",
            "",
            "## Recommendations",
            "",
        ]
    )
    if errors:
        lines.append("- Resolve validation errors before training.")
    if stats.get("label_distribution", {}).get("unknown", 0):
        lines.append("- Test-set participants have withheld labels (expected for DAIC).")
    if not errors:
        lines.append("- Dataset ready for preprocessing pipeline.")

    output.write_text("\n".join(lines), encoding="utf-8")
    return output
