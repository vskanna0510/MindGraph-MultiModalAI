"""Corruption detection for dataset files."""

from __future__ import annotations

from pathlib import Path

from ml_pipeline.datasets.parsers.types import ParticipantMetadata, ValidationIssue


def detect_corruption(participant: ParticipantMetadata) -> list[ValidationIssue]:
    """Detect corrupted or broken participant data."""
    issues: list[ValidationIssue] = []
    pid = participant.participant_id

    if participant.audio_length is not None and participant.audio_length <= 0:
        issues.append(ValidationIssue(pid, "corruption", "error", "Broken audio: zero duration"))
    if participant.transcript_length == 0 and participant.files.get("transcript"):
        issues.append(ValidationIssue(pid, "corruption", "error", "Empty transcript"))
    if participant.label.value == "excluded":
        issues.append(ValidationIssue(pid, "corruption", "warning", "Invalid/excluded label"))

    for path in participant.files.values():
        if path and path.exists() and path.stat().st_size == 0:
            issues.append(ValidationIssue(pid, "corruption", "error", f"Empty file: {path.name}"))

    return issues


def write_corruption_report(issues: list[ValidationIssue], output: Path) -> Path:
    """Write corruption_report.md."""
    output.parent.mkdir(parents=True, exist_ok=True)
    corruption = [i for i in issues if i.check == "corruption"]
    lines = ["# Corruption Report", "", f"Total issues: {len(corruption)}", ""]
    for issue in corruption:
        lines.append(f"- [{issue.severity}] {issue.participant_id}: {issue.message}")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
