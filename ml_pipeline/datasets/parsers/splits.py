"""Official split validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ml_pipeline.datasets.parsers.labels import OfficialLabelParser
from ml_pipeline.datasets.parsers.types import SplitInfo, ValidationIssue


def validate_splits(
    split_records: list[SplitInfo],
    discovered_participant_ids: set[str],
) -> tuple[list[ValidationIssue], dict]:
    """Verify unique IDs, no overlap, expected structure."""
    issues: list[ValidationIssue] = []
    by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}

    for record in split_records:
        by_split.setdefault(record.split, set()).add(record.participant_id)

    # Duplicate within same split file
    split_counts = Counter((r.participant_id, r.split) for r in split_records)
    for (pid, split), count in split_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(pid, "split_duplicate", "error", f"Duplicate in split {split}")
            )

    # Overlap across splits
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        overlap = by_split.get(a, set()) & by_split.get(b, set())
        for pid in overlap:
            issues.append(
                ValidationIssue(pid, "split_overlap", "error", f"Participant in both {a} and {b}")
            )

    all_assigned = by_split.get("train", set()) | by_split.get("dev", set()) | by_split.get("test", set())
    unassigned = discovered_participant_ids - all_assigned
    for pid in sorted(unassigned):
        issues.append(
            ValidationIssue(pid, "split_unassigned", "warning", "Participant not in official splits")
        )

    summary = {
        "train_count": len(by_split.get("train", set())),
        "dev_count": len(by_split.get("dev", set())),
        "test_count": len(by_split.get("test", set())),
        "unassigned": sorted(unassigned),
        "overlap_errors": [i for i in issues if i.check == "split_overlap"],
        "valid": not any(i.severity == "error" for i in issues),
    }
    return issues, summary


def write_split_report(summary: dict, output: Path) -> Path:
    """Generate split_report.md."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Split Validation Report",
        "",
        f"- Train participants: {summary.get('train_count', 0)}",
        f"- Dev participants: {summary.get('dev_count', 0)}",
        f"- Test participants: {summary.get('test_count', 0)}",
        f"- Valid: {summary.get('valid')}",
        "",
    ]
    unassigned = summary.get("unassigned", [])
    if unassigned:
        lines.append(f"## Unassigned ({len(unassigned)})")
        lines.append("")
        for pid in unassigned[:20]:
            lines.append(f"- {pid}")
        if len(unassigned) > 20:
            lines.append(f"- ... and {len(unassigned) - 20} more")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
