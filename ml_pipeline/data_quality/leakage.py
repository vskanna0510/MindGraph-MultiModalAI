"""Data leakage detection."""

from __future__ import annotations

import pandas as pd

from ml_pipeline.data_quality.types import GateResult, GateStatus, ValidationIssue


def detect_leakage(sessions_df: pd.DataFrame, index_df: pd.DataFrame) -> GateResult:
    issues: list[ValidationIssue] = []

    if not sessions_df.empty and "participant_id" in sessions_df.columns and "split" in sessions_df.columns:
        split_map = sessions_df.groupby("participant_id")["split"].nunique()
        leaked = split_map[split_map > 1]
        for pid in leaked.index:
            issues.append(
                ValidationIssue(
                    "split_leakage",
                    "error",
                    f"Participant in multiple splits: {leaked[pid]}",
                    str(pid),
                )
            )

    if not index_df.empty and "file_path" in index_df.columns:
        dup_paths = index_df[index_df.duplicated("file_path", keep=False)]
        for path in dup_paths["file_path"].unique()[:20]:
            issues.append(ValidationIssue("duplicate_file", "error", f"Duplicated file: {_basename(path)}"))

    if not sessions_df.empty and "participant_id" in sessions_df.columns:
        dup_sessions = sessions_df.duplicated(subset=["participant_id", "session_id"])
        for pid in sessions_df.loc[dup_sessions, "participant_id"].unique()[:20]:
            issues.append(ValidationIssue("duplicate_session", "error", "Duplicated session", str(pid)))

    errors = [i for i in issues if i.severity == "error"]
    status = GateStatus.FAILED if errors else GateStatus.PASSED
    return GateResult("leakage", status, issues, {"leakage_count": len(errors)})


def _basename(path) -> str:
    return str(path).replace("\\", "/").split("/")[-1]


def write_leakage_report(result: GateResult, output_path) -> Path:
    from pathlib import Path

    output_path = Path(output_path)
    lines = ["# Data Leakage Report\n", f"Status: **{result.status.value}**\n\n"]
    if not result.issues:
        lines.append("No leakage detected.\n")
    for issue in result.issues:
        lines.append(f"- [{issue.severity}] {issue.participant_id}: {issue.message}\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
