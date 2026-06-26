"""Metadata and consistency validation."""

from __future__ import annotations

import pandas as pd

from ml_pipeline.data_quality.types import GateResult, GateStatus, ValidationIssue


def validate_consistency(sessions_df: pd.DataFrame) -> GateResult:
    issues: list[ValidationIssue] = []
    if sessions_df.empty:
        return GateResult("consistency", GateStatus.FAILED, [ValidationIssue("consistency", "error", "No sessions")])

    required = ["participant_id", "session_id", "dataset", "split", "label"]
    for col in required:
        if col not in sessions_df.columns:
            issues.append(ValidationIssue("metadata", "error", f"Missing column: {col}"))

    if "participant_id" in sessions_df.columns and "session_id" in sessions_df.columns:
        dup = sessions_df.duplicated(subset=["participant_id", "session_id"], keep=False)
        for pid in sessions_df.loc[dup, "participant_id"].unique()[:20]:
            issues.append(ValidationIssue("duplicate_session", "error", "Conflicting session records", str(pid)))

    valid_splits = {"train", "dev", "valid", "test", "unknown"}
    if "split" in sessions_df.columns:
        bad = sessions_df[~sessions_df["split"].isin(valid_splits)]
        for pid in bad["participant_id"].head(10):
            issues.append(ValidationIssue("split", "warning", f"Non-standard split", str(pid)))

    valid_labels = {"depression", "normal", "unknown", "excluded"}
    if "label" in sessions_df.columns:
        bad_labels = sessions_df[~sessions_df["label"].astype(str).str.lower().isin(valid_labels)]
        for pid in bad_labels["participant_id"].head(10):
            issues.append(ValidationIssue("label", "warning", "Non-standard label", str(pid)))

    errors = [i for i in issues if i.severity == "error"]
    status = GateStatus.FAILED if errors else GateStatus.PASSED
    return GateResult("consistency", status, issues, {"sessions": len(sessions_df)})
