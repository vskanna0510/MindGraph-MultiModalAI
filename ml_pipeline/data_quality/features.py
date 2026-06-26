"""Feature tensor validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.data_quality.types import GateResult, GateStatus, ValidationIssue


def validate_features(sessions_df: pd.DataFrame, feature_root: Path) -> GateResult:
    issues: list[ValidationIssue] = []
    checked = 0
    found = 0

    for _, row in sessions_df.iterrows():
        pid = str(row["participant_id"])
        checked += 1
        for subdir in ("audio", "visual", "text"):
            mod_dir = feature_root / subdir
            if not mod_dir.exists():
                continue
            matches = list(mod_dir.glob(f"{pid}*.npy"))
            if not matches:
                continue
            found += 1
            try:
                arr = np.load(matches[0])
                if np.isnan(arr).any():
                    issues.append(ValidationIssue("nan", "error", f"NaN in {subdir}", pid))
                if np.isinf(arr).any():
                    issues.append(ValidationIssue("inf", "error", f"Inf in {subdir}", pid))
            except Exception as exc:
                issues.append(ValidationIssue("corrupt", "error", str(exc), pid))

    status = GateStatus.PASSED if not [i for i in issues if i.severity == "error"] else GateStatus.WARNING
    return GateResult("features", status, issues, {"checked": checked, "features_found": found})
