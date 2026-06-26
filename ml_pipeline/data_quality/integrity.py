"""Integrity validation — file existence, checksum, corruption."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ml_pipeline.data_quality.types import GateResult, GateStatus, ValidationIssue


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_integrity(index_df: pd.DataFrame, cfg: dict) -> GateResult:
    issues: list[ValidationIssue] = []
    integrity_cfg = cfg.get("integrity", {})
    min_size = int(integrity_cfg.get("min_file_size_bytes", 1))
    allowed = integrity_cfg.get("allowed_extensions", {})
    seen_paths: set[str] = set()

    if index_df.empty:
        return GateResult("integrity", GateStatus.FAILED, [ValidationIssue("integrity", "error", "Empty index")])

    for _, row in index_df.iterrows():
        pid = str(row.get("participant_id", ""))
        path_str = str(row.get("file_path", ""))
        if not path_str:
            issues.append(ValidationIssue("existence", "error", "Missing file path", pid))
            continue
        path = Path(path_str)
        if not path.exists():
            issues.append(ValidationIssue("existence", "error", f"File missing: {path.name}", pid))
            continue
        if path_str in seen_paths:
            issues.append(ValidationIssue("duplicate", "warning", f"Duplicate path: {path.name}", pid))
        seen_paths.add(path_str)
        size = path.stat().st_size
        if size < min_size:
            issues.append(ValidationIssue("size", "error", f"Too small ({size} bytes)", pid))
        ext = path.suffix.lower()
        modality = str(row.get("modality", "numpy"))
        valid_exts = allowed.get(modality, allowed.get("numpy", []))
        if valid_exts and ext not in valid_exts:
            issues.append(ValidationIssue("extension", "warning", f"Unexpected extension {ext}", pid))

    errors = [i for i in issues if i.severity == "error"]
    status = GateStatus.FAILED if errors else GateStatus.PASSED
    return GateResult("integrity", status, issues, {"checked": len(index_df), "errors": len(errors)})


def write_integrity_report(result: GateResult, output_path: Path) -> Path:
    lines = ["# Integrity Report\n", f"Status: **{result.status.value}**\n\n"]
    lines.append(f"- Files checked: {result.metadata.get('checked', 0)}\n")
    lines.append(f"- Errors: {result.metadata.get('errors', 0)}\n\n")
    for issue in result.issues[:100]:
        lines.append(f"- [{issue.severity}] {issue.participant_id}: {issue.message}\n")
    if len(result.issues) > 100:
        lines.append(f"\n... and {len(result.issues) - 100} more\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
