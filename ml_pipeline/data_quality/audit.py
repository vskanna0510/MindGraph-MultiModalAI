"""Audit log for validation runs."""

from __future__ import annotations

import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path

from ml_pipeline.data_quality.types import ApprovalStatus, AuditEntry, DataQualityResult


def append_audit_log(entry: AuditEntry, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.__dict__, default=str) + "\n")
    return path


def build_audit_entry(
    result: DataQualityResult,
    start: str,
    config_hash: str,
    pipeline_version: str,
) -> AuditEntry:
    warnings = sum(1 for g in result.gates for i in g.issues if i.severity == "warning")
    errors = sum(1 for g in result.gates for i in g.issues if i.severity == "error")
    return AuditEntry(
        validation_start=start,
        validation_end=datetime.now(UTC).isoformat(),
        user=os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        machine=socket.gethostname(),
        configuration_hash=config_hash,
        pipeline_version=pipeline_version,
        warnings=warnings,
        errors=errors,
        approval_status=ApprovalStatus.APPROVED if result.approved else ApprovalStatus.REJECTED,
        gates=[{"gate": g.gate, "status": g.status.value, "issues": len(g.issues)} for g in result.gates],
    )
