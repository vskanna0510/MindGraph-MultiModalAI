"""Embedding validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.data_quality.types import GateResult, GateStatus, ValidationIssue


def validate_embeddings(sessions_df: pd.DataFrame, roots: list[Path]) -> GateResult:
    issues: list[ValidationIssue] = []
    checksums: dict[str, str] = {}
    found = 0

    for _, row in sessions_df.iterrows():
        pid = str(row["participant_id"])
        sid = str(row.get("session_id", pid))
        emb_path = None
        for root in roots:
            for pattern in (f"{pid}_{sid}_text.npy", f"{pid}_text_v1.npy", f"{pid}*.npy"):
                matches = list(root.glob(pattern)) if "*" in pattern else [root / pattern]
                matches = [m for m in matches if m.exists()]
                if matches:
                    emb_path = matches[0]
                    break
            if emb_path:
                break
        if emb_path is None:
            continue
        found += 1
        try:
            arr = np.load(emb_path)
            if arr.size == 0:
                issues.append(ValidationIssue("empty_embedding", "error", "Empty embedding", pid))
            cs = hashlib.sha256(arr.tobytes()).hexdigest()
            if cs in checksums.values():
                issues.append(ValidationIssue("duplicate_embedding", "error", "Duplicate embedding checksum", pid))
            checksums[emb_path.name] = cs
            meta_path = emb_path.with_suffix(".json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if "model_version" not in meta and "feature_version" not in meta:
                    issues.append(ValidationIssue("metadata", "warning", "Missing version in sidecar", pid))
        except Exception as exc:
            issues.append(ValidationIssue("corrupt_embedding", "error", str(exc), pid))

    errors = [i for i in issues if i.severity == "error"]
    status = GateStatus.PASSED if not errors else GateStatus.WARNING
    return GateResult("embeddings", status, issues, {"embeddings_found": found})


def write_embedding_report(result: GateResult, output_path: Path) -> Path:
    lines = ["# Embedding Validation Report\n", f"Status: **{result.status.value}**\n\n"]
    lines.append(f"- Embeddings found: {result.metadata.get('embeddings_found', 0)}\n\n")
    for issue in result.issues[:50]:
        lines.append(f"- [{issue.severity}] {issue.participant_id}: {issue.message}\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
