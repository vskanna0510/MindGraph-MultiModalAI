"""Feature validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml_pipeline.feature_store.types import FeatureRecord, SessionBatch


def validate_feature(record: FeatureRecord, array: np.ndarray | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if array is None and record.storage_path.exists():
        array = np.load(record.storage_path)
    if array is None:
        errors.append("missing_array")
        return False, errors
    if np.isnan(array).any():
        errors.append("nan_values")
    if np.isinf(array).any():
        errors.append("inf_values")
    if tuple(array.shape) != record.embedding_shape and array.ndim > 0:
        if array.shape[-1] != record.embedding_dim and array.size != record.embedding_dim:
            errors.append("shape_mismatch")
    if record.embedding_dim <= 0:
        errors.append("invalid_dimension")
    return len(errors) == 0, errors


def validate_session_batch(batch: SessionBatch, require_features: bool = False) -> tuple[bool, list[str]]:
    errors: list[str] = []
    has_any = False
    for name, arr in (("audio", batch.audio), ("visual", batch.visual), ("text", batch.text)):
        if arr is None:
            if batch.modality_mask.get(name) and require_features:
                errors.append(f"missing_{name}")
            continue
        has_any = True
        if np.isnan(arr).any():
            errors.append(f"nan_{name}")
        if np.isinf(arr).any():
            errors.append(f"inf_{name}")
    if require_features and not has_any:
        errors.append("no_features_loaded")
    return len(errors) == 0, errors


def write_validation_report(results: list[dict[str, Any]], output_path: Path) -> Path:
    lines = ["# Feature Validation Report\n"]
    passed = sum(1 for r in results if r.get("passed"))
    lines.append(f"- Total: {len(results)}\n")
    lines.append(f"- Passed: {passed}\n")
    lines.append(f"- Failed: {len(results) - passed}\n\n")
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        lines.append(f"## {r.get('participant_id', 'unknown')} — {status}\n")
        if r.get("errors"):
            for e in r["errors"]:
                lines.append(f"- {e}\n")
        lines.append("\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
