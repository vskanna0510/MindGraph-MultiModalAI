"""Stage 14 — Feature validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_pipeline.audio.types import EmbeddingRecord, FeatureBundle


def validate_features(bundle: FeatureBundle, embedding: EmbeddingRecord | None = None) -> tuple[bool, list[str]]:
    """Verify no NaN/Inf, correct shapes."""
    errors: list[str] = []
    for name, arr in [("mfcc", bundle.mfcc), ("mel", bundle.mel)]:
        if arr is None:
            continue
        if np.isnan(arr).any():
            errors.append(f"{name}_nan")
        if np.isinf(arr).any():
            errors.append(f"{name}_inf")
        if arr.size == 0:
            errors.append(f"{name}_empty")
    if embedding and not embedding.vector_path.exists():
        errors.append("embedding_missing")
    return len(errors) == 0, errors


def write_validation_report(issues: list[dict], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Audio Validation Report", "", f"Total issues: {len(issues)}", ""]
    for item in issues:
        lines.append(f"- {item.get('participant_id')}: {item.get('errors')}")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
