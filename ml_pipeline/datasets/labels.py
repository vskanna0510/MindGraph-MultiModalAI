"""Label normalization — mappings from config only."""

from __future__ import annotations

from ml_pipeline.datasets.config import label_mapping
from ml_pipeline.datasets.types import LabelClass


def normalize_label(raw: str) -> LabelClass:
    """Map raw dataset label strings to normalized LabelClass."""
    value = raw.strip().lower()
    if value in {"1", "depressed", "depression", "positive", "yes", "true"}:
        return LabelClass.DEPRESSION
    if value in {"0", "normal", "negative", "no", "false", "non-depressed"}:
        return LabelClass.NORMAL
    if value in {"excluded", "exclude", "drop"}:
        return LabelClass.EXCLUDED
    if value in {"", "na", "nan", "none", "unknown", "-1"}:
        return LabelClass.UNKNOWN
    return LabelClass.UNKNOWN


def label_to_int(label: LabelClass) -> int:
    """Convert LabelClass to integer using config mapping."""
    mapping = label_mapping()
    return int(mapping.get(label.value, -1))
