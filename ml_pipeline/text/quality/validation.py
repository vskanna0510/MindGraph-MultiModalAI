"""Quality assessment and validation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def assess_quality(
    text: str,
    language_confidence: float,
    unknown_token_ratio: float = 0.0,
) -> float:
    length_score = min(1.0, len(text.split()) / 50.0)
    encoding_score = 1.0 if text and not text.count("\ufffd") else 0.0
    lang_score = language_confidence
    unknown_score = 1.0 - unknown_token_ratio
    return round(0.3 * length_score + 0.2 * encoding_score + 0.3 * lang_score + 0.2 * unknown_score, 4)


def validate_result(result: Any) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not result.raw_text and not result.cleaned_text:
        errors.append("empty_text")
    if result.embedding is not None:
        vec = np.load(result.embedding.vector_path)
        if np.isnan(vec).any():
            errors.append("nan_embedding")
        if vec.shape[-1] != result.embedding.embedding_dim:
            errors.append("embedding_dim_mismatch")
    if result.language and result.language.confidence < 0:
        errors.append("invalid_language_confidence")
    return len(errors) == 0, errors
