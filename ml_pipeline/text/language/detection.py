"""Language and code-mix detection."""

from __future__ import annotations

import re

from ml_pipeline.text.types import LanguageInfo

TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
HINDI_RE = re.compile(r"[\u0900-\u097F]")
ENGLISH_RE = re.compile(r"[a-zA-Z]")


def detect_language(text: str) -> LanguageInfo:
    """Detect primary language with confidence."""
    if not text.strip():
        return LanguageInfo(primary="unknown", confidence=0.0)

    tamil_chars = len(TAMIL_RE.findall(text))
    hindi_chars = len(HINDI_RE.findall(text))
    english_chars = len(ENGLISH_RE.findall(text))
    total = tamil_chars + hindi_chars + english_chars + 1

    dist = {
        "ta": tamil_chars / total,
        "hi": hindi_chars / total,
        "en": english_chars / total,
    }
    primary = max(dist, key=dist.get)
    mixed = sum(1 for v in dist.values() if v > 0.15) > 1

    if mixed and english_chars > 0 and tamil_chars > 0:
        primary = "tanglish"
    elif mixed:
        primary = "mixed"

    confidence = dist.get(primary, 0.0) if primary in dist else 0.7
    return LanguageInfo(
        primary=primary,
        mixed=mixed,
        confidence=round(min(1.0, confidence + 0.2), 4),
        distribution={k: round(v, 4) for k, v in dist.items()},
    )


def detect_code_mix(text: str) -> dict[str, str]:
    """Token-level language tags."""
    tags: dict[str, str] = {}
    for token in text.split():
        if TAMIL_RE.search(token):
            tags[token] = "ta"
        elif HINDI_RE.search(token):
            tags[token] = "hi"
        elif ENGLISH_RE.search(token):
            tags[token] = "en"
        else:
            tags[token] = "unknown"
    return tags
