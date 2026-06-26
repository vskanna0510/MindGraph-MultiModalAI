"""Encoding validation and text cleaning."""

from __future__ import annotations

import re
import unicodedata

INVALID_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")
WHITESPACE_RE = re.compile(r"\s+")
REPEATED_RE = re.compile(r"(.)\1{4,}")


def validate_encoding(text: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        errors.append("invalid_utf8")
    if INVALID_RE.search(text):
        errors.append("corrupted_characters")
    return len(errors) == 0, errors


def clean_text(text: str, preserve_clinical: bool = True) -> str:
    """Clean without removing clinical meaning."""
    text = INVALID_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("''", "'").replace('""', '"')
    text = WHITESPACE_RE.sub(" ", text).strip()
    if not preserve_clinical:
        text = re.sub(r"[^\w\s.,!?'-]", "", text)
    text = REPEATED_RE.sub(r"\1\1\1", text)
    return text


def normalize_text(text: str, lowercase: bool = False) -> str:
    if lowercase:
        return text.lower()
    return text
