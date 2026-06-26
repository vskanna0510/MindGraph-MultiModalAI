"""Stage 1 — Audio integrity verification."""

from __future__ import annotations

import wave
from pathlib import Path

from ml_pipeline.audio.utils.io import SUPPORTED_EXTENSIONS


def verify_integrity(path: Path) -> tuple[bool, list[str]]:
    """Verify file exists, readable, and has valid header."""
    errors: list[str] = []
    if not path.exists():
        return False, ["missing_file"]
    if path.stat().st_size == 0:
        return False, ["zero_length"]
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        errors.append("unsupported_format")
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                if wav.getnframes() == 0:
                    errors.append("zero_frames")
        except wave.Error:
            errors.append("corrupted_header")
    return len(errors) == 0, errors
