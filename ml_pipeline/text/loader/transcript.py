"""Transcript loading utilities."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from ml_pipeline.text.types import Utterance


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_transcript(path: Path) -> tuple[str, list[Utterance]]:
    """Load DAIC-style CSV or plain text transcript."""
    utterances: list[Utterance] = []
    if path.suffix.lower() == ".csv":
        raw = path.read_text(encoding="utf-8", errors="replace")
        delimiter = "\t" if "\t" in raw.splitlines()[0] else ","
        reader = csv.DictReader(raw.splitlines(), delimiter=delimiter)
        lines: list[str] = []
        for order, row in enumerate(reader):
            speaker = row.get("speaker") or row.get("Speaker") or "unknown"
            text = row.get("value") or row.get("Text") or row.get("text") or ""
            start = row.get("start_time") or row.get("Start_Time")
            stop = row.get("stop_time") or row.get("Stop_Time")
            utterances.append(
                Utterance(
                    speaker=speaker,
                    text=text.strip(),
                    start_time=float(start) if start else None,
                    stop_time=float(stop) if stop else None,
                    order=order,
                )
            )
            lines.append(text.strip())
        return "\n".join(lines), utterances
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, [Utterance(speaker="unknown", text=text, order=0)]
