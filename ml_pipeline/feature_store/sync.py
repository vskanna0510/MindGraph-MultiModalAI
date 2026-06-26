"""Synchronization engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def compute_sync_score(audio_duration: float, transcript_chars: int, frame_count: int = 0) -> float:
    """Heuristic sync score when frame-level timestamps unavailable."""
    if audio_duration <= 0:
        return 0.0
    expected_chars = audio_duration * 15  # ~15 chars/sec speech heuristic
    char_ratio = min(transcript_chars, expected_chars) / max(expected_chars, 1)
    frame_ratio = 1.0
    if frame_count > 0:
        expected_frames = audio_duration * 30
        frame_ratio = min(frame_count, expected_frames) / max(expected_frames, 1)
    return round(0.6 * char_ratio + 0.4 * frame_ratio, 4)


def assess_transcript_audio_sync(transcript_path: Path, audio_duration: float) -> float:
    if not transcript_path.exists():
        return 0.0
    try:
        if transcript_path.suffix == ".csv":
            df = pd.read_csv(transcript_path, sep=None, engine="python")
            text_col = next((c for c in df.columns if c.lower() in {"value", "text"}), None)
            chars = int(df[text_col].astype(str).str.len().sum()) if text_col else transcript_path.stat().st_size
        else:
            chars = transcript_path.stat().st_size
        return compute_sync_score(audio_duration, chars)
    except Exception:
        return 0.5


def write_sync_report(scores: list[dict[str, Any]], output_path: Path) -> Path:
    lines = ["# Synchronization Report\n"]
    avg = sum(s.get("sync_score", 0) for s in scores) / max(len(scores), 1)
    lines.append(f"Average sync score: {avg:.4f}\n\n")
    for s in scores:
        lines.append(f"- {s.get('participant_id')}: {s.get('sync_score', 0):.4f}\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
