"""Stage 10 — Speech segmentation."""

from __future__ import annotations

import numpy as np

from ml_pipeline.audio.config import audio_config
from ml_pipeline.audio.types import SpeechSegment, VADResult


def segment_speech(
    audio: np.ndarray,
    sample_rate: int,
    vad: VADResult,
) -> list[tuple[np.ndarray, SpeechSegment]]:
    """Split long recordings into overlapping segments."""
    cfg = audio_config().get("segmentation", {})
    max_len = float(cfg.get("max_segment_seconds", 30.0))
    overlap = float(cfg.get("overlap_seconds", 0.5))
    min_len = float(cfg.get("min_segment_seconds", 0.5))
    padding = float(cfg.get("padding_seconds", 0.1))

    segments: list[tuple[np.ndarray, SpeechSegment]] = []
    for seg in vad.speech_segments:
        start, end = seg.start_seconds, seg.end_seconds
        duration = end - start
        if duration < min_len:
            continue
        cursor = start
        idx = 0
        while cursor < end:
            seg_end = min(cursor + max_len, end)
            s_idx = max(0, int((cursor - padding) * sample_rate))
            e_idx = min(len(audio), int((seg_end + padding) * sample_rate))
            chunk = audio[s_idx:e_idx]
            if len(chunk) / sample_rate >= min_len:
                meta = SpeechSegment(cursor, seg_end, idx, seg.source_start, seg.source_end)
                segments.append((chunk, meta))
                idx += 1
            cursor += max_len - overlap
            if seg_end >= end:
                break
    return segments
