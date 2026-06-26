"""Stage 8-9 — VAD and silence detection."""

from __future__ import annotations

import numpy as np

from ml_pipeline.audio.config import audio_config
from ml_pipeline.audio.types import PauseStatistics, SpeechSegment, VADResult


def energy_vad(audio: np.ndarray, sample_rate: int) -> VADResult:
    """Energy-based voice activity detection."""
    cfg = audio_config().get("vad", {})
    frame_ms = int(cfg.get("frame_ms", 30))
    threshold = float(cfg.get("energy_threshold", 0.02))
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = max(1, len(audio) // frame_size)
    speech_segments: list[SpeechSegment] = []
    silence_segments: list[SpeechSegment] = []
    in_speech = False
    start = 0.0
    idx = 0
    for i in range(n_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        energy = float(np.sqrt(np.mean(frame**2))) if len(frame) else 0.0
        t0 = i * frame_size / sample_rate
        t1 = (i + 1) * frame_size / sample_rate
        is_speech = energy >= threshold
        if is_speech and not in_speech:
            if i > 0:
                silence_segments.append(SpeechSegment(start, t0, len(silence_segments), start, t0))
            start = t0
            in_speech = True
        elif not is_speech and in_speech:
            speech_segments.append(SpeechSegment(start, t0, idx, start, t0))
            idx += 1
            start = t0
            in_speech = False
    duration = len(audio) / sample_rate
    if in_speech:
        speech_segments.append(SpeechSegment(start, duration, idx, start, duration))
    else:
        silence_segments.append(SpeechSegment(start, duration, len(silence_segments), start, duration))

    speech_dur = sum(s.end_seconds - s.start_seconds for s in speech_segments)
    speech_ratio = speech_dur / duration if duration else 0.0
    avg_seg = speech_dur / max(len(speech_segments), 1)
    return VADResult(
        speech_segments=speech_segments,
        silence_segments=silence_segments,
        speech_ratio=round(speech_ratio, 4),
        avg_segment_duration=round(avg_seg, 3),
        method="energy",
    )


def run_vad(audio: np.ndarray, sample_rate: int) -> VADResult:
    """Run configured VAD method."""
    method = audio_config().get("vad", {}).get("method", "energy")
    if method == "energy":
        return energy_vad(audio, sample_rate)
    return energy_vad(audio, sample_rate)


def compute_pauses(audio: np.ndarray, sample_rate: int, vad: VADResult) -> PauseStatistics:
    """Compute leading/trailing/internal pause statistics."""
    duration = len(audio) / sample_rate if sample_rate else 0.0
    leading = vad.silence_segments[0].end_seconds - vad.silence_segments[0].start_seconds if vad.silence_segments else 0.0
    trailing = 0.0
    if vad.silence_segments:
        trailing = vad.silence_segments[-1].end_seconds - vad.silence_segments[-1].start_seconds
    internal = max(0, len(vad.silence_segments) - 2)
    pause_durs = [s.end_seconds - s.start_seconds for s in vad.silence_segments[1:-1]] if internal else []
    avg_pause = float(np.mean(pause_durs)) if pause_durs else 0.0
    silence_pct = 1.0 - vad.speech_ratio
    return PauseStatistics(
        leading_silence=round(leading, 3),
        trailing_silence=round(trailing, 3),
        internal_pause_count=internal,
        avg_pause_duration=round(avg_pause, 3),
        silence_percentage=round(silence_pct, 4),
    )
