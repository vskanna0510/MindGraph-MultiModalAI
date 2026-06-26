"""Audio statistics generation."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ml_pipeline.audio.config import audio_paths
from ml_pipeline.audio.types import AudioPipelineResult


def compute_audio_statistics(results: list[AudioPipelineResult]) -> dict[str, Any]:
    durations = [r.metadata.duration_seconds for r in results if r.metadata]
    speech_ratios = [r.metadata.speech_ratio for r in results if r.metadata]
    f0_means = [
        r.features.pitch.get("f0_mean", 0) for r in results if r.features and r.features.pitch
    ]
    return {
        "count": len(results),
        "avg_duration": round(statistics.mean(durations), 2) if durations else 0,
        "median_duration": round(statistics.median(durations), 2) if durations else 0,
        "sample_rate_distribution": dict(Counter(r.metadata.sample_rate for r in results if r.metadata)),
        "avg_speech_ratio": round(statistics.mean(speech_ratios), 4) if speech_ratios else 0,
        "avg_silence_ratio": round(1 - statistics.mean(speech_ratios), 4) if speech_ratios else 0,
        "pitch_mean_avg": round(statistics.mean(f0_means), 2) if f0_means else 0,
        "validation_pass_rate": sum(1 for r in results if r.validation_passed) / max(len(results), 1),
    }


def write_audio_statistics(results: list[AudioPipelineResult]) -> Path:
    stats = compute_audio_statistics(results)
    out = audio_paths()["reports"] / "audio_statistics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out
