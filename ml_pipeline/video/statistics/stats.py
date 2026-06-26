"""Video statistics."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from ml_pipeline.video.config import video_paths
from ml_pipeline.video.types import VideoPipelineResult


def compute_video_statistics(results: list[VideoPipelineResult]) -> dict:
    fps_vals = [r.metadata.fps for r in results if r.metadata]
    face_rates = [r.metadata.face_detection_rate for r in results if r.metadata]
    return {
        "count": len(results),
        "avg_fps": round(statistics.mean(fps_vals), 2) if fps_vals else 0,
        "avg_resolution": f"{round(statistics.mean([r.metadata.width for r in results if r.metadata]))}x{round(statistics.mean([r.metadata.height for r in results if r.metadata]))}" if results else "0x0",
        "face_detection_rate": round(statistics.mean(face_rates), 4) if face_rates else 0,
        "landmark_success_rate": round(statistics.mean([r.metadata.landmark_success_rate for r in results if r.metadata]), 4) if results else 0,
        "validation_pass_rate": sum(1 for r in results if r.validation_passed) / max(len(results), 1),
    }


def write_video_statistics(results: list[VideoPipelineResult]) -> Path:
    stats = compute_video_statistics(results)
    out = video_paths()["reports"] / "video_statistics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out
