"""File discovery and validation utilities."""

from __future__ import annotations

import wave
from pathlib import Path

from ml_pipeline.datasets.parsers.types import FileInfo, ValidationIssue

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
TRANSCRIPT_EXTENSIONS = {".csv", ".txt", ".transcript"}
FACIAL_KEYWORDS = ("clnf", "hog", "gaze", "pose", "_aus", "clnf_au")


def classify_file(path: Path) -> str:
    """Classify file by extension and name heuristics — no hardcoded filenames."""
    lower = path.name.lower()
    ext = path.suffix.lower()
    if any(kw in lower for kw in FACIAL_KEYWORDS):
        return "facial_features"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in TRANSCRIPT_EXTENSIONS:
        if "transcript" in lower:
            return "transcript"
        if ext == ".csv" and any(kw in lower for kw in ("covarep", "formant")):
            return "audio_features"
        if ext == ".csv":
            return "transcript"
        return "text"
    return "other"


def discover_files_in_directory(directory: Path) -> list[FileInfo]:
    """Discover all files in a participant directory."""
    files: list[FileInfo] = []
    if not directory.exists():
        return files
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            modality = classify_file(path)
            if modality == "other":
                continue
            info = FileInfo(path=path, modality=modality, size_bytes=path.stat().st_size)
            files.append(info)
    return files


def validate_audio_file(path: Path) -> tuple[dict, list[str]]:
    """Validate audio file and extract metadata."""
    errors: list[str] = []
    metadata: dict = {}
    if not path.exists():
        return metadata, ["missing"]
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            sample_width = wav.getsampwidth()
            duration = frames / float(sample_rate) if sample_rate else 0.0
            metadata = {
                "channels": channels,
                "sample_rate": sample_rate,
                "bit_depth": sample_width * 8,
                "duration_seconds": round(duration, 3),
                "codec": "pcm",
            }
            if duration <= 0:
                errors.append("zero_duration")
            if sample_rate not in (8000, 16000, 22050, 44100, 48000):
                errors.append(f"unexpected_sample_rate:{sample_rate}")
    except (wave.Error, OSError) as exc:
        errors.append(f"unreadable:{exc}")
    return metadata, errors


def validate_transcript_file(path: Path) -> tuple[dict, list[str]]:
    """Basic transcript file validation."""
    errors: list[str] = []
    metadata: dict = {}
    if not path.exists():
        return metadata, ["missing"]
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            errors.append("empty_transcript")
        metadata["char_count"] = len(text)
        metadata["line_count"] = len(text.splitlines())
    except OSError as exc:
        errors.append(f"unreadable:{exc}")
    return metadata, errors


def validate_video_file(path: Path) -> tuple[dict, list[str]]:
    """Validate video file — uses OpenCV when available."""
    errors: list[str] = []
    metadata: dict = {"duration_seconds": None, "fps": None, "frame_count": None, "resolution": None}
    if not path.exists():
        return metadata, ["missing"]
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            errors.append("unreadable")
            return metadata, errors
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps and fps > 0 else None
        metadata.update(
            {
                "fps": round(fps, 2) if fps else None,
                "frame_count": frame_count,
                "resolution": f"{width}x{height}",
                "duration_seconds": round(duration, 3) if duration else None,
                "codec": "unknown",
            }
        )
        capture.release()
        if frame_count <= 0:
            errors.append("no_frames")
    except ImportError:
        metadata["size_bytes"] = path.stat().st_size
    except OSError as exc:
        errors.append(f"unreadable:{exc}")
    return metadata, errors


def file_to_validation_issues(
    participant_id: str,
    path: Path,
    modality: str,
    errors: list[str],
) -> list[ValidationIssue]:
    """Convert file errors to validation issues."""
    return [
        ValidationIssue(
            participant_id=participant_id,
            check=f"{modality}_validation",
            severity="error" if "missing" in e or "unreadable" in e else "warning",
            message=f"{path.name}: {e}",
        )
        for e in errors
    ]
