"""Video integrity and metadata preprocessing."""

from __future__ import annotations

from pathlib import Path

from ml_pipeline.video.types import VideoMetadata
from ml_pipeline.video.utils.io import SUPPORTED_EXTENSIONS, read_video_metadata

SUPPORTED_NPY = {".npy"}


def verify_integrity(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return False, ["missing_file"]
    if path.stat().st_size == 0:
        return False, ["zero_length"]
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS and ext not in SUPPORTED_NPY:
        errors.append("unsupported_format")
    if ext in SUPPORTED_EXTENSIONS:
        try:
            from ml_pipeline.video.utils.io import open_capture

            cap = open_capture(path)
            if cap.get(__import__("cv2").CAP_PROP_FRAME_COUNT) <= 0:
                errors.append("zero_frames")
            cap.release()
        except OSError:
            errors.append("unreadable")
    return len(errors) == 0, errors


def extract_metadata(path: Path, participant_id: str, session_id: str | None = None) -> VideoMetadata:
    info = read_video_metadata(path)
    return VideoMetadata(
        participant_id=participant_id,
        session_id=session_id or participant_id,
        source_path=str(path),
        duration_seconds=round(info["duration_seconds"], 3),
        fps=round(info["fps"], 3),
        width=info["width"],
        height=info["height"],
        frame_count=info["frame_count"],
        codec=info["codec"],
    )


def validate_fps(metadata: VideoMetadata, supported: list[float] | None = None) -> list[str]:
    supported = supported or [1, 2, 5, 10, 15, 24, 25, 30]
    if metadata.fps <= 0:
        return ["invalid_fps"]
    if metadata.codec == "npy":
        return []
    nearest = min(supported, key=lambda x: abs(x - metadata.fps))
    if abs(metadata.fps - nearest) > 5:
        return [f"fps_warning:{metadata.fps}"]
    return []


def validate_resolution(metadata: VideoMetadata, min_size: int = 64) -> list[str]:
    errors: list[str] = []
    if metadata.width < min_size or metadata.height < min_size:
        errors.append("resolution_too_small")
    return errors
