#!/usr/bin/env python3
"""Initialize ml_pipeline/video/ documentation tree."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = ROOT / "ml_pipeline" / "video"
DIRS = [
    "config", "loaders", "preprocess", "sampling", "face_detection", "face_alignment",
    "mediapipe", "tracking", "quality", "features", "embeddings", "augmentation",
    "normalization", "visualization", "statistics", "cache", "export", "tests", "reports",
]


def main() -> None:
    for name in DIRS:
        d = VIDEO_ROOT / name
        d.mkdir(parents=True, exist_ok=True)
        for doc, body in [
            ("README.md", f"# Video — {name}\n\nOwner: `ml_pipeline.video.{name}`\n"),
            ("architecture.md", f"# Architecture\n\nComponent: `{name}`\n"),
            ("workflow.md", f"# Workflow\n\nStage in `VideoPipeline`.\n"),
        ]:
            p = d / doc
            if not p.exists():
                p.write_text(body, encoding="utf-8")
    readme = VIDEO_ROOT / "README.md"
    if not readme.exists():
        readme.write_text("# Video Pipeline\n\nSee `configs/video_pipeline.yaml`.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
