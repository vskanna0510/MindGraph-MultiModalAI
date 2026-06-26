#!/usr/bin/env python3
"""Initialize ml_pipeline/audio/ documentation tree."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIO_ROOT = ROOT / "ml_pipeline" / "audio"

DIRS = [
    "config",
    "loaders",
    "preprocess",
    "augmentation",
    "segmentation",
    "normalization",
    "features",
    "embeddings",
    "quality",
    "cache",
    "export",
    "utils",
    "visualization",
    "statistics",
    "tests",
    "reports",
]

TEMPLATE = """# {title}

## Purpose
{purpose}

## Architecture
See `architecture.md`.

## Workflow
See `workflow.md`.
"""


def main() -> None:
    for name in DIRS:
        d = AUDIO_ROOT / name
        d.mkdir(parents=True, exist_ok=True)
        for doc in ("README.md", "architecture.md", "workflow.md"):
            path = d / doc
            if not path.exists():
                title = f"Audio — {name}"
                purpose = f"Module: ml_pipeline.audio.{name}"
                if doc == "architecture.md":
                    path.write_text(f"# Architecture\n\nComponent: `{name}` in the audio pipeline.\n", encoding="utf-8")
                elif doc == "workflow.md":
                    path.write_text(f"# Workflow\n\nInvoked by `AudioPipeline` stage orchestration.\n", encoding="utf-8")
                else:
                    path.write_text(TEMPLATE.format(title=title, purpose=purpose), encoding="utf-8")

    root_readme = AUDIO_ROOT / "README.md"
    if not root_readme.exists():
        root_readme.write_text(
            "# Audio Engineering Pipeline\n\nSee `configs/audio_pipeline.yaml` and `scripts/audio/run_audio_pipeline.py`.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
