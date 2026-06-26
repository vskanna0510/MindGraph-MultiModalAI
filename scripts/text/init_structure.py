#!/usr/bin/env python3
"""Initialize text pipeline directory structure."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.text.config import text_paths

DIRS = [
    "embeddings",
    "exports",
    "preprocessed",
    "features",
]

REPORT_DIRS = ["visualizations"]


def main() -> None:
    paths = text_paths()
    for sub in DIRS:
        (paths["processed"] / sub).mkdir(parents=True, exist_ok=True)
    for sub in REPORT_DIRS:
        (paths["reports"] / sub).mkdir(parents=True, exist_ok=True)
    (paths["cache"] / "text").mkdir(parents=True, exist_ok=True)
    readme = paths["processed"].parent / "README.md"
    if not readme.exists():
        readme.write_text("# Processed text artifacts\n", encoding="utf-8")
    print(f"Text pipeline structure ready under {paths['processed']}")


if __name__ == "__main__":
    main()
