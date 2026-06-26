#!/usr/bin/env python3
"""Initialize feature store directory structure."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.feature_store.config import feature_store_paths

SUBDIRS = [
    "audio", "visual", "text", "image", "graph",
    "metadata", "clinical", "quality", "fusion",
    "statistics", "exports", "logs", "cache",
]
VIZ = ["visualizations"]


def main() -> None:
    paths = feature_store_paths()
    for sub in SUBDIRS:
        d = paths["root"] / sub
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.md"
        if not readme.exists():
            readme.write_text(f"# Feature store — {sub}\n", encoding="utf-8")
    for sub in VIZ:
        d = paths["statistics"] / sub
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.md"
        if not readme.exists():
            readme.write_text(f"# Visualizations — {sub}\n", encoding="utf-8")
    paths["cache"].mkdir(parents=True, exist_ok=True)
    print(f"Feature store ready at {paths['root']}")


if __name__ == "__main__":
    main()
