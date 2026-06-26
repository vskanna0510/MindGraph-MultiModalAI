#!/usr/bin/env python3
"""Initialize data quality report directories."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.data_quality.config import data_quality_paths

README = "# {name}\n\nAutomated data quality outputs (MP2 Part 7).\n"


def main() -> None:
    paths = data_quality_paths()
    for name, p in paths.items():
        if name in {"dataset_index", "feature_sessions", "ci_report", "passport", "audit_log"}:
            continue
        p.mkdir(parents=True, exist_ok=True)
        readme = p / "README.md"
        if not readme.exists():
            readme.write_text(README.format(name=name), encoding="utf-8")
    print(f"Data quality directories ready at {paths['reports']}")


if __name__ == "__main__":
    main()
