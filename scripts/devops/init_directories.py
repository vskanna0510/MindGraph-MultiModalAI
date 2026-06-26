#!/usr/bin/env python3
"""Create required runtime directories."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config.settings import get_settings  # noqa: E402
from app.core.startup import ensure_directories  # noqa: E402

if __name__ == "__main__":
    created = ensure_directories(get_settings())
    print(f"Created {len(created)} directories")
