#!/usr/bin/env python3
"""Build feature store session index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.feature_store.pipeline import FeatureStorePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature store index")
    parser.add_argument("--dataset", choices=["combined", "daic_woz", "dvlog"], default="combined")
    args = parser.parse_args()
    pipeline = FeatureStorePipeline()
    path = pipeline.build_index(args.dataset)
    print(f"Session index: {path}")


if __name__ == "__main__":
    main()
