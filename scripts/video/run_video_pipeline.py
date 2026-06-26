#!/usr/bin/env python3
"""Run video preprocessing pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.video.loaders.datasets import DAICVisualDataset, DVLOGVisualDataset
from ml_pipeline.video.pipeline import VideoPipeline
from ml_pipeline.video.statistics.stats import write_video_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="MindGraph++ video pipeline")
    parser.add_argument("--dataset", choices=["daic_woz", "dvlog", "both"], default="dvlog")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records: list[dict] = []
    if args.dataset in ("daic_woz", "both"):
        records.extend(DAICVisualDataset().discover())
    if args.dataset in ("dvlog", "both"):
        records.extend(DVLOGVisualDataset().discover())
    if args.limit > 0:
        records = records[: args.limit]

    print(f"Processing {len(records)} visual files...")
    pipeline = VideoPipeline()
    results = pipeline.process_batch(records)
    meta = pipeline.save_metadata_parquet(results)
    stats = write_video_statistics(results)
    passed = sum(1 for r in results if r.validation_passed)
    print(f"Metadata: {meta}")
    print(f"Statistics: {stats}")
    print(f"Validation passed: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
