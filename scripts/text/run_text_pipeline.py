#!/usr/bin/env python3
"""Run multilingual text engineering pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.text.loaders.datasets import DAICTextDataset, DVLOGTextDataset
from ml_pipeline.text.pipeline import TextPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="MindGraph++ text pipeline")
    parser.add_argument("--dataset", choices=["daic_woz", "dvlog", "both"], default="daic_woz")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--init-structure", action="store_true")
    args = parser.parse_args()

    if args.init_structure:
        from scripts.text.init_structure import main as init_main

        init_main()

    records: list[dict] = []
    if args.dataset in ("daic_woz", "both"):
        records.extend(DAICTextDataset().discover())
    if args.dataset in ("dvlog", "both"):
        records.extend(DVLOGTextDataset().discover())
    if args.limit > 0:
        records = records[: args.limit]

    print(f"Processing {len(records)} transcript files...")
    pipeline = TextPipeline()
    results = pipeline.process_batch(records)
    meta = pipeline.save_metadata_parquet(results)
    exports = pipeline.export_all(results)
    passed = sum(1 for r in results if r.validation_passed)
    print(f"Metadata: {meta}")
    print(f"Exports: {list(exports.keys())}")
    print(f"Validation passed: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
