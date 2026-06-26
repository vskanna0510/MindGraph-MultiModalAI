#!/usr/bin/env python3
"""Run full feature store pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.feature_store.pipeline import FeatureStorePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="MindGraph++ feature store pipeline")
    parser.add_argument("--init-structure", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--stage", choices=["all", "index", "validate", "stats", "export"], default="all")
    args = parser.parse_args()

    if args.init_structure:
        from scripts.feature_store.init_structure import main as init_main

        init_main()

    pipeline = FeatureStorePipeline()

    if args.stage == "index":
        path = pipeline.build_index()
        print(f"Index: {path}")
        return
    if args.stage == "validate":
        results, report = pipeline.run_validation(limit=args.limit)
        passed = sum(1 for r in results if r["passed"])
        print(f"Validation: {passed}/{len(results)} — {report}")
        return
    if args.stage == "stats":
        stats = pipeline.run_statistics()
        print(f"Statistics: {stats}")
        return
    if args.stage == "export":
        exports = pipeline.run_export()
        print(f"Exports: {exports}")
        return

    summary = pipeline.run_all(limit=args.limit)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
