#!/usr/bin/env python3
"""Run the full dataset preparation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.datasets.pipeline import DatasetPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="MindGraph++ dataset pipeline")
    parser.add_argument("--datasets", nargs="*", default=None, help="Adapter names")
    parser.add_argument("--init-structure", action="store_true", help="Create dataset dirs")
    parser.add_argument("--skip-processing", action="store_true", help="Run discovery through metadata only")
    args = parser.parse_args()

    if args.init_structure:
        from scripts.datasets.init_structure import main as init_main

        init_main()

    pipeline = DatasetPipeline(datasets=args.datasets)
    result = pipeline.run(skip_processing=args.skip_processing)
    print(f"Pipeline complete. Steps: {result.steps_completed}")
    print(f"Samples: {len(result.records)}")
    if result.verification:
        print(f"Verification passed: {result.verification.get('passed')}")


if __name__ == "__main__":
    main()
