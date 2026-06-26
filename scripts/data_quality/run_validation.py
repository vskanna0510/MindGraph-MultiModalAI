#!/usr/bin/env python3
"""Run data quality governance pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.data_quality.pipeline import DataQualityPipeline
from ml_pipeline.data_quality.types import GateStatus


def main() -> int:
    parser = argparse.ArgumentParser(description="MindGraph++ data quality validation")
    parser.add_argument("--init-structure", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ci", action="store_true", help="Exit non-zero if validation fails")
    args = parser.parse_args()

    if args.init_structure:
        from scripts.data_quality.init_structure import main as init_main

        init_main()

    pipeline = DataQualityPipeline()
    result = pipeline.run(limit=args.limit)

    print(f"Approved: {result.approved}")
    print(f"Gates: {result.passed_gates}/{len(result.gates)} passed")
    for gate in result.gates:
        print(f"  - {gate.gate}: {gate.status.value} ({len(gate.issues)} issues)")
    print(f"Passport: {result.report_paths.get('ci', '').replace('ci_validation_report.json', 'dataset_passport.json')}")
    for k, v in result.report_paths.items():
        print(f"  {k}: {v}")

    if args.ci and not result.approved:
        return 1
    if args.ci and any(g.status == GateStatus.FAILED for g in result.gates):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
