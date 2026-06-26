#!/usr/bin/env python3
"""Benchmark fusion strategies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.models.config import model_config
from ml_pipeline.models.fusion.benchmark import benchmark_all_strategies, save_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fusion strategies")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/models/fusion_benchmark.json"))
    args = parser.parse_args()

    cfg = model_config()
    d_model = args.d_model or int(cfg.get("fusion", {}).get("d_model", 512))
    strategies = cfg.get("fusion", {}).get("benchmark", {}).get("strategies")
    results = benchmark_all_strategies(d_model=d_model, runs=args.runs, strategies=strategies)
    save_benchmark_report(results, args.output)

    print(f"Fusion benchmark (d_model={d_model})")
    for name, stats in sorted(results.items()):
        if "error" in stats:
            print(f"  {name}: ERROR — {stats['error']}")
        else:
            print(f"  {name}: {stats.get('mean_ms', 0):.2f} ms | params={int(stats.get('params', 0))}")


if __name__ == "__main__":
    main()
