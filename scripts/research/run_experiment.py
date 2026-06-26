#!/usr/bin/env python3
"""Run or initialize a MindGraph++ research experiment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.evaluation.report_generator import generate_experiment_report
from ml_pipeline.experiments.experiment_tracker import ExperimentTracker
from ml_pipeline.utils.reproducibility import load_ml_config, seeds_from_config, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MindGraph++ experiment runner")
    parser.add_argument("--config", default="configs/ml/baseline.yaml", help="ML config YAML path")
    parser.add_argument("--experiment-id", default=None, help="Optional fixed experiment ID")
    parser.add_argument("--model-version", default="MODEL_001_BASELINE", help="Model registry version")
    parser.add_argument("--init-only", action="store_true", help="Create experiment folder only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_ml_config(str(config_path))
    seeds = seeds_from_config(config)
    deterministic = config.get("reproducibility", {}).get("deterministic", True)
    set_global_seeds(seeds, deterministic=deterministic)

    tracker = ExperimentTracker(ROOT / "experiments")
    exp_dir = tracker.create_experiment(
        config_path=config_path,
        experiment_id=args.experiment_id,
        model_version=args.model_version,
    )
    print(f"Experiment initialized: {exp_dir}")

    if args.init_only:
        generate_experiment_report(exp_dir, ROOT / "reports" / "experiment")
        return 0

    # Training integration point — executed when ML trainers are implemented.
    start = time.perf_counter()
    validation_metrics = {"status": "pending_training_pipeline"}
    test_metrics = {"status": "pending_training_pipeline"}
    tracker.finalize(exp_dir, validation_metrics, test_metrics, start)
    report = generate_experiment_report(exp_dir, ROOT / "reports" / "experiment")
    print(f"Report generated: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
