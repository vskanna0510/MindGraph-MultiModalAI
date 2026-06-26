#!/usr/bin/env python3
"""Run model evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from ml_pipeline.evaluation.config import evaluation_paths
from ml_pipeline.evaluation.evaluator import ModelEvaluator
from ml_pipeline.evaluation.reports.research_report import generate_research_report
from ml_pipeline.feature_store.datasets.daic import DAICDataset
from ml_pipeline.feature_store.dataloader import build_dataloader
from ml_pipeline.models.config import model_config
from ml_pipeline.models.multimodal import MindGraphMultimodal


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MindGraph++ model")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    model = MindGraphMultimodal(model_config())
    if args.checkpoint and args.checkpoint.exists():
        loaded = MindGraphMultimodal.load(args.checkpoint)
        model.load_state_dict(loaded.state_dict())

    dataset = DAICDataset(split=args.split)
    loader = build_dataloader(dataset, batch_size=8, num_workers=0, shuffle=False)
    evaluator = ModelEvaluator(model)

    if loader is None or len(dataset) == 0:
        print("No data — running synthetic evaluation")
        batch = model._dummy_batch()
        batch.labels = torch.tensor([0])
        out = model(batch)
        expl = evaluator.explain_batch(batch)
        print(json.dumps({"logits_shape": list(out["logits"].shape), "explanation_keys": list(expl.keys())}, indent=2))
        return

    try:
        results = evaluator.evaluate(loader, split=args.split)
    except RuntimeError as exc:
        print(f"Real-data evaluation failed ({exc}) — falling back to synthetic batch")
        batch = model._dummy_batch()
        batch.labels = torch.tensor([0])
        out = model(batch)
        expl = evaluator.explain_batch(batch)
        results = {
            "split": args.split,
            "metrics": {"f1": 0.0, "note": "synthetic_fallback"},
            "error_analysis": {"recommendations": ["Fix feature-store tensor dimensions for full evaluation."]},
            "explanation_sample": expl,
        }
    results["checklist"] = evaluator.publication_checklist(results)
    if args.ablation:
        results["ablation"] = evaluator.run_ablation(loader)

    paths = evaluation_paths()
    report = generate_research_report(results, paths["root"])
    print(f"Evaluation complete — F1={results.get('metrics', {}).get('f1', 0):.4f}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
