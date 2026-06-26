#!/usr/bin/env python3
"""Validate and benchmark MindGraph multimodal model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.models.card import ModelCard
from ml_pipeline.models.config import model_paths
from ml_pipeline.models.multimodal import MindGraphMultimodal
from ml_pipeline.models.registry import ModelRegistry
from ml_pipeline.models.types import MultimodalBatch

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="MindGraph++ model validation")
    parser.add_argument("--init-structure", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--export", choices=["torchscript", "onnx"], default=None)
    args = parser.parse_args()

    if args.init_structure:
        from scripts.models.init_structure import main as init_main

        init_main()

    model = MindGraphMultimodal()
    batch = model._dummy_batch()
    ok, errs = model.validate()
    print(f"Validation: {'PASS' if ok else 'FAIL'}")
    if errs:
        print(f"  Errors: {errs}")

    summary = model.summary()
    print(f"Parameters: {summary['parameters']:,}")

    if args.benchmark:
        bench = model.benchmark(batch)
        print(f"Benchmark: {bench['mean_ms']:.2f} ms (mean)")

    registry = ModelRegistry()
    entry = registry.get("MODEL_010")
    if entry:
        entry.performance = {"validation": ok}
        registry.save()

    card = ModelCard(
        overview="MindGraph++ multimodal depression detection architecture",
        architecture="Independent encoders + cross-modal transformer + GNN",
        input_spec={"modalities": ["audio", "visual", "text", "image"]},
        output_spec={"logits": "[B, 2]", "risk_logits": "[B, 3]"},
        training_dataset="DAIC-WOZ, D-VLOG",
        export_formats=["pytorch", "torchscript", "onnx"],
    )
    card_path = model_paths()["experiments"] / "MODEL_010_card.md"
    card.save(card_path)
    print(f"Model card: {card_path}")

    if args.export:
        out = model_paths()["weights"] / f"MODEL_010.{args.export}"
        path = model.export(out, fmt=args.export)
        print(f"Export ({args.export}): {path}")


if __name__ == "__main__":
    main()
