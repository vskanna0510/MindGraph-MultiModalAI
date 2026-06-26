#!/usr/bin/env python3
"""MindGraph++ training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from ml_pipeline.feature_store.datasets.daic import DAICDataset
from ml_pipeline.feature_store.dataloader import build_dataloader
from ml_pipeline.models.config import model_config
from ml_pipeline.models.multimodal import MindGraphMultimodal
from ml_pipeline.training.config import training_config
from ml_pipeline.training.trainer import Trainer
from ml_pipeline.utils.reproducibility import set_global_seeds, SeedBundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MindGraph++ multimodal model")
    parser.add_argument("--config", type=Path, default=Path("configs/training_pipeline.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = training_config()
    mcfg = model_config()
    if args.epochs:
        cfg["training"]["max_epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    seed = int(cfg["training"].get("seed", 42))
    set_global_seeds(SeedBundle(python=seed, numpy=seed, torch=seed))

    model = MindGraphMultimodal(mcfg)
    dataset = DAICDataset(split="train")
    val_dataset = DAICDataset(split="val") if hasattr(DAICDataset, "__init__") else dataset

    bs = int(cfg["training"].get("batch_size", 16))
    nw = int(cfg["training"].get("num_workers", 0))
    train_loader = build_dataloader(dataset, batch_size=bs, num_workers=nw, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=bs, num_workers=nw, shuffle=False)

    if args.dry_run:
        out = model(model._dummy_batch())
        print(f"Dry run OK — logits {out['logits'].shape}")
        if "risk_score" in out:
            print(f"  risk_score: {out['risk_score'].shape}, confidence: {out['confidence'].shape}")
        return

    if train_loader is None:
        print("No train loader available.")
        return

    trainer = Trainer(model, cfg, train_loader, val_loader)
    result = trainer.fit(resume_path=args.resume)
    print(f"Training complete. Best metrics: {result.get('best', {})}")


if __name__ == "__main__":
    main()
