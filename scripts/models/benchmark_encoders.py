#!/usr/bin/env python3
"""Benchmark encoders independently."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from ml_pipeline.models.audio.encoder import build_audio_encoder
from ml_pipeline.models.config import model_config
from ml_pipeline.models.encoders.benchmark import benchmark_encoder
from ml_pipeline.models.image.encoder import build_image_encoder
from ml_pipeline.models.text.encoder import build_text_encoder
from ml_pipeline.models.visual.encoder import build_visual_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark modality encoders")
    parser.add_argument("--modality", choices=["audio", "visual", "text", "image", "all"], default="all")
    args = parser.parse_args()
    cfg = model_config()

    builders = {
        "audio": (build_audio_encoder, cfg.get("audio", {}), (1, 32, 512)),
        "visual": (build_visual_encoder, cfg.get("visual", {}), (1, 16, 512)),
        "text": (build_text_encoder, cfg.get("text", {}), (1, 24, 512)),
        "image": (build_image_encoder, cfg.get("image", {}), (1, 8, 512)),
    }
    mods = list(builders.keys()) if args.modality == "all" else [args.modality]
    for name in mods:
        build_fn, mcfg, shape = builders[name]
        enc = build_fn(mcfg)
        x = torch.randn(*shape)
        mask = torch.ones(shape[0], shape[1])
        stats = benchmark_encoder(enc, x, mask, runs=5)
        print(f"\n{name.upper()} encoder ({enc.__class__.__name__})")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
