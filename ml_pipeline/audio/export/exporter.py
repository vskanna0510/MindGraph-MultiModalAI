"""Stage 16 — Feature export."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ml_pipeline.audio.config import audio_paths
from ml_pipeline.audio.types import AudioPipelineResult


def export_artifacts(result: AudioPipelineResult, formats: list[str] | None = None) -> dict[str, Path]:
    """Export to numpy, torch, parquet, pickle."""
    paths = audio_paths()
    export_dir = paths["processed"] / "export" / result.participant_id
    export_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    formats = formats or ["numpy", "pickle"]

    if result.features and result.features.mfcc is not None:
        npy = export_dir / f"{result.participant_id}_mfcc.npy"
        np.save(npy, result.features.mfcc)
        outputs["numpy"] = npy

    if "pickle" in formats:
        pkl = export_dir / f"{result.participant_id}_pipeline.pkl"
        with pkl.open("wb") as handle:
            pickle.dump(result, handle)
        outputs["pickle"] = pkl

    if "torch" in formats and result.features and result.features.mfcc is not None:
        try:
            import torch

            pt = export_dir / f"{result.participant_id}_mfcc.pt"
            torch.save(torch.from_numpy(result.features.mfcc), pt)
            outputs["torch"] = pt
        except ImportError:
            pass

    return outputs
