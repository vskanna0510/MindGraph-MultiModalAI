"""Export engine for training, evaluation, deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.feature_store.reproducibility import build_reproducibility_manifest


def export_for_training(sessions: list[dict[str, Any]], output_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    df = pd.DataFrame(sessions)
    csv_path = output_dir / "training_manifest.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    try:
        pq = output_dir / "training_manifest.parquet"
        df.to_parquet(pq, index=False)
        paths["parquet"] = pq
    except ImportError:
        pass
    meta_path = output_dir / "reproducibility.json"
    meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["reproducibility"] = meta_path
    return paths


def export_numpy_bundle(batches: list[Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "participant_ids": [b.participant_id for b in batches],
        "labels": [b.label for b in batches],
    }
    path = output_dir / "session_bundle.npz"
    np.savez(path, **{k: np.array(v) for k, v in payload.items()})
    return path


def export_torch_checkpoint(state: dict[str, Any], output_dir: Path) -> Path | None:
    try:
        import torch

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "feature_store_checkpoint.pt"
        torch.save(state, path)
        return path
    except ImportError:
        return None
