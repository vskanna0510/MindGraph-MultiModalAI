"""Export video pipeline artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ml_pipeline.video.config import video_paths
from ml_pipeline.video.types import VideoPipelineResult


def export_visual_artifacts(result: VideoPipelineResult) -> dict[str, Path]:
    paths = video_paths()
    out_dir = paths["processed"] / "export" / result.participant_id
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    if result.facial_features:
        feat_path = out_dir / f"{result.participant_id}_facial_features.npy"
        np.save(
            feat_path,
            {
                "ear": result.facial_features.ear,
                "mar": result.facial_features.mar,
                "blink_rate": result.facial_features.blink_rate,
            },
            allow_pickle=True,
        )
        outputs["numpy"] = feat_path

    pkl = out_dir / f"{result.participant_id}_video_pipeline.pkl"
    with pkl.open("wb") as handle:
        pickle.dump(result, handle)
    outputs["pickle"] = pkl
    return outputs
