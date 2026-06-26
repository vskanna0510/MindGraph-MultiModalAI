"""Multi-format dataset index export."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from ml_pipeline.datasets.config import dataset_paths


def export_index(samples: list[dict[str, Any]], output_dir: Path | None = None) -> dict[str, Path]:
    """Export dataset index to CSV, JSON, Parquet, Feather, Pickle."""
    paths = dataset_paths()
    out = output_dir or paths["metadata"]
    out.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    df = pd.DataFrame(samples)

    csv_path = out / "dataset_index.csv"
    df.to_csv(csv_path, index=False)
    outputs["csv"] = csv_path

    json_path = out / "dataset_index.json"
    json_path.write_text(json.dumps(samples, indent=2, default=str), encoding="utf-8")
    outputs["json"] = json_path

    pickle_path = out / "dataset_index.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(samples, handle)
    outputs["pickle"] = pickle_path

    try:
        parquet_path = out / "dataset_index.parquet"
        df.to_parquet(parquet_path, index=False)
        outputs["parquet"] = parquet_path
    except (ImportError, ValueError):
        pass

    try:
        feather_path = out / "dataset_index.feather"
        df.to_feather(feather_path)
        outputs["feather"] = feather_path
    except (ImportError, ValueError):
        pass

    return outputs
