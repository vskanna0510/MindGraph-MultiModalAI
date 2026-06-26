"""Feature export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def export_features(rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    df = pd.DataFrame(rows)

    if "csv" in formats:
        p = output_dir / "text_features.csv"
        df.to_csv(p, index=False)
        paths["csv"] = p
    if "json" in formats:
        p = output_dir / "text_features.json"
        p.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        paths["json"] = p
    if "parquet" in formats:
        p = output_dir / "text_features.parquet"
        try:
            df.to_parquet(p, index=False)
            paths["parquet"] = p
        except ImportError:
            pass
    if "numpy" in formats and rows:
        p = output_dir / "text_features_meta.npy"
        np.save(p, np.array([str(r) for r in rows], dtype=object))
        paths["numpy"] = p
    if "pickle" in formats:
        p = output_dir / "text_features.pkl"
        df.to_pickle(p)
        paths["pickle"] = p
    return paths
