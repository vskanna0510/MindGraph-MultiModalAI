"""Feature store visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def generate_visualizations(index_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if index_df.empty:
        return paths
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "quality_score" in index_df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(index_df["quality_score"].dropna(), bins=20, color="#4C72B0")
            ax.set_title("Quality Distribution")
            p = output_dir / "quality_histogram.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

        if "label" in index_df.columns:
            counts = index_df["label"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(counts.index.astype(str), counts.values, color="#55A868")
            ax.set_title("Class Balance")
            p = output_dir / "class_balance.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

        if all(c in index_df.columns for c in ("has_audio", "has_visual", "has_transcript")):
            mod = pd.DataFrame(
                {
                    "audio": index_df["has_audio"].astype(int),
                    "visual": index_df["has_visual"].astype(int),
                    "text": index_df["has_transcript"].astype(int),
                }
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(mod.T, aspect="auto", cmap="Blues")
            ax.set_yticks(range(3))
            ax.set_yticklabels(["audio", "visual", "text"])
            ax.set_title("Missing Modality Heatmap")
            p = output_dir / "missing_modality_heatmap.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
    except Exception:
        pass
    return paths
