"""Data quality visualizations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_figures(sessions_df: pd.DataFrame, quality_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "label" in sessions_df.columns:
            counts = sessions_df["label"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
            ax.set_title("Class Distribution")
            for fmt in ("png", "svg"):
                p = output_dir / f"class_distribution.{fmt}"
                fig.savefig(p, dpi=100, bbox_inches="tight")
                paths.append(p)
            plt.close(fig)

        if not quality_df.empty and "overall" in quality_df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(quality_df["overall"], bins=20, color="#55A868")
            ax.set_title("Quality Histogram")
            p = output_dir / "quality_histogram.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

        if all(c in sessions_df.columns for c in ("has_audio", "has_visual", "has_transcript")):
            mod = pd.DataFrame(
                {
                    "audio": sessions_df["has_audio"].astype(int),
                    "visual": sessions_df["has_visual"].astype(int),
                    "text": sessions_df["has_transcript"].astype(int),
                }
            )
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.imshow(mod.T, aspect="auto", cmap="Blues")
            ax.set_yticks(range(3))
            ax.set_yticklabels(["audio", "visual", "text"])
            ax.set_title("Missing Modality Heatmap")
            p = output_dir / "missing_modality_heatmap.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

        cols = [c for c in ("audio_length", "quality_score", "sync_score") if c in sessions_df.columns]
        if len(cols) >= 2:
            corr = sessions_df[cols].apply(pd.to_numeric, errors="coerce").corr()
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45)
            ax.set_yticklabels(cols)
            ax.set_title("Correlation Matrix")
            fig.colorbar(im)
            p = output_dir / "correlation_matrix.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
    except Exception:
        pass
    return paths
