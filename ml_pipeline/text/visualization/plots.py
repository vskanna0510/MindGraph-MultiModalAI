"""Text pipeline visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_visualizations(results: list[Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lengths = [len(r.tokens) for r in results]
        if lengths:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(lengths, bins=20, color="#4C72B0")
            ax.set_title("Token Length Distribution")
            ax.set_xlabel("Tokens")
            p = output_dir / "token_length_histogram.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

        sentiments = [r.sentiment.get("label", "neutral") for r in results]
        if sentiments:
            from collections import Counter

            counts = Counter(sentiments)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(list(counts.keys()), list(counts.values()), color="#55A868")
            ax.set_title("Sentiment Distribution")
            p = output_dir / "sentiment_distribution.png"
            fig.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
    except Exception:
        pass
    return paths
