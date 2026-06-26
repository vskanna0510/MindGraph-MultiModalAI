"""Dataset visualization generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_visualizations(stats: dict[str, Any], output_dir: Path) -> list[Path]:
    """Generate statistics charts; skips gracefully if matplotlib unavailable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return created

    def _bar_chart(data: dict, title: str, filename: str) -> None:
        if not data:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(list(data.keys()), list(data.values()))
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        created.append(path)

    def _histogram(values: list[float], title: str, filename: str) -> None:
        if not values:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(values, bins=20, edgecolor="black")
        ax.set_title(title)
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        created.append(path)

    _bar_chart(stats.get("label_distribution", {}), "Class Distribution", "class_distribution.png")
    _bar_chart(stats.get("split_distribution", {}), "Session Distribution", "session_distribution.png")
    _histogram(
        [stats.get("average_duration", 0)] * max(stats.get("participant_count", 1), 1),
        "Duration Histogram",
        "duration_histogram.png",
    )
    return created
