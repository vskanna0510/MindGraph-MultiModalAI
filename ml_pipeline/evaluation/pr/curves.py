"""Precision-recall analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def pr_curve_data(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(labels, probs)
        ap = float(average_precision_score(labels, probs))
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "average_precision": ap,
        }
    except Exception:
        return {"precision": [], "recall": [], "average_precision": 0.0}


def plot_pr(curves: list[dict[str, Any]], labels: list[str], path: Path, dpi: int = 300) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    for curve, name in zip(curves, labels):
        ax.plot(curve.get("recall", []), curve.get("precision", []), label=f"{name} (AP={curve.get('average_precision', 0):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
