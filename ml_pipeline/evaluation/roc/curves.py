"""ROC curve analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def roc_curve_data(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    try:
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, thresholds = roc_curve(labels, probs)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist(), "auc": float(auc(fpr, tpr))}
    except Exception:
        return {"fpr": [], "tpr": [], "auc": 0.0}


def plot_roc(curves: list[dict[str, Any]], labels: list[str], path: Path, dpi: int = 300) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    for curve, name in zip(curves, labels):
        ax.plot(curve.get("fpr", []), curve.get("tpr", []), label=f"{name} (AUC={curve.get('auc', 0):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
