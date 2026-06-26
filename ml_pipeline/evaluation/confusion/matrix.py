"""Confusion matrix generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def confusion_matrix_data(preds: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    valid = labels >= 0
    preds, labels = preds[valid], labels[valid]
    classes = sorted(set(labels.tolist()) | set(preds.tolist()))
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for p, y in zip(preds, labels):
        cm[idx[int(p)], idx[int(y)]] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    col_sum = cm.sum(axis=0, keepdims=True)
    total = cm.sum() or 1
    return {
        "matrix": cm.tolist(),
        "normalized_row": (cm / np.maximum(row_sum, 1)).tolist(),
        "normalized_col": (cm / np.maximum(col_sum, 1)).tolist(),
        "percentage": (cm / total).tolist(),
        "classes": classes,
    }


def plot_confusion_matrix(data: dict[str, Any], path: Path, title: str = "Confusion Matrix", dpi: int = 300) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    cm = np.array(data["matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(data["classes"])))
    ax.set_yticks(range(len(data["classes"])))
    ax.set_xticklabels(data["classes"])
    ax.set_yticklabels(data["classes"])
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
