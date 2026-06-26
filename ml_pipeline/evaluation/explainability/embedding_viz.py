"""Embedding visualization (PCA / UMAP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def reduce_embeddings(embeddings: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
    if embeddings.shape[0] < 2:
        return embeddings[:, :n_components] if embeddings.shape[1] >= n_components else embeddings
    if method == "umap":
        try:
            from umap import UMAP

            return UMAP(n_components=n_components, random_state=42).fit_transform(embeddings)
        except ImportError:
            pass
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            return TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings) - 1)).fit_transform(embeddings)
        except Exception:
            pass
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, random_state=42).fit_transform(embeddings)


def plot_embedding_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    path: Path,
    title: str = "Embedding Projection",
    dpi: int = 300,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="coolwarm", alpha=0.7, s=20)
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
