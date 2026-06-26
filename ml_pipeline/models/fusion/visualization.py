"""Fusion attention and embedding visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ml_pipeline.models.fusion.output import FusionOutput


def _safe_import_plt():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_modality_weights(weights: dict[str, torch.Tensor], path: Path) -> Path | None:
    plt = _safe_import_plt()
    if plt is None or not weights:
        return None
    names = list(weights.keys())
    vals = [weights[n].detach().float().mean().item() for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals)
    ax.set_title("Modality Importance")
    ax.set_ylabel("Mean weight")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_attention_heatmap(attn: torch.Tensor, path: Path, title: str = "Attention") -> Path | None:
    plt = _safe_import_plt()
    if plt is None:
        return None
    data = attn.detach().float().cpu()
    if data.dim() > 2:
        data = data.mean(dim=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data.numpy(), aspect="auto", cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def export_fusion_report(output: FusionOutput, output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    meta_path = output_dir / "fusion_metadata.json"
    meta: dict[str, Any] = {"metadata": output.metadata}
    if output.modality_weights:
        meta["modality_weights"] = {k: v.detach().float().mean().item() for k, v in output.modality_weights.items()}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths.append(meta_path)

    wpath = output_dir / "modality_weights.png"
    if plot_modality_weights(output.modality_weights, wpath):
        paths.append(wpath)

    for name, attn in output.attention_maps.items():
        apath = output_dir / f"attention_{name}.png"
        if plot_attention_heatmap(attn, apath, title=name):
            paths.append(apath)

    if output.fusion_embedding is not None:
        emb_path = output_dir / "fusion_embedding.pt"
        torch.save(output.fusion_embedding.detach().cpu(), emb_path)
        paths.append(emb_path)

    return paths


def visualize_fusion(output: FusionOutput, output_dir: Path) -> list[Path]:
    return export_fusion_report(output, output_dir)
