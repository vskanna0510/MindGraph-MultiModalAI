"""Global and local explainability."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def modality_contribution(
    modality_weights: dict[str, torch.Tensor] | None,
    fusion_weights: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    if modality_weights:
        for k, v in modality_weights.items():
            scores[k] = float(v.detach().float().mean().cpu())
    if fusion_weights:
        for k, v in fusion_weights.items():
            scores[k] = scores.get(k, 0.0) + float(v.detach().float().mean().cpu())
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


def permutation_importance(model, batch, labels: torch.Tensor, baseline_metric: float, n_repeats: int = 3) -> dict[str, float]:
    """Simple modality dropout importance."""
    from ml_pipeline.training.metrics import metrics_from_outputs

    importances: dict[str, float] = {}
    base_out = model(batch)
    base_f1 = metrics_from_outputs(base_out, labels).get("f1", baseline_metric)
    for mod in ("audio", "visual", "text", "image"):
        key = f"{mod}_presence"
        if key not in batch.modality_presence and mod not in batch.modality_presence:
            continue
        drops = []
        for _ in range(n_repeats):
            b2 = batch
            mp = dict(batch.modality_presence)
            mp[mod] = torch.zeros_like(mp.get(mod, torch.zeros(1)))
            from ml_pipeline.models.types import MultimodalBatch

            b2 = MultimodalBatch(
                audio=batch.audio,
                visual=batch.visual,
                text=batch.text,
                image=batch.image,
                graph=batch.graph,
                audio_mask=batch.audio_mask,
                visual_mask=batch.visual_mask,
                text_mask=batch.text_mask,
                modality_presence=mp,
                labels=batch.labels,
                metadata=batch.metadata,
            )
            out = model(b2)
            f1 = metrics_from_outputs(out, labels).get("f1", 0.0)
            drops.append(base_f1 - f1)
        importances[mod] = float(np.mean(drops)) if drops else 0.0
    return importances


def local_explanation(
    model_output: dict[str, Any],
    modality_contrib: dict[str, float],
    graph_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "prediction": model_output.get("preds", model_output.get("logits")),
        "confidence": model_output.get("confidence"),
        "modality_contribution": modality_contrib,
        "top_modality": max(modality_contrib, key=modality_contrib.get) if modality_contrib else None,
        "risk_score": model_output.get("risk_score"),
        "graph": graph_meta or {},
        "temporal_risks": model_output.get("temporal_risks"),
    }


def attention_summary(attention_maps: dict[str, torch.Tensor] | None) -> dict[str, float]:
    if not attention_maps:
        return {}
    return {k: float(v.detach().float().mean().cpu()) for k, v in attention_maps.items()}
