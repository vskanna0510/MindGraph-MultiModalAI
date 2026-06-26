"""Explainability engine."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.base.explainability import BaseExplainability


class ExplainabilityEngine(BaseExplainability):
    """Modality attribution via gradient-free attention weights."""

    def __init__(self, modalities: list[str]) -> None:
        super().__init__()
        self.modalities = modalities
        self.attribution = nn.Linear(len(modalities), len(modalities), bias=False)
        nn.init.eye_(self.attribution.weight)

    def explain(self, features: dict[str, torch.Tensor], prediction: torch.Tensor, fusion_output=None) -> dict[str, Any]:
        scores: dict[str, float] = {}
        for i, mod in enumerate(self.modalities):
            if mod in features and features[mod] is not None:
                scores[mod] = float(features[mod].abs().mean().detach().cpu())
        total = sum(scores.values()) or 1.0
        attribution = {k: v / total for k, v in scores.items()}
        result = {
            "attribution": attribution,
            "prediction": prediction.detach().cpu().tolist() if hasattr(prediction, "detach") else prediction,
            "top_modality": max(attribution, key=attribution.get) if attribution else None,
            "why": f"Prediction driven primarily by {max(attribution, key=attribution.get)}" if attribution else "Insufficient modality signal",
        }
        if fusion_output is not None and hasattr(fusion_output, "attention_maps"):
            result["attention_layers"] = list(fusion_output.attention_maps.keys())
        return result

    def forward(self, modality_scores: torch.Tensor) -> torch.Tensor:
        return self.attribution(modality_scores)
