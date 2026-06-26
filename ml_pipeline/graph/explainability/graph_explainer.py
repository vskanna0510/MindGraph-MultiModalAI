"""Graph-level explainability."""

from __future__ import annotations

from typing import Any

import torch

from ml_pipeline.graph.reasoning.trends import reason_over_history


def explain_graph_prediction(
    kg_vector: torch.Tensor | None,
    modality_weights: dict[str, torch.Tensor] | None = None,
    risk_history: list[float] | None = None,
) -> dict[str, Any]:
    explanation: dict[str, Any] = {"source": "graph_reasoning"}
    if modality_weights:
        explanation["modality_influence"] = {k: float(v.mean()) for k, v in modality_weights.items()}
    if risk_history:
        explanation["longitudinal"] = reason_over_history(risk_history)
    if kg_vector is not None:
        explanation["kg_activation"] = float(kg_vector.norm().item())
    return explanation
