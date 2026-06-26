"""Uncertainty estimation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def predictive_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-8, 1.0)
    return float(-np.sum(p * np.log(p)))


def monte_carlo_dropout_uncertainty(model, batch, passes: int = 10) -> dict[str, float]:
    model.train()
    preds = []
    for _ in range(passes):
        with torch.no_grad():
            out = model(batch)
            preds.append(F.softmax(out["logits"], dim=-1).cpu().numpy())
    model.eval()
    stacked = np.stack(preds, axis=0)
    mean = stacked.mean(axis=0)
    epistemic = float(stacked.var(axis=0).mean())
    aleatoric = float((-mean * np.log(mean + 1e-8)).sum(axis=-1).mean())
    return {"epistemic": epistemic, "aleatoric": aleatoric, "total": epistemic + aleatoric}


def energy_score(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=-1)
