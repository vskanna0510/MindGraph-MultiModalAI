"""Prediction and risk heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_pipeline.models.base.classifier import BaseClassifier, MLPClassifier


class PredictionHead(MLPClassifier):
    """Depression classification head."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__(in_dim, hidden_dim, num_classes, dropout)


class RiskHead(BaseClassifier):
    """Multi-level risk prediction."""

    def __init__(self, in_dim: int, hidden_dim: int, risk_levels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, risk_levels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
