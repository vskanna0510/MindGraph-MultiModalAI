"""Base classifier contract."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseClassifier(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""


class MLPClassifier(BaseClassifier):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
