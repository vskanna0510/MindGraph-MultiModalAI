"""Attention pooling strategies — configurable via YAML."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Mean, max, attention, or CLS pooling."""

    def __init__(self, dim: int, mode: str = "attention") -> None:
        super().__init__()
        self.mode = mode
        self.attn = nn.Linear(dim, 1)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (B, T, D)
        Returns:
            pooled (B, D), attention_weights (B, T)
        """
        if self.mode == "cls":
            pooled = hidden[:, 0]
            weights = torch.zeros(hidden.shape[0], hidden.shape[1], device=hidden.device)
            weights[:, 0] = 1.0
            return pooled, weights
        if self.mode == "mean":
            if mask is not None:
                m = mask.unsqueeze(-1)
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
            else:
                pooled = hidden.mean(dim=1)
            weights = torch.full((hidden.shape[0], hidden.shape[1]), 1.0 / hidden.shape[1], device=hidden.device)
            return pooled, weights
        if self.mode == "max":
            pooled = hidden.max(dim=1).values
            weights = (hidden == pooled.unsqueeze(1)).float().mean(dim=-1)
            return pooled, weights
        # attention pooling
        scores = self.attn(hidden).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled, weights
