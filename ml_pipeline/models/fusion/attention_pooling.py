"""Fusion attention pooling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionAttentionPooling(nn.Module):
    def __init__(self, d_model: int, mode: str = "attention") -> None:
        super().__init__()
        self.mode = mode
        self.attn = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "mean":
            return hidden.mean(dim=1), torch.ones(hidden.shape[0], hidden.shape[1], device=hidden.device) / hidden.shape[1]
        if self.mode == "max":
            return hidden.max(dim=1).values, torch.zeros(hidden.shape[0], hidden.shape[1], device=hidden.device)
        scores = self.attn(hidden).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled, weights
