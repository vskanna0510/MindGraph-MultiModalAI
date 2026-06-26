"""Temporal feature encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Encode minutes, days, weeks, months, elapsed and relative time."""

    def __init__(self, d_model: int, mode: str = "multi_scale") -> None:
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(6, d_model)
        self.out = nn.Linear(d_model, d_model)

    def _features(self, time_delta: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if time_delta is None:
            time_delta = torch.zeros(batch, device=device)
        if time_delta.dim() == 0:
            time_delta = time_delta.unsqueeze(0).expand(batch)
        minutes = time_delta / 60.0
        hours = time_delta / 3600.0
        days = time_delta / 86400.0
        weeks = days / 7.0
        months = days / 30.0
        rel = torch.sin(time_delta / 86400.0 * math.pi)
        return torch.stack([minutes, hours, days, weeks, months, rel], dim=-1)

    def forward(self, x: torch.Tensor, time_delta: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        feats = self._features(time_delta, b, x.device)
        return self.out(torch.relu(self.proj(feats)))
