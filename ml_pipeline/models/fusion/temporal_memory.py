"""Temporal memory for variable session history."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMemory(nn.Module):
    """Maintain historical session embeddings with truncation."""

    def __init__(self, d_model: int, max_sessions: int = 12) -> None:
        super().__init__()
        self.max_sessions = max_sessions
        self.d_model = d_model
        self.register_buffer("_empty", torch.zeros(0))

    def build_sequence(
        self,
        current: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history is None or history.numel() == 0:
            return current.unsqueeze(1)
        seq = torch.cat([history, current.unsqueeze(1)], dim=1)
        return seq[:, -self.max_sessions :]

    def truncate(self, history: torch.Tensor, keep: int) -> torch.Tensor:
        return history[:, -keep:]

    def random_drop(self, history: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
        if not training or drop_prob <= 0:
            return history
        mask = torch.rand(history.shape[1], device=history.device) > drop_prob
        return history[:, mask]
