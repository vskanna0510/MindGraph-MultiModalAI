"""Positional encoding variants."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, mode: str = "sinusoidal") -> None:
        super().__init__()
        self.mode = mode
        if mode == "learnable":
            self.pe = nn.Embedding(max_len, d_model)
        elif mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("sin_pe", pe.unsqueeze(0))
        else:
            self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        if self.mode == "sinusoidal":
            return x + self.sin_pe[:, :seq_len]
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        return x + self.pe(positions.clamp(0, self.pe.num_embeddings - 1))


class TemporalPositionalEncoding(nn.Module):
    """Encode session order and elapsed time."""

    def __init__(self, d_model: int, max_sessions: int = 12) -> None:
        super().__init__()
        self.order_embed = nn.Embedding(max_sessions, d_model)
        self.time_proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor, session_idx: torch.Tensor | None = None, elapsed_days: torch.Tensor | None = None) -> torch.Tensor:
        if session_idx is not None:
            x = x + self.order_embed(session_idx.clamp(0, self.order_embed.num_embeddings - 1))
        if elapsed_days is not None:
            x = x + self.time_proj(elapsed_days.unsqueeze(-1).float())
        return x
