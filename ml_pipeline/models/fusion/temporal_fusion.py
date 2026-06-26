"""Temporal fusion and session memory."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.fusion.attention_pooling import FusionAttentionPooling
from ml_pipeline.models.fusion.output import FusionOutput
from ml_pipeline.models.fusion.positional_encoding import TemporalPositionalEncoding
from ml_pipeline.models.fusion.temporal_memory import TemporalMemory


class TemporalFusion(nn.Module):
    """Session sequence → temporal transformer → temporal embedding."""

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.d_model = d_model
        layers = int(cfg.get("num_layers", cfg.get("temporal_layers", 2)))
        self.memory = TemporalMemory(d_model, int(cfg.get("max_sessions", 12)))
        self.pos = TemporalPositionalEncoding(d_model, int(cfg.get("max_sessions", 12)))
        enc_layer = nn.TransformerEncoderLayer(d_model, 8, d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pool = FusionAttentionPooling(d_model)
        self.short = nn.Linear(d_model, d_model)
        self.long = nn.Linear(d_model, d_model)

    def forward(
        self,
        fusion_embedding: torch.Tensor,
        history: torch.Tensor | None = None,
        session_idx: torch.Tensor | None = None,
    ) -> FusionOutput:
        seq = self.memory.build_sequence(fusion_embedding, history)
        b, t, d = seq.shape
        if session_idx is None:
            session_idx = torch.arange(t, device=seq.device).unsqueeze(0).expand(b, -1)
        seq = self.pos(seq, session_idx)
        hidden = self.encoder(seq)
        pooled, attn = self.pool(hidden)
        short = self.short(hidden[:, -1])
        long = self.long(hidden.mean(dim=1))
        temporal = (short + long) / 2
        return FusionOutput(
            fusion_embedding=fusion_embedding,
            temporal_embedding=temporal,
            attention_maps={"temporal": attn},
            historical_weights=attn,
            hidden_states=hidden,
            metadata={"sessions": t},
        )
