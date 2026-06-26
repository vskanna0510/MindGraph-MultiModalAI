"""Transformer-based multimodal fusion."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.fusion.attention_pooling import FusionAttentionPooling
from ml_pipeline.models.fusion.base_fusion import BaseFusionModule
from ml_pipeline.models.fusion.modality_tokens import ModalityTokens
from ml_pipeline.models.fusion.output import FusionOutput
from ml_pipeline.models.fusion.positional_encoding import PositionalEncoding


class TransformerFusion(BaseFusionModule):
    strategy_name = "transformer"

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(d_model, config)
        cfg = config or {}
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg.get("num_heads", 8)),
            dim_feedforward=int(cfg.get("ffn_dim", d_model * 4)),
            dropout=float(cfg.get("dropout", 0.1)),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg.get("num_layers", 2)))
        self.tokens = ModalityTokens(d_model)
        self.pos = PositionalEncoding(d_model, mode=cfg.get("positional_encoding", "learnable"))
        self.pool = FusionAttentionPooling(d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def fuse(self, embeddings, presence, quality_scores=None) -> FusionOutput:
        device = next(self.parameters()).device
        parts, type_ids, weights = [], [], {}
        id_map = {"audio": 1, "visual": 2, "text": 3, "image": 4}
        b = None
        for name, emb in embeddings.items():
            pres = presence.get(name)
            if pres is not None and pres.sum() == 0:
                continue
            b = emb.shape[0]
            parts.append(emb.unsqueeze(1))
            type_ids.append(id_map.get(name, 0))
            weights[name] = pres if pres is not None else torch.ones(emb.shape[0], device=device)
        if not parts or b is None:
            z = torch.zeros(1, self.d_model, device=device)
            return FusionOutput(fusion_embedding=z)
        cls = self.cls.expand(b, -1, -1)
        stacked = torch.cat([cls] + parts, dim=1)
        stacked = self.tokens.add_type_embedding(stacked[:, 1:], type_ids)
        stacked = torch.cat([cls, stacked], dim=1)
        stacked = self.pos(stacked)
        hidden = self.encoder(stacked)
        pooled, attn = self.pool(hidden)
        return FusionOutput(
            fusion_embedding=pooled,
            attention_maps={"transformer": attn},
            modality_weights=weights,
            hidden_states=hidden,
            confidence=torch.sigmoid(pooled.mean(dim=-1)),
            metadata={"strategy": "transformer"},
        )
