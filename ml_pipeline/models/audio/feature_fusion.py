"""Attention-based fusion of MFCC, prosody, pitch, wav2vec, speech quality."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioFeatureFusion(nn.Module):
    """Fuse acoustic feature streams with learned attention — no blind concat."""

    def __init__(self, feature_dims: dict[str, int], out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.names = list(feature_dims.keys())
        self.proj = nn.ModuleDict({k: nn.Linear(d, out_dim) for k, d in feature_dims.items()})
        self.attn = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: name → (B, T, D) or (B, D)
        Returns:
            fused (B, T, out_dim), attention weights (B, num_features)
        """
        projected: list[torch.Tensor] = []
        for name in self.names:
            if name not in features:
                continue
            x = features[name]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            projected.append(self.proj[name](x))
        if not projected:
            raise ValueError("No audio features provided for fusion")
        stacked = torch.stack(projected, dim=2)
        scores = self.attn(self.dropout(stacked)).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        return self.norm(fused), weights.mean(dim=1)
