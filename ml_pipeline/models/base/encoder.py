"""Base encoder contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.types import TensorContract


class BaseEncoder(nn.Module):
    """Modality encoder — projects to shared latent space."""

    modality: str = "unknown"

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        presence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sequence_embedding, pooled_embedding)."""

    def tensor_contract(self) -> TensorContract:
        return TensorContract(
            name=f"{self.modality}_encoder",
            input_shape=("B", "T", self.input_dim),
            output_shape=("B", self.latent_dim),
            has_mask=True,
            has_sequence=True,
            embedding_dim=self.latent_dim,
        )


class ProjectionEncoder(BaseEncoder):
    """MLP + attention pooling encoder for any modality."""

    def __init__(
        self,
        modality: str,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(input_dim, latent_dim, dropout)
        self.modality = modality
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, latent_dim))
        self.projection = nn.Sequential(*layers)
        self.pool_weight = nn.Linear(latent_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        presence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            if x.shape[-1] != self.input_dim:
                x = nn.functional.pad(x, (0, max(0, self.input_dim - x.shape[-1])))
                x = x[..., : self.input_dim].unsqueeze(1)
        if x.shape[-1] != self.input_dim:
            x = nn.functional.linear(
                x,
                torch.eye(min(x.shape[-1], self.input_dim), device=x.device, dtype=x.dtype),
            )
            if x.shape[-1] < self.input_dim:
                x = nn.functional.pad(x, (0, self.input_dim - x.shape[-1]))

        seq = self.projection(x)
        weights = self.pool_weight(seq).squeeze(-1)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(weights, dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), seq).squeeze(1)
        if presence is not None:
            pooled = pooled * presence.unsqueeze(-1)
        return seq, pooled
