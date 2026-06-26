"""Base fusion contract."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn

from ml_pipeline.models.types import TensorContract


class BaseFusion(nn.Module):
    d_model: int

    @abstractmethod
    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        presence: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse modality embeddings → fused representation."""


class FeatureProjection(nn.Module):
    """Independent projection into shared latent dimension."""

    def __init__(self, modalities: list[str], in_dims: dict[str, int], latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.ModuleDict(
            {m: nn.Linear(in_dims[m], latent_dim) for m in modalities}
        )

    def forward(self, embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {m: self.proj[m](emb) for m, emb in embeddings.items() if m in self.proj}
