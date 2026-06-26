"""Text projection layers."""

from __future__ import annotations

from ml_pipeline.models.encoders.projection_head import ProjectionHead


class TextInputProjection(ProjectionHead):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__(in_dim, out_dim, out_dim, dropout)
