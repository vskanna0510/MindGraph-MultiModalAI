"""Fusion output contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class FusionOutput:
    fusion_embedding: torch.Tensor
    temporal_embedding: torch.Tensor | None = None
    attention_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    modality_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    quality_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    historical_weights: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
