"""Shared encoder output contract (MP3 Part 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class EncoderOutput:
    """Every encoder returns traceable outputs — no anonymous tensors."""

    embedding: torch.Tensor
    hidden_states: torch.Tensor | None = None
    attention_maps: torch.Tensor | None = None
    intermediate: dict[str, torch.Tensor] = field(default_factory=dict)
    confidence: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pooled(self) -> torch.Tensor:
        return self.embedding
