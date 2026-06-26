"""Base graph module contract."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class BaseGraphModule(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """Graph reasoning on fused features."""
