"""Base explainability contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseExplainability(nn.Module):
    @abstractmethod
    def explain(self, features: dict[str, torch.Tensor], prediction: torch.Tensor) -> dict[str, Any]:
        """Return explanation artifacts."""
