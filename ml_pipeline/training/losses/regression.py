"""Regression loss variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_regression_loss(name: str) -> nn.Module:
    name = name.lower()

    class _Loss(nn.Module):
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if name == "huber":
                return F.smooth_l1_loss(pred, target)
            if name == "mae":
                return F.l1_loss(pred, target)
            return F.mse_loss(pred, target)

    return _Loss()
