"""Classification loss variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, weight: torch.Tensor | None = None) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def label_smoothing_ce(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1, weight: torch.Tensor | None = None) -> torch.Tensor:
    n_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    loss = (-true_dist * log_probs).sum(dim=-1)
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def build_classification_loss(name: str, class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0, focal_gamma: float = 2.0) -> nn.Module:
    name = name.lower()

    class _Loss(nn.Module):
        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if name == "focal":
                return focal_loss(logits, targets, focal_gamma, class_weights)
            if label_smoothing > 0:
                return label_smoothing_ce(logits, targets, label_smoothing, class_weights)
            return F.cross_entropy(logits, targets, weight=class_weights)

    return _Loss()
