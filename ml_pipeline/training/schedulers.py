"""Learning rate scheduler factory."""

from __future__ import annotations

from typing import Any

import torch


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, config: dict[str, Any], steps_per_epoch: int, total_epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
    key = name.lower()
    warmup = int(config.get("warmup_epochs", 0))
    min_lr = float(config.get("min_lr", 1e-6))

    if key == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup, 1), eta_min=min_lr)
    if key == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    if key == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.get("learning_rate", 1e-4), steps_per_epoch=steps_per_epoch, epochs=total_epochs)
    if key == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if key == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_epochs, power=0.9)
    if key == "warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(total_epochs // 4, 1))
    return None
