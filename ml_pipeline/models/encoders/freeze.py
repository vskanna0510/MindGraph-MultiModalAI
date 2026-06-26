"""Encoder freezing utilities."""

from __future__ import annotations

import torch.nn as nn


def set_freeze_mode(module: nn.Module, mode: str = "full") -> None:
    """
    Modes: frozen | partial | full | layerwise
    """
    if mode == "frozen":
        for p in module.parameters():
            p.requires_grad = False
        return
    if mode == "full":
        for p in module.parameters():
            p.requires_grad = True
        return
    if mode == "partial":
        for name, p in module.named_parameters():
            p.requires_grad = "projection" in name or "head" in name or "pool" in name
        return
    if mode == "layerwise":
        for p in module.parameters():
            p.requires_grad = True
