"""Optimizer factory."""

from __future__ import annotations

from typing import Any, Iterable

import torch


def build_optimizer(name: str, params: Iterable[torch.nn.Parameter], config: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(config.get("learning_rate", 1e-4))
    wd = float(config.get("weight_decay", 0.01))
    key = name.lower()

    if key == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    if key == "radam":
        try:
            from torch.optim import RAdam

            return RAdam(params, lr=lr, weight_decay=wd)
        except ImportError:
            pass
    if key == "lion":
        try:
            from lion_pytorch import Lion

            return Lion(params, lr=lr, weight_decay=wd)
        except ImportError:
            pass
    if key == "adabelief":
        try:
            from adabelief_pytorch import AdaBelief

            return AdaBelief(params, lr=lr, weight_decay=wd)
        except ImportError:
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
