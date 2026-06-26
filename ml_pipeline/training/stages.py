"""Staged training — freeze/unfreeze modules."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


STAGES = ("encoder_warmup", "fusion", "temporal", "graph", "end_to_end")


def apply_training_stage(model: nn.Module, stage: str) -> None:
    """Configure which modules train in each stage."""
    for p in model.parameters():
        p.requires_grad = False

    def _unfreeze(module: nn.Module | None) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = True

    if stage == "encoder_warmup":
        for name in ("audio_encoder", "visual_encoder", "text_encoder", "image_encoder"):
            _unfreeze(getattr(model, name, None))
        _unfreeze(getattr(model, "prediction_head", None))
    elif stage == "fusion":
        _unfreeze(getattr(model, "fusion_stack", None))
        _unfreeze(getattr(model, "heads", None) or getattr(model, "prediction_head", None))
    elif stage == "temporal":
        fs = getattr(model, "fusion_stack", None)
        if fs and hasattr(fs, "temporal"):
            _unfreeze(fs.temporal)
        _unfreeze(getattr(model, "heads", None) or getattr(model, "prediction_head", None))
    elif stage == "graph":
        _unfreeze(getattr(model, "kg_injection", None))
        _unfreeze(getattr(model, "gnn", None))
        _unfreeze(getattr(model, "heads", None) or getattr(model, "prediction_head", None))
    else:
        for p in model.parameters():
            p.requires_grad = True


def stage_for_epoch(epoch: int, stages_cfg: dict[str, Any]) -> str:
    bounds = [
        ("encoder_warmup", int(stages_cfg.get("encoder_warmup_epochs", 0))),
        ("fusion", int(stages_cfg.get("fusion_epochs", 0))),
        ("temporal", int(stages_cfg.get("temporal_epochs", 0))),
        ("graph", int(stages_cfg.get("graph_epochs", 0))),
    ]
    cursor = 0
    for name, length in bounds:
        if length <= 0:
            continue
        if epoch < cursor + length:
            return name
        cursor += length
    return "end_to_end"
