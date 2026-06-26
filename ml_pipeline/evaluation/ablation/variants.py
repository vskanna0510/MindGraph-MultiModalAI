"""Ablation study definitions."""

from __future__ import annotations

from typing import Any

import torch

ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {"modalities": [], "graph": False, "temporal": False},
    "audio_only": {"modalities": ["audio"], "graph": False, "temporal": False},
    "visual_only": {"modalities": ["visual"], "graph": False, "temporal": False},
    "text_only": {"modalities": ["text"], "graph": False, "temporal": False},
    "image_only": {"modalities": ["image"], "graph": False, "temporal": False},
    "audio_text": {"modalities": ["audio", "text"], "graph": False, "temporal": False},
    "audio_visual": {"modalities": ["audio", "visual"], "graph": False, "temporal": False},
    "visual_text": {"modalities": ["visual", "text"], "graph": False, "temporal": False},
    "three_modalities": {"modalities": ["audio", "visual", "text"], "graph": False, "temporal": False},
    "four_modalities": {"modalities": ["audio", "visual", "text", "image"], "graph": False, "temporal": False},
    "without_graph": {"modalities": ["audio", "visual", "text"], "graph": False, "temporal": True},
    "with_graph": {"modalities": ["audio", "visual", "text"], "graph": True, "temporal": True},
    "without_temporal": {"modalities": ["audio", "visual", "text"], "graph": True, "temporal": False},
    "with_temporal": {"modalities": ["audio", "visual", "text"], "graph": True, "temporal": True},
    "full_architecture": {"modalities": ["audio", "visual", "text", "image"], "graph": True, "temporal": True},
}


def list_ablation_variants() -> list[str]:
    return list(ABLATION_VARIANTS.keys())


def apply_ablation_mask(batch, variant: str):
    """Zero out modalities per ablation variant."""
    cfg = ABLATION_VARIANTS.get(variant, ABLATION_VARIANTS["full_architecture"])
    allowed = set(cfg.get("modalities", []))
    mp = {}
    for mod in ("audio", "visual", "text", "image"):
        mp[mod] = batch.modality_presence.get(mod, torch.zeros(1)) * (1.0 if mod in allowed else 0.0)
    from ml_pipeline.models.types import MultimodalBatch

    meta = dict(batch.metadata or {})
    if not cfg.get("graph", True):
        meta["graph_disabled"] = True
    if not cfg.get("temporal", True):
        meta.pop("session_history", None)
    return MultimodalBatch(
        audio=batch.audio,
        visual=batch.visual,
        text=batch.text,
        image=batch.image,
        graph=None if not cfg.get("graph", True) else batch.graph,
        audio_mask=batch.audio_mask,
        visual_mask=batch.visual_mask,
        text_mask=batch.text_mask,
        modality_presence=mp,
        labels=batch.labels,
        participant_ids=batch.participant_ids,
        session_ids=batch.session_ids,
        languages=batch.languages,
        metadata=meta,
    )
