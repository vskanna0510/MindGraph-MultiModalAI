"""Tensor contracts and multimodal batch types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class TensorContract:
    """Documented tensor shape contract for a module."""

    name: str
    input_shape: tuple[int, ...] | str
    output_shape: tuple[int, ...] | str
    dtype: str = "float32"
    has_mask: bool = False
    has_batch: bool = True
    has_sequence: bool = False
    embedding_dim: int = 0


@dataclass
class MultimodalBatch:
    """Immutable multimodal input container for the full pipeline."""

    audio: torch.Tensor | None = None
    visual: torch.Tensor | None = None
    text: torch.Tensor | None = None
    image: torch.Tensor | None = None
    graph: torch.Tensor | None = None
    audio_mask: torch.Tensor | None = None
    visual_mask: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    modality_presence: dict[str, torch.Tensor] = field(default_factory=dict)
    labels: torch.Tensor | None = None
    quality_scores: torch.Tensor | None = None
    participant_ids: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> MultimodalBatch:
        def _mv(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(device) if t is not None else None

        return MultimodalBatch(
            audio=_mv(self.audio),
            visual=_mv(self.visual),
            text=_mv(self.text),
            image=_mv(self.image),
            graph=_mv(self.graph),
            audio_mask=_mv(self.audio_mask),
            visual_mask=_mv(self.visual_mask),
            text_mask=_mv(self.text_mask),
            modality_presence={k: v.to(device) for k, v in self.modality_presence.items()},
            labels=_mv(self.labels),
            quality_scores=_mv(self.quality_scores),
            participant_ids=list(self.participant_ids),
            session_ids=list(self.session_ids),
            languages=list(self.languages),
            splits=list(self.splits),
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_collated(cls, batch: dict[str, Any]) -> MultimodalBatch:
        def _t(key: str) -> torch.Tensor | None:
            v = batch.get(key)
            if v is None:
                return None
            return v if isinstance(v, torch.Tensor) else torch.as_tensor(v)

        presence = {}
        for mod in ("audio", "visual", "text", "image"):
            key = f"{mod}_presence"
            if key in batch:
                presence[mod] = _t(key)  # type: ignore[assignment]
            elif "modality_presence" in batch and mod in batch["modality_presence"]:
                presence[mod] = torch.as_tensor(batch["modality_presence"][mod])

        return cls(
            audio=_t("audio"),
            visual=_t("visual"),
            text=_t("text"),
            image=_t("image"),
            graph=_t("graph"),
            audio_mask=_t("audio_mask"),
            visual_mask=_t("visual_mask"),
            text_mask=_t("text_mask"),
            modality_presence=presence,
            labels=_t("labels"),
            participant_ids=list(batch.get("participant_ids", [])),
            session_ids=list(batch.get("session_ids", [])),
        )
