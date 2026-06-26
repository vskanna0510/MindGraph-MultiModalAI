"""Base modality encoder contract."""

from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.encoders.freeze import set_freeze_mode
from ml_pipeline.models.encoders.output import EncoderOutput
from ml_pipeline.models.encoders.projection_head import ProjectionHead


class BaseModalityEncoder(nn.Module):
    """Independent encoder — train, benchmark, export separately."""

    modality: str = "unknown"
    backbone_name: str = "projection"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = int(config.get("latent_dim", 512))
        self.dropout = float(config.get("dropout", 0.1))
        self.pooling_mode = config.get("pooling", "attention")
        self.freeze_mode = config.get("freeze_mode", "full")
        set_freeze_mode(self, self.freeze_mode)

    @abstractmethod
    def encode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> EncoderOutput:
        """Full encode pipeline."""

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        presence: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> EncoderOutput:
        out = self.encode(x, mask, **kwargs)
        if presence is not None:
            out.embedding = out.embedding * presence.unsqueeze(-1)
        return out

    def extract_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutput:
        self.eval()
        with torch.no_grad():
            return self.encode(x, mask)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config, "backbone": self.backbone_name}, path)
        path.with_suffix(".json").write_text(
            json.dumps({"modality": self.modality, "backbone": self.backbone_name}, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load_checkpoint(cls, path: Path, encoder_cls: type) -> BaseModalityEncoder:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = encoder_cls(payload["config"])
        model.load_state_dict(payload["state_dict"])
        return model

    def export(self, path: Path, fmt: str = "torchscript") -> Path | None:
        self.eval()
        try:
            dummy = torch.randn(1, 8, self.config.get("input_dim", 512))
            if fmt == "torchscript":
                traced = torch.jit.trace(self, (dummy, torch.ones(1, 8)))
                traced.save(str(path))
                return path
            if fmt == "onnx":
                torch.onnx.export(self, (dummy, torch.ones(1, 8)), str(path), opset_version=17)
                return path
        except Exception:
            return None
        return None

    def build_refinement(self, dim: int) -> nn.Module:
        return ProjectionHead(dim, dim, dim, self.dropout)
