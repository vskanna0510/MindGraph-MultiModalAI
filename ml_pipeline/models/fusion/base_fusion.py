"""Base fusion contract."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.fusion.output import FusionOutput


class BaseFusionModule(nn.Module, ABC):
    """Every fusion strategy implements the full research contract."""

    strategy_name: str = "base"

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.config = config or {}

    @abstractmethod
    def fuse(
        self,
        embeddings: dict[str, torch.Tensor],
        presence: dict[str, torch.Tensor],
        quality_scores: dict[str, torch.Tensor] | None = None,
    ) -> FusionOutput:
        """Primary fusion entry point."""

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        presence: dict[str, torch.Tensor],
        quality_scores: dict[str, torch.Tensor] | None = None,
    ) -> FusionOutput:
        return self.fuse(embeddings, presence, quality_scores)

    def compute_attention(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.softmax(hidden.mean(dim=-1), dim=-1)

    def validate(self) -> tuple[bool, list[str]]:
        errors: list[str] = []
        try:
            b = 2
            d = self.d_model
            embs = {"audio": torch.randn(b, d), "text": torch.randn(b, d)}
            pres = {"audio": torch.ones(b), "text": torch.ones(b), "visual": torch.zeros(b), "image": torch.zeros(b)}
            out = self.fuse(embs, pres)
            if out.fusion_embedding.shape != (b, d):
                errors.append("shape_mismatch")
        except Exception as exc:
            errors.append(str(exc))
        return len(errors) == 0, errors

    def benchmark(self, runs: int = 10) -> dict[str, float]:
        b, d = 4, self.d_model
        embs = {m: torch.randn(b, d) for m in ("audio", "visual", "text")}
        pres = {m: torch.ones(b) for m in embs}
        self.eval()
        times: list[float] = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                self.fuse(embs, pres)
                times.append(time.perf_counter() - start)
        return {"mean_ms": sum(times) / len(times) * 1000, "strategy": self.strategy_name}

    def export(self, path: Path, fmt: str = "torchscript") -> Path | None:
        return None

    def visualize(self, output: FusionOutput, output_dir: Path) -> list[Path]:
        from ml_pipeline.models.fusion.visualization import visualize_fusion

        return visualize_fusion(output, output_dir)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config, "strategy": self.strategy_name}, path)
        return path
