"""Base model contract."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.device import autocast_context, resolve_device, resolve_dtype
from ml_pipeline.models.types import MultimodalBatch, TensorContract


class BaseModel(nn.Module, ABC):
    """Every model implements the full research/production contract."""

    model_id: str = "MODEL_000"
    version: str = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config or {}
        self._device = resolve_device(self.config.get("device", "auto"))

    @abstractmethod
    def forward(self, batch: MultimodalBatch) -> dict[str, torch.Tensor]:
        """Full forward pass."""

    def predict(self, batch: MultimodalBatch) -> dict[str, torch.Tensor]:
        self.eval()
        dtype = resolve_dtype(self.config.get("mixed_precision", "auto"), self._device)
        with torch.no_grad(), autocast_context(self._device, dtype):
            out = self.forward(batch.to(self._device))
        if "logits" in out:
            out["probs"] = torch.softmax(out["logits"], dim=-1)
            out["preds"] = out["probs"].argmax(dim=-1)
        return out

    def extract_features(self, batch: MultimodalBatch) -> dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            out = self.forward(batch.to(self._device))
        return {k: v for k, v in out.items() if "embedding" in k or k == "fused"}

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.model_id,
            "version": self.version,
            "state_dict": self.state_dict(),
            "config": self.config,
        }
        torch.save(payload, path)
        meta = path.with_suffix(".json")
        meta.write_text(
            json.dumps({"model_id": self.model_id, "version": self.version}, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> BaseModel:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(config=payload.get("config", {}))
        model.load_state_dict(payload["state_dict"])
        return model

    def export(self, path: Path, fmt: str = "torchscript") -> Path | None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.eval()
        dummy = self._dummy_batch()
        try:
            if fmt == "torchscript":
                traced = torch.jit.trace(self._trace_wrapper(), (dummy,))
                traced.save(str(path))
                return path
            if fmt == "onnx":
                torch.onnx.export(
                    self._trace_wrapper(),
                    (dummy,),
                    str(path),
                    opset_version=int(self.config.get("export", {}).get("onnx_opset", 17)),
                    input_names=["batch"],
                    output_names=["logits"],
                    dynamic_axes={"batch": {0: "batch"}},
                )
                return path
        except Exception:
            return None
        return None

    def _trace_wrapper(self) -> nn.Module:
        """Simplified wrapper for export — subclasses may override."""
        return _ExportWrapper(self)

    def _dummy_batch(self) -> MultimodalBatch:
        b, d, t = 1, self.config.get("model", {}).get("latent_dim", 512), 8
        return MultimodalBatch(
            audio=torch.zeros(b, t, d),
            text=torch.zeros(b, t, d),
            audio_mask=torch.ones(b, t),
            text_mask=torch.ones(b, t),
            modality_presence={
                "audio": torch.ones(b),
                "text": torch.ones(b),
                "visual": torch.zeros(b),
                "image": torch.zeros(b),
            },
        )

    def summary(self) -> dict[str, Any]:
        params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_id": self.model_id,
            "version": self.version,
            "parameters": params,
            "trainable": trainable,
        }

    def benchmark(self, batch: MultimodalBatch | None = None, runs: int = 10) -> dict[str, float]:
        batch = batch or self._dummy_batch()
        batch = batch.to(self._device)
        self.eval()
        times: list[float] = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                self.forward(batch)
                if self._device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        return {
            "mean_ms": sum(times) / len(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }

    def validate(self) -> tuple[bool, list[str]]:
        errors: list[str] = []
        try:
            out = self.forward(self._dummy_batch().to(self._device))
            if "logits" not in out:
                errors.append("missing_logits")
        except Exception as exc:
            errors.append(str(exc))
        return len(errors) == 0, errors

    @abstractmethod
    def tensor_contracts(self) -> list[TensorContract]:
        """Document I/O shapes."""


class _ExportWrapper(nn.Module):
    def __init__(self, model: BaseModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: MultimodalBatch) -> torch.Tensor:
        return self.model.forward(batch)["logits"]
