"""Checkpoint management."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointManager:
    directory: Path
    save_top_k: int = 3
    monitors: dict[str, str] = field(default_factory=lambda: {"val_f1": "max", "val_loss": "min"})
    every_n_epochs: int = 0

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._best: dict[str, float] = {}
        self._scores: list[tuple[float, Path]] = []

    def save(self, model: torch.nn.Module, epoch: int, metrics: dict[str, float], tag: str = "latest") -> Path:
        path = self.directory / f"{tag}_epoch{epoch}.pt"
        if hasattr(model, "save"):
            model.save(path)
        else:
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": metrics}, path)
        meta = self.directory / f"{tag}_epoch{epoch}.json"
        meta.write_text(json.dumps({"epoch": epoch, "metrics": metrics, "tag": tag}, indent=2), encoding="utf-8")
        return path

    def maybe_save_best(self, model: torch.nn.Module, epoch: int, metrics: dict[str, float]) -> list[Path]:
        saved: list[Path] = []
        for key, mode in self.monitors.items():
            if key not in metrics:
                continue
            score = metrics[key]
            prev = self._best.get(key)
            improved = prev is None or (mode == "max" and score > prev) or (mode == "min" and score < prev)
            if improved:
                self._best[key] = score
                path = self.save(model, epoch, metrics, tag=f"best_{key}")
                saved.append(path)
                self._scores.append((score, path))
                self._scores.sort(key=lambda x: x[0], reverse=(mode == "max"))
                while len(self._scores) > self.save_top_k:
                    _, old = self._scores.pop()
                    if old.exists():
                        old.unlink(missing_ok=True)
        return saved

    def load_resume(self, model: torch.nn.Module, path: Path) -> dict[str, Any]:
        if hasattr(model, "load"):
            loaded = type(model).load(path)
            model.load_state_dict(loaded.state_dict())
            payload = torch.load(path, map_location="cpu", weights_only=False)
            return payload
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        return payload
