"""Model registry and versioning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ml_pipeline.models.config import model_paths


@dataclass
class ModelRegistryEntry:
    model_id: str
    name: str
    version: str
    architecture: str
    paper: str = ""
    training_dataset: str = ""
    performance: dict = field(default_factory=dict)
    checkpoint: str = ""
    export_status: str = "pending"
    owner: str = "mindgraph"


DEFAULT_REGISTRY = [
    ModelRegistryEntry("MODEL_001", "baseline", "1.0", "baseline_mlp"),
    ModelRegistryEntry("MODEL_002", "audio_encoder", "1.0", "audio_projection"),
    ModelRegistryEntry("MODEL_003", "visual_encoder", "1.0", "vit_projection"),
    ModelRegistryEntry("MODEL_004", "text_encoder", "1.0", "text_projection"),
    ModelRegistryEntry("MODEL_005", "image_encoder", "1.0", "cnn_projection"),
    ModelRegistryEntry("MODEL_006", "fusion", "1.0", "cross_modal_transformer"),
    ModelRegistryEntry("MODEL_007", "temporal", "1.0", "temporal_fusion"),
    ModelRegistryEntry("MODEL_008", "graph", "1.0", "graphsage"),
    ModelRegistryEntry("MODEL_009", "explainable", "1.0", "attribution_engine"),
    ModelRegistryEntry("MODEL_010", "production", "1.0", "mindgraph_multimodal"),
]


class ModelRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or model_paths()["registry"]
        self.entries: dict[str, ModelRegistryEntry] = {}
        if self.path.exists():
            self.load()
        else:
            for e in DEFAULT_REGISTRY:
                self.entries[e.model_id] = e

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for item in data.get("models", []):
            entry = ModelRegistryEntry(**item)
            self.entries[entry.model_id] = entry

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "models": [
                {
                    "model_id": e.model_id,
                    "name": e.name,
                    "version": e.version,
                    "architecture": e.architecture,
                    "paper": e.paper,
                    "training_dataset": e.training_dataset,
                    "performance": e.performance,
                    "checkpoint": e.checkpoint,
                    "export_status": e.export_status,
                    "owner": e.owner,
                }
                for e in self.entries.values()
            ]
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.path

    def get(self, model_id: str) -> ModelRegistryEntry | None:
        return self.entries.get(model_id)

    def register(self, entry: ModelRegistryEntry) -> None:
        self.entries[entry.model_id] = entry
