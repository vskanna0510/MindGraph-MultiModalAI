"""Feature registry — tracks versions, pipelines, and validation status."""

from __future__ import annotations

import json
from pathlib import Path

from ml_pipeline.feature_store.config import feature_store_config, feature_store_paths
from ml_pipeline.feature_store.types import FeatureRegistryEntry


def default_registry() -> list[FeatureRegistryEntry]:
    cfg = feature_store_config()
    versions = cfg.get("feature_store", {}).get("feature_versions", {})
    extraction = cfg.get("feature_store", {}).get("extraction_versions", {})
    models = cfg.get("feature_store", {}).get("model_versions", {})
    paths = feature_store_paths()
    entries = []
    for modality in ("audio", "visual", "text", "image", "graph"):
        entries.append(
            FeatureRegistryEntry(
                name=f"{modality}_embedding",
                version=versions.get(modality, "v1.0"),
                owner="mindgraph",
                description=f"{modality} embedding features",
                dependencies=[f"ml_pipeline.{modality}"],
                extraction_pipeline=f"ml_pipeline.{modality}.pipeline",
                storage_location=str(paths.get(modality, paths["root"] / modality)),
                validation_status="pending",
            )
        )
    return entries


class FeatureRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or feature_store_paths()["root"] / feature_store_config().get("paths", {}).get(
            "registry_file", "feature_registry.json"
        )
        self.entries: dict[str, FeatureRegistryEntry] = {}
        if self.path.exists():
            self.load()
        else:
            for entry in default_registry():
                self.entries[entry.name] = entry

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for item in data.get("features", []):
            entry = FeatureRegistryEntry(**item)
            self.entries[entry.name] = entry

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": [
                {
                    "name": e.name,
                    "version": e.version,
                    "owner": e.owner,
                    "description": e.description,
                    "dependencies": e.dependencies,
                    "extraction_pipeline": e.extraction_pipeline,
                    "storage_location": e.storage_location,
                    "validation_status": e.validation_status,
                    "deprecated": e.deprecated,
                }
                for e in self.entries.values()
            ]
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.path

    def get(self, name: str) -> FeatureRegistryEntry | None:
        return self.entries.get(name)

    def register(self, entry: FeatureRegistryEntry) -> None:
        self.entries[entry.name] = entry

    def mark_validated(self, name: str) -> None:
        if name in self.entries:
            self.entries[name].validation_status = "passed"
