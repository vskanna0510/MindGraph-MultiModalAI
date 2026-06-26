"""Dataset adapter registry."""

from __future__ import annotations

from pathlib import Path

from ml_pipeline.datasets.adapters.base import BaseDatasetAdapter
from ml_pipeline.datasets.adapters.daic_woz import DaicWozAdapter
from ml_pipeline.datasets.adapters.dvlog import DvlogAdapter
from ml_pipeline.datasets.config import dataset_paths

ADAPTERS: dict[str, type[BaseDatasetAdapter]] = {
    "daic_woz": DaicWozAdapter,
    "dvlog": DvlogAdapter,
}


def get_adapter(name: str, raw_root: Path | None = None) -> BaseDatasetAdapter:
    """Instantiate adapter by name."""
    if name not in ADAPTERS:
        raise ValueError(f"Unknown dataset adapter: {name}. Available: {list(ADAPTERS)}")
    paths = dataset_paths()
    root = raw_root or (paths["raw"] / name)
    return ADAPTERS[name](root)


def list_adapters() -> list[str]:
    return list(ADAPTERS.keys())
