"""Graph pipeline configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def graph_config() -> dict[str, Any]:
    cfg = _load_yaml(ROOT / "configs" / "graph_pipeline.yaml")
    neo4j = _load_yaml(ROOT / "configs" / "neo4j.yaml")
    model_graph = _load_yaml(ROOT / "configs" / "model" / "graph.yaml")
    return {
        "pipeline": cfg.get("graph_pipeline", {}),
        "storage": cfg.get("storage", {}),
        "neo4j": {**neo4j.get("graph", {}), **cfg.get("neo4j", {}), **neo4j},
        "builders": cfg.get("builders", {}),
        "gnn": {**model_graph, **cfg.get("gnn", {})},
        "embeddings": {**neo4j.get("embeddings", {}), **cfg.get("embeddings", {})},
        "temporal": {**neo4j.get("temporal", {}), **cfg.get("temporal", {})},
        "validation": cfg.get("validation", {}),
        "export": cfg.get("export", {}),
        "benchmark": cfg.get("benchmark", {}),
    }


def graph_paths() -> dict[str, Path]:
    cfg = graph_config()
    root = ROOT / "datasets" / "processed"
    storage = cfg.get("storage", {})
    return {
        "root": root / "graph",
        "cache": Path(storage.get("cache_dir", root / "graph_cache")),
        "snapshots": Path(storage.get("snapshots_dir", root / "graph_snapshots")),
        "exports": root / "graph" / "exports",
    }
