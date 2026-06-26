"""Load graph platform YAML configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(filename: str) -> dict[str, Any]:
    path = _project_root() / "configs" / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class GraphPlatformConfig:
    """Merged configuration for the knowledge graph platform."""

    def __init__(self) -> None:
        self.neo4j = load_yaml("neo4j.yaml")
        self.graph = load_yaml("graph.yaml")
        self.temporal = load_yaml("temporal.yaml")
        self.analytics = load_yaml("analytics.yaml")
        self.reasoning = load_yaml("reasoning.yaml")
        self.recommendation = load_yaml("recommendation.yaml")
        self.cache = load_yaml("cache.yaml")
        self.twin = load_yaml("twin.yaml")
        self.database = load_yaml("database.yaml")

    @property
    def schema_version(self) -> str:
        return str(self.neo4j.get("schema", {}).get("version", "1.0.0"))

    @property
    def migrations_path(self) -> Path:
        rel = self.neo4j.get("schema", {}).get("migrations_path", "graphs/migrations")
        return _project_root() / rel

    @property
    def platform_migrations_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "schema" / "migrations"

    @property
    def max_subgraph_depth(self) -> int:
        return int(self.neo4j.get("graph", {}).get("max_subgraph_depth", 3))

    @property
    def max_nodes_per_query(self) -> int:
        return int(self.neo4j.get("graph", {}).get("max_nodes_per_query", 500))

    @property
    def auto_create_indexes(self) -> bool:
        return bool(self.neo4j.get("indexes", {}).get("auto_create", True))

    @property
    def pool_size(self) -> int:
        neo = self.database.get("neo4j", {})
        return int(neo.get("max_connection_pool_size", 50))

    @property
    def cache_enabled(self) -> bool:
        return bool(self.cache.get("graph_cache", {}).get("enabled", True))

    @property
    def cache_ttl(self) -> int:
        return int(self.cache.get("graph_cache", {}).get("default_ttl_seconds", 300))


@lru_cache
def get_graph_config() -> GraphPlatformConfig:
    return GraphPlatformConfig()
