"""Graph backup utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any


class GraphBackup:
    """Export graph snapshot metadata for backup (not raw media)."""

    @staticmethod
    def create_manifest(nodes: int, edges: int, *, version: str = "2.0.0") -> dict[str, Any]:
        return {
            "backup_version": version,
            "created_at": datetime.now(UTC).isoformat(),
            "node_count": nodes,
            "edge_count": edges,
            "includes_raw_media": False,
        }

    @staticmethod
    def serialize_snapshot(data: dict[str, Any]) -> str:
        return json.dumps(data, default=str, indent=2)
