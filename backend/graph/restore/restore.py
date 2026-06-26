"""Graph restore utilities."""

from __future__ import annotations

import json
from typing import Any


class GraphRestore:
    """Validate and parse backup manifests."""

    @staticmethod
    def parse_manifest(payload: str) -> dict[str, Any]:
        data = json.loads(payload)
        if "backup_version" not in data:
            raise ValueError("Invalid backup manifest: missing backup_version")
        if data.get("includes_raw_media"):
            raise ValueError("Raw media must not be stored in graph backups")
        return data
