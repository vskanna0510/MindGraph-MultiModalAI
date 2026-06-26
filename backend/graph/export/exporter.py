"""Graph export utilities."""

from __future__ import annotations

import json
from typing import Any


class GraphExporter:
    """Export graph query results (no raw media)."""

    @staticmethod
    def to_json(data: Any) -> str:
        return json.dumps(data, default=str, indent=2)

    @staticmethod
    def to_csv_rows(records: list[dict[str, Any]]) -> str:
        if not records:
            return ""
        headers = list(records[0].keys())
        lines = [",".join(headers)]
        for row in records:
            lines.append(",".join(str(row.get(h, "")) for h in headers))
        return "\n".join(lines)
